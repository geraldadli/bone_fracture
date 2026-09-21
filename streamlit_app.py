"""
streamlit_app.py
================
Bone Fracture Detection — Classical CV Pipeline
Deployed with Streamlit. Loads Random_Forest.pkl from the same directory.

Directory layout expected:
    ├── streamlit_app.py
    ├── Random_Forest.pkl
    ├── Fractured/          ← sample X-rays with fractures
    ├── Not Fractured/      ← sample X-rays without fractures
    └── requirements.txt
"""

import io
import base64
from html import escape
import json
import warnings
import numpy as np
import cv2
import joblib
import streamlit as st
import streamlit.components.v1 as components
from matplotlib import colormaps
from matplotlib.colors import Normalize

from pathlib import Path
from PIL import Image
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bone Fracture Detector",
    page_icon="✚",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
IMG_SIZE          = (256, 256)
CANNY_LOW         = 30
CANNY_HIGH        = 100
HOUGH_THRESHOLD   = 60
HOUGH_MIN_LEN     = 40
HOUGH_MAX_GAP     = 10
MODEL_PATH        = Path(__file__).parent / "Random_Forest.pkl"
FRACTURED_DIR     = Path(__file__).parent / "Fractured"
NOT_FRACTURED_DIR = Path(__file__).parent / "Not Fractured"


# ─────────────────────────────────────────────────────────────
# Load model (cached — loads once per session)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found: `{MODEL_PATH}`\n\n"
            "Make sure `Random_Forest.pkl` is in the same folder as `streamlit_app.py`."
        )
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: `{e}`")
        st.stop()


# ─────────────────────────────────────────────────────────────
# CV Pipeline — identical logic to the Kaggle notebook
# ─────────────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image, size: tuple = IMG_SIZE) -> dict:
    img_rgb   = pil_image.convert("RGB")
    img_bgr   = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    img_bgr   = cv2.resize(img_bgr, size)
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = clahe_obj.apply(gray)
    bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    blurred   = cv2.GaussianBlur(bilateral, (5, 5), 0)
    return {"gray": gray, "clahe": enhanced, "bilateral": bilateral, "blurred": blurred}


def extract_sobel_features(blurred):
    sobelx    = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely    = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(np.abs(sobely), np.abs(sobelx))
    mag_flat  = magnitude.ravel()
    mean_mag  = mag_flat.mean()
    std_mag   = mag_flat.std()
    mag_norm  = (mag_flat - mag_flat.min()) / (mag_flat.max() - mag_flat.min() + 1e-8)
    hist, _   = np.histogram(mag_norm, bins=50, density=True)
    high_mask = magnitude > (mean_mag + 2 * std_mag)
    feats = {
        "sobel_mean": mean_mag, "sobel_std": std_mag,
        "sobel_max": mag_flat.max(),
        "sobel_p25": np.percentile(mag_flat, 25),
        "sobel_p50": np.percentile(mag_flat, 50),
        "sobel_p75": np.percentile(mag_flat, 75),
        "sobel_p90": np.percentile(mag_flat, 90),
        "sobel_p95": np.percentile(mag_flat, 95),
        "sobel_energy": float(np.sum(magnitude**2)),
        "sobel_entropy": float(scipy_entropy(hist + 1e-10)),
        "sobel_high_ratio": float(high_mask.sum() / high_mask.size),
        "sobel_dir_std": float(direction.std()),
        "sobel_horiz_energy": float(np.sum(sobelx**2)),
        "sobel_vert_energy": float(np.sum(sobely**2)),
    }
    return feats, magnitude


def extract_canny_features(blurred, low=CANNY_LOW, high=CANNY_HIGH):
    edges        = cv2.Canny(blurred, low, high)
    edge_pixels  = int(edges.sum() / 255)
    contours, _  = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    perimeters   = [cv2.arcLength(c, closed=False) for c in contours] or [0.0]
    small_ratio  = sum(1 for p in perimeters if p < 10) / max(len(perimeters), 1)
    defect_depths = []
    for cnt in contours:
        if len(cnt) >= 5:
            hull = cv2.convexHull(cnt, returnPoints=False)
            if hull is not None and len(hull) > 3:
                try:
                    defects = cv2.convexityDefects(cnt, hull)
                    if defects is not None:
                        defect_depths.extend((defects.reshape(-1, 4)[:, 3] / 256.0).tolist())
                except cv2.error:
                    pass
    edge_coords = np.argwhere(edges > 0)
    feats = {
        "canny_edge_density": edge_pixels / edges.size,
        "canny_edge_count": edge_pixels,
        "canny_contour_count": len(contours),
        "canny_mean_contour_len": float(np.mean(perimeters)),
        "canny_std_contour_len": float(np.std(perimeters)),
        "canny_max_contour_len": float(np.max(perimeters)),
        "canny_small_contour_ratio": small_ratio,
        "canny_convexity_defect": float(np.mean(defect_depths)) if defect_depths else 0.0,
        "canny_edge_variance": float(edge_coords.var()) if len(edge_coords) > 1 else 0.0,
    }
    return feats, edges


def extract_hough_features(edges, threshold=HOUGH_THRESHOLD,
                            min_length=HOUGH_MIN_LEN, max_gap=HOUGH_MAX_GAP):
    zero = {"hough_line_count": 0, "hough_mean_length": 0.0, "hough_std_length": 0.0,
            "hough_mean_angle": 0.0, "hough_std_angle": 0.0, "hough_angle_entropy": 0.0,
            "hough_dominant_angle": 0.0, "hough_perpendicular_ratio": 0.0,
            "hough_short_line_ratio": 0.0}
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                            minLineLength=min_length, maxLineGap=max_gap)
    if lines is None:
        return zero, None
    lines   = lines.reshape(-1, 4)
    dx      = lines[:, 2] - lines[:, 0]
    dy      = lines[:, 3] - lines[:, 1]
    lengths = np.sqrt(dx**2 + dy**2)
    angles  = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))
    hist, bin_edges = np.histogram(angles, bins=18, range=(0, 90))
    hist_n  = (hist.astype(float) + 1e-10); hist_n /= hist_n.sum()
    dom_bin = np.argmax(hist)
    dom_ang = float((bin_edges[dom_bin] + bin_edges[dom_bin + 1]) / 2)
    feats = {
        "hough_line_count": len(lines),
        "hough_mean_length": float(lengths.mean()),
        "hough_std_length": float(lengths.std()),
        "hough_mean_angle": float(angles.mean()),
        "hough_std_angle": float(angles.std()),
        "hough_angle_entropy": float(scipy_entropy(hist_n)),
        "hough_dominant_angle": dom_ang,
        "hough_perpendicular_ratio": float((np.abs(angles - dom_ang) >= 45).sum() / len(lines)),
        "hough_short_line_ratio": float((lengths < np.median(lengths)).sum() / len(lines)),
    }
    return feats, lines


def extract_watershed_features(clahe, bilateral):
    zero = {k: 0.0 for k in ["ws_region_count", "ws_mean_region_area", "ws_std_region_area",
            "ws_max_region_area", "ws_min_region_area", "ws_area_ratio",
            "ws_boundary_mean", "ws_compactness_mean", "ws_small_region_ratio", "ws_region_entropy"]}
    _, thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist    = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return zero, np.zeros_like(clahe)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg  = np.uint8(sure_fg)
    unknown  = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1; markers[unknown == 255] = 0
    markers  = cv2.watershed(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR), markers)

    region_labels = [l for l in np.unique(markers) if l > 1]
    areas, compactnesses = [], []
    for lbl in region_labels:
        mask = (markers == lbl).astype(np.uint8) * 255
        area = int(mask.sum() / 255); areas.append(area)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts and area > 0:
            p = cv2.arcLength(cnts[0], closed=True)
            if p > 0: compactnesses.append((p**2) / (4 * np.pi * area))
    if not areas: areas = [0]
    max_area = max(areas); other = sum(areas) - max_area
    bv = clahe[(markers == -1).astype(np.uint8) == 1]
    ah, _ = np.histogram(areas, bins=20); ah = ah.astype(float) + 1e-10
    feats = {
        "ws_region_count": len(region_labels),
        "ws_mean_region_area": float(np.mean(areas)),
        "ws_std_region_area": float(np.std(areas)),
        "ws_max_region_area": float(max_area),
        "ws_min_region_area": float(min(areas)),
        "ws_area_ratio": float(max_area / (other + 1e-6)),
        "ws_boundary_mean": float(bv.mean()) if len(bv) > 0 else 0.0,
        "ws_compactness_mean": float(np.mean(compactnesses)) if compactnesses else 0.0,
        "ws_small_region_ratio": float(sum(1 for a in areas if a < 200) / len(areas)),
        "ws_region_entropy": float(scipy_entropy(ah / ah.sum())),
    }
    return feats, markers


def extract_all_features(pil_image: Image.Image) -> dict:
    stages = preprocess_image(pil_image)
    sf, _        = extract_sobel_features(stages["blurred"])
    cf, edges    = extract_canny_features(stages["blurred"])
    hf, _        = extract_hough_features(edges)
    wf, _        = extract_watershed_features(stages["clahe"], stages["bilateral"])
    return {**sf, **cf, **hf, **wf}


# Display maps use the same processing functions as the classifier.
def build_pipeline_images(pil_image: Image.Image) -> list[dict]:
    stages = preprocess_image(pil_image)
    _, magnitude = extract_sobel_features(stages["blurred"])
    _, edges = extract_canny_features(stages["blurred"])
    _, lines = extract_hough_features(edges)
    _, markers = extract_watershed_features(stages["clahe"], stages["bilateral"])
    hough = cv2.cvtColor(stages["clahe"], cv2.COLOR_GRAY2RGB)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(hough, (x1, y1), (x2, y2), (80, 220, 0), 1)
    watershed = np.zeros((*markers.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    for label in np.unique(markers):
        if label > 1:
            watershed[markers == label] = rng.integers(60, 255, 3)
    watershed[markers == -1] = [255, 50, 50]
    # Match the original Matplotlib pipeline's colormaps and per-image normalization.
    clahe = colormaps["bone"](Normalize()(stages["clahe"]), bytes=True)[..., :3]
    sobel = colormaps["hot"](Normalize()(magnitude), bytes=True)[..., :3]
    maps = [
        ("Original", "Resized grayscale X-ray", stages["gray"]),
        ("CLAHE", "Local contrast enhancement · bone colormap", clahe),
        ("Sobel gradient", "Gradient strength · black to red to yellow to white", sobel),
        ("Canny edges", "Edge contours after noise reduction", edges),
        ("Hough lines", "Detected line segments in green", hough),
        ("Watershed", "Colored regions · red boundaries · black background", watershed),
    ]
    result = []
    for name, description, pixels in maps:
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        result.append({"name": name, "description": description,
                       "src": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")})
    return result


def get_sample_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def show_pipeline_diagram():
    with st.container(key="pipeline_graph", border=True):
        st.markdown('<div class="graph-heading"><div><div class="eyebrow">THE PROCESS</div><h3>From X-ray to prediction</h3></div><span class="graph-badge">42 image features</span></div>', unsafe_allow_html=True)
        animate = st.toggle("Animate flow", value=True, key="animate_pipeline")
        play_state = "running" if animate else "paused"
        st.markdown(f'<style>.st-key-pipeline_graph .node path,.st-key-pipeline_graph .node polygon,.st-key-pipeline_graph .edge path,.st-key-pipeline_graph .edge polygon {{animation-play-state:{play_state}!important}}</style>', unsafe_allow_html=True)
        st.graphviz_chart("""digraph {
            graph [rankdir=TB bgcolor="transparent" pad="0.2" nodesep="0.38" ranksep="0.65"
                   splines=ortho compound=true fontname="Arial"]
            node [shape=box style="rounded,filled" fillcolor="#ffffff" color="#c9dcd9"
                  penwidth=1.3 fontname="Arial" fontsize=16 fontcolor="#123b4a"
                  margin="0.23,0.18" width=1.65 height=0.8]
            edge [color="#8ba8ad" penwidth=1.5 arrowsize=0.65]
            subgraph cluster_prepare {
                label="01   PREPARE THE IMAGE" labelloc=t labeljust=l
                fontcolor="#59757c" fontsize=12 style="rounded,filled" color="#e6ece7"
                fillcolor="#f5f6f2" margin=22
                {rank=same; input; clahe; bilateral; gaussian}
                input [class="flow-step-0" label="X-ray\\nGrayscale" fillcolor="#e7efed"]
                clahe [class="flow-step-1" label="CLAHE\\nContrast"]
                bilateral [class="flow-step-2" label="Bilateral\\nDenoise"]
                gaussian [class="flow-step-3" label="Gaussian\\nSmooth"]
                input -> clahe [class="flow-step-1"]
                clahe -> bilateral [class="flow-step-2"]
                bilateral -> gaussian [class="flow-step-3"]
            }
            subgraph cluster_extract {
                label="02   EXTRACT IMAGE FEATURES" labelloc=t labeljust=l
                fontcolor="#59757c" fontsize=12 style="rounded,filled" color="#e6ece7"
                fillcolor="#f5f6f2" margin=22
                {rank=same; ws; sobel; canny; hough}
                ws [class="flow-step-7" label="Watershed\\n10 region features" color="#cb997b"]
                sobel [class="flow-step-4" label="Sobel\\n14 gradient features" color="#92b0b8"]
                canny [class="flow-step-5" label="Canny\\n9 edge features" color="#92b0b8"]
                hough [class="flow-step-6" label="Hough\\n9 line features" color="#92b0b8"]
                ws -> sobel -> canny [style=invis]
                canny -> hough [class="flow-step-6"]
            }
            subgraph cluster_predict {
                label="03   CLASSIFY" labelloc=t labeljust=l
                fontcolor="#59757c" fontsize=12 style="rounded,filled" color="#e6ece7"
                fillcolor="#f5f6f2" margin=22
                {rank=same; features; model; result}
                features [class="flow-step-8" label="42 features\\nCombined vector" fillcolor="#e1efeb" color="#adcbc4"]
                model [class="flow-step-9" label="Random Forest\\nClassifier" fillcolor="#07516a" color="#07516a" fontcolor=white]
                result [class="flow-step-10" label="Prediction\\nFractured / not fractured" fillcolor="#e1efeb" color="#adcbc4"]
                features -> model [class="flow-step-9" color="#07516a"]
                model -> result [class="flow-step-10" color="#07516a"]
            }
            clahe -> ws [class="flow-step-7"]
            bilateral -> ws [class="flow-step-7"]
            gaussian -> sobel [class="flow-step-4"]
            gaussian -> canny [class="flow-step-5"]
            ws -> features [class="flow-step-8"]
            sobel -> features [class="flow-step-8"]
            canny -> features [class="flow-step-8"]
            hough -> features [class="flow-step-8"]
        }""", use_container_width=True)


def main():
    st.markdown("""<style>
        .block-container {max-width:none;width:100%;padding:4rem clamp(16px,2vw,36px) 2rem}
        .st-key-analysis_workspace [data-testid="stHorizontalBlock"] {gap:24px}
        .st-key-analysis_workspace [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:first-child {flex:0 0 clamp(260px,23vw,340px);min-width:0}
        .st-key-analysis_workspace [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child {flex:1 1 0;min-width:0}
        .st-key-analysis_workspace iframe {height:clamp(560px,75vh,850px)!important;width:100%}
        .st-key-analysis_workspace [data-testid="stElementContainer"]:has(>iframe) {height:auto}
        @media(max-width:900px) {
            .st-key-analysis_workspace [data-testid="stHorizontalBlock"] {flex-direction:column}
            .st-key-analysis_workspace [data-testid="stHorizontalBlock"]>[data-testid="stColumn"] {flex:1 1 auto!important;width:100%!important}
        }
        h1,h2,h3 {color:#073d50;letter-spacing:-.035em}
        h1 {font-size:2.65rem!important;padding:0!important}
        h3 {font-size:1.15rem!important}
        .brand {display:flex;align-items:center;gap:12px;margin-bottom:28px}
        .cross {background:#073d50;color:white;border-radius:12px;padding:7px 13px;font-size:27px}
        .brand-name {font-weight:750;letter-spacing:.12em;font-size:13px}
        .brand-sub {color:#657e84;font-size:12px;margin-top:3px}
        .research {margin-left:auto;border:1px solid #d8ddd7;border-radius:20px;padding:6px 12px;font-size:12px;color:#647779}
        .intro {color:#647779;margin:10px 0 28px;font-size:16px}
        .eyebrow {font-size:11px;letter-spacing:.14em;font-weight:700;color:#6a8185;margin:8px 0}
        .result {border-radius:16px;background:white;border:1px solid #d8ddd7;border-top:4px solid var(--accent);padding:22px;margin-top:20px}
        .result h2 {font-size:1.6rem;margin:8px 0 14px;padding:0;color:var(--accent)}
        .probability {display:flex;justify-content:space-between;font-size:13px;margin-top:18px}
        .meter {height:7px;border-radius:8px;background:#eceee9;margin:10px 0 0;overflow:hidden}
        .meter span {height:100%;display:block;background:var(--accent);border-radius:8px}
        .file-label {font-size:12px;color:#6a8185;overflow-wrap:anywhere;margin-top:18px}
        .empty {height:390px;display:grid;place-content:center;text-align:center;border:1px dashed #adbfbe;border-radius:20px;background:#ebe7dd;color:#647779}
        .empty strong {font-size:22px;color:#073d50;margin-bottom:10px}
        [data-testid="stFileUploader"] {border-radius:14px}
        [data-testid="stExpander"] {background:rgba(255,255,255,.55)}
        .st-key-threshold_control {background:linear-gradient(135deg,#fff 30%,#edf4f1);border-color:#cbded8!important;border-radius:16px!important;padding:18px!important;box-shadow:0 5px 20px #073d5006}
        .threshold-heading {display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:14px;font-weight:650}
        .threshold-heading strong {font-size:27px;letter-spacing:-.05em;color:#07516a;line-height:1.2;animation:threshold-pop .3s ease-out}
        .threshold-heading small {font-size:14px;margin-left:2px}
        .threshold-scale {display:flex;justify-content:space-between;font-size:11px;color:#526d74;margin-top:-9px}
        .threshold-help {font-size:12px;line-height:1.5;color:#526d74;margin:2px 0 0}
        .st-key-threshold_control :is([role="slider"],[data-rac]:has(>div>input[type="range"])) {width:20px!important;height:20px!important;background:#07516a!important;border:3px solid white!important;box-shadow:0 0 0 2px #07516a,0 3px 8px #07516a33;transition:box-shadow .2s ease,scale .2s ease}
        .st-key-threshold_control :is([role="slider"],[data-rac]:has(>div>input[type="range"])):is(:hover,:focus-within) {scale:1.15;animation:threshold-pulse 1.5s ease-out infinite}
        .st-key-threshold_control [data-rac][role="group"]>[data-rac]>div:first-child {height:7px!important;border-radius:10px;overflow:hidden}
        .st-key-threshold_control [data-rac][role="group"]>[data-rac]>div:first-child:after {content:"";position:absolute;inset:0;background:linear-gradient(100deg,transparent,#ffffff99,transparent);transform:translateX(-100%);pointer-events:none}
        .st-key-threshold_control:hover [data-rac][role="group"]>[data-rac]>div:first-child:after {animation:slider-shine 1.8s ease-in-out infinite}
        .st-key-threshold_control [data-testid="stSliderThumbValue"] {opacity:0;transition:opacity .15s}
        .st-key-threshold_control [data-rac]:is(:hover,:focus-within)>[data-testid="stSliderThumbValue"] {opacity:1}
        .st-key-threshold_control [data-testid="stSliderTickBar"] {visibility:hidden}
        .st-key-pipeline_graph {background:#fff;border:1px solid #d6e1db!important;border-radius:20px!important;padding:24px!important;box-shadow:0 8px 30px #123b4a05}
        .graph-heading {display:flex;align-items:center;justify-content:space-between;gap:14px}
        .graph-heading h3 {padding:0;margin:0;font-size:1.4rem!important}
        .graph-heading .eyebrow {margin:0 0 7px;color:#617c82}
        .graph-badge {border:1px solid #c5dcd5;border-radius:8px;background:#eef5f1;padding:8px 12px;color:#376a62;font-size:12px;white-space:nowrap}
        [data-testid="stGraphVizChart"] {padding:8px 0;width:100%}
        [data-testid="stGraphVizChart"] svg {width:100%!important;min-width:0!important;max-width:100%;height:auto!important}
        [data-testid="stGraphVizChart"] .node text {font-size:16px!important}
        [data-testid="stGraphVizChart"] .cluster text {font-size:12px!important}
        [data-testid="stGraphVizChart"] .node {transition:filter .2s ease}
        [data-testid="stGraphVizChart"] .node:hover {filter:drop-shadow(0 3px 4px #07516a26)}
        .st-key-pipeline_graph .flow-step-0 {--flow-delay:0s;--node-fill:#e7efed}
        .st-key-pipeline_graph .flow-step-1 {--flow-delay:2s}
        .st-key-pipeline_graph .flow-step-2 {--flow-delay:4s}
        .st-key-pipeline_graph .flow-step-3 {--flow-delay:6s}
        .st-key-pipeline_graph .flow-step-4 {--flow-delay:8s}
        .st-key-pipeline_graph .flow-step-5 {--flow-delay:10s}
        .st-key-pipeline_graph .flow-step-6 {--flow-delay:12s}
        .st-key-pipeline_graph .flow-step-7 {--flow-delay:14s;--node-stroke:#cb997b}
        .st-key-pipeline_graph .flow-step-8 {--flow-delay:16s;--node-fill:#e1efeb}
        .st-key-pipeline_graph .flow-step-9 {--flow-delay:18s;--node-fill:#07516a;--highlight-fill:#087c87}
        .st-key-pipeline_graph .flow-step-10 {--flow-delay:20s;--node-fill:#e1efeb}
        .st-key-pipeline_graph .node :is(path,polygon) {animation:flow-node 22s ease-in-out var(--flow-delay) infinite}
        .st-key-pipeline_graph .edge path {animation:flow-edge 22s linear calc(var(--flow-delay) - .4s) infinite}
        .st-key-pipeline_graph .edge polygon {animation:flow-arrow 22s ease-in-out calc(var(--flow-delay) - .4s) infinite}
        @keyframes flow-node {
            0%,9.09%,100% {fill:var(--node-fill,#fff);stroke:var(--node-stroke,#c9dcd9);stroke-width:1.3;filter:none}
            1.5%,7% {fill:var(--highlight-fill,#d2f0e5);stroke:#0b9980;stroke-width:3;filter:drop-shadow(0 0 6px #0b99804d)}
        }
        @keyframes flow-edge {
            0%,9.09%,100% {stroke:#8ba8ad;stroke-width:1.5;stroke-dasharray:7 0;stroke-dashoffset:0}
            1.5%,7% {stroke:#0b9980;stroke-width:3;stroke-dasharray:7 5}
            8% {stroke-dashoffset:-48}
        }
        @keyframes flow-arrow {0%,9.09%,100%{fill:#8ba8ad;stroke:#8ba8ad}1.5%,7%{fill:#0b9980;stroke:#0b9980}}
        @keyframes slider-shine {to{transform:translateX(100%)}}
        @keyframes threshold-pop {from{opacity:.4;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
        @keyframes threshold-pulse {0%{box-shadow:0 0 0 2px #07516a,0 0 0 0 #07516a30}100%{box-shadow:0 0 0 2px #07516a,0 0 0 10px #07516a00}}
        @media(prefers-reduced-motion:reduce) {.threshold-heading strong,.st-key-threshold_control *,.st-key-threshold_control :after,[data-testid="stGraphVizChart"] :is(path,polygon){animation:none!important;transition:none!important}}
        @media(max-width:640px) {.st-key-pipeline_graph{padding:16px!important}.graph-heading{align-items:flex-start;flex-direction:column}.graph-badge{font-size:11px;padding:6px 9px}}
        @media(max-width:640px) {.block-container{padding-top:4rem}h1{font-size:2rem!important}.research{display:none}}
        </style>
<div class="brand"><div class="cross" aria-hidden="true">✚</div>
<div><div class="brand-name">BONE FRACTURE DETECTOR</div><div class="brand-sub">Computer vision imaging workspace</div></div>
<span class="research">Research project</span></div>
<h1>X-ray analysis</h1>
<p class="intro">Review a prediction and explore the image behind it.</p>""", unsafe_allow_html=True)

    model = load_model()
    with st.container(key="analysis_workspace"):
        controls, viewer = st.columns([1, 3], gap="medium")
    pil_image = None
    image_label = ""
    with controls:
        st.markdown("### 01 / Select an X-ray")
        source = st.radio("Image source", ["Sample", "Upload"], horizontal=True, label_visibility="collapsed")
        if source == "Upload":
            uploaded = st.file_uploader("X-ray image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
            if uploaded is not None:
                try:
                    pil_image = Image.open(uploaded)
                    pil_image.load()
                    image_label = uploaded.name
                except (OSError, ValueError, Image.DecompressionBombError):
                    pil_image = None
                    st.error("This image could not be opened. Choose a valid PNG or JPEG.")
        else:
            category = st.selectbox("Sample group", ["Fractured", "Not Fractured"])
            files = get_sample_files(FRACTURED_DIR if category == "Fractured" else NOT_FRACTURED_DIR)
            if files:
                selected = st.selectbox("Image", files, format_func=lambda p: p.name)
                try:
                    pil_image = Image.open(selected)
                    pil_image.load()
                    image_label = selected.name
                except (OSError, ValueError, Image.DecompressionBombError):
                    pil_image = None
                    st.error("This sample could not be opened. Choose another image.")
            else:
                st.info("No samples available in this group. Upload an X-ray to begin.")
        with st.container(key="threshold_control", border=True):
            cutoff = st.session_state.get("threshold_percent", 50)
            st.markdown(f'<div class="threshold-heading"><span>Fracture threshold</span><strong>{cutoff}<small>%</small></strong></div>', unsafe_allow_html=True)
            threshold = st.slider("Fracture threshold", 10, 90, 50, 5, format="%d%%",
                                  key="threshold_percent", label_visibility="collapsed") / 100
            st.markdown('<div class="threshold-scale"><span>More sensitive</span><span>More selective</span></div>', unsafe_allow_html=True)
            st.markdown('<p class="threshold-help">Predict fractured at or above this probability.</p>', unsafe_allow_html=True)

    if pil_image is None:
        with viewer:
            st.markdown('<div class="empty"><strong>Your X-ray workspace</strong><span>Upload an image to explore its processing stages.</span></div>', unsafe_allow_html=True)
        show_pipeline_diagram()
        return

    with st.spinner("Analyzing X-ray…"):
        try:
            feats = extract_all_features(pil_image)
            X = np.array(list(feats.values()), dtype=np.float32).reshape(1, -1)
            prob = model.predict_proba(X)[0]
            p_frac = float(prob[0])
            pred = 0 if p_frac >= threshold else 1
            maps = build_pipeline_images(pil_image)
        except Exception as e:
            st.error(f"Analysis could not be completed: {e}")
            return

    with controls:
        accent = "#b6491a" if pred == 0 else "#267466"
        label = "Fractured" if pred == 0 else "Not fractured"
        st.markdown(f"""<div class="result" style="--accent:{accent}">
            <div class="eyebrow">MODEL PREDICTION</div><h2>{label}</h2>
            <div class="probability"><span>Fracture probability</span><strong>{p_frac:.1%}</strong></div>
            <div class="meter"><span style="width:{p_frac * 100:.2f}%"></span></div>
            <div class="file-label">{escape(image_label)}</div></div>""", unsafe_allow_html=True)
        st.caption("Research output. Not a clinical diagnosis.")
        with st.expander("Feature data"):
            st.dataframe({"Feature": list(feats), "Value": [float(v) for v in feats.values()]}, hide_index=True, use_container_width=True)
            st.download_button("Download JSON", json.dumps({k: float(v) for k, v in feats.items()}, indent=2),
                               "features.json", "application/json")

    with viewer:
        st.markdown("### 02 / Explore the image")
        template = Path(__file__).with_name("pipeline_viewer.html").read_text(encoding="utf-8")
        components.html(template.replace("__PIPELINE_DATA__", json.dumps(maps)), height=700, scrolling=False)
    show_pipeline_diagram()


if __name__ == "__main__":
    main()
