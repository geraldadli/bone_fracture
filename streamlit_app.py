"""
streamlit_app.py
================
Bone Fracture Detection — Classical CV Pipeline
Deployed with Streamlit. Loads Random_Forest.pkl from the same directory.

Directory layout expected:
    ├── streamlit_app.py
    ├── Random_Forest.pkl
    └── requirements.txt
"""

import io
import json
import warnings
import numpy as np
import cv2
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bone Fracture Detector",
    page_icon="🦴",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
IMG_SIZE        = (256, 256)
CANNY_LOW       = 30
CANNY_HIGH      = 100
HOUGH_THRESHOLD = 60
HOUGH_MIN_LEN   = 40
HOUGH_MAX_GAP   = 10
CLASS_NAMES     = ["fractured", "not fractured"]
MODEL_PATH      = Path(__file__).parent / "Random_Forest.pkl"


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
                        defect_depths.extend((defects[:, 0, 3] / 256.0).tolist())
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


# ─────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────
def build_pipeline_figure(pil_image: Image.Image) -> plt.Figure:
    stages = preprocess_image(pil_image)
    _, sobel_mag   = extract_sobel_features(stages["blurred"])
    _, edges       = extract_canny_features(stages["blurred"])
    _, hough_lines = extract_hough_features(edges)
    _, ws_markers  = extract_watershed_features(stages["clahe"], stages["bilateral"])

    hough_vis = cv2.cvtColor(stages["clahe"], cv2.COLOR_GRAY2BGR)
    if hough_lines is not None:
        for x1, y1, x2, y2 in hough_lines:
            cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 220, 80), 1)
    hough_vis = cv2.cvtColor(hough_vis, cv2.COLOR_BGR2RGB)

    ws_vis = np.zeros((*ws_markers.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    for lbl in np.unique(ws_markers):
        if lbl <= 1: continue
        ws_vis[ws_markers == lbl] = rng.integers(60, 255, 3)
    ws_vis[ws_markers == -1] = [255, 50, 50]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.patch.set_facecolor("#0f1117")
    panels = [
        (stages["clahe"], "CLAHE",          "bone"),
        (sobel_mag,       "Sobel Gradient", "hot"),
        (edges,           "Canny Edges",    "gray"),
        (hough_vis,       "Hough Lines",    None),
        (ws_vis,          "Watershed",      None),
    ]
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, **( {"cmap": cmap} if cmap else {} ))
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
        ax.set_facecolor("#0f1117")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
    plt.tight_layout(pad=0.4)
    return fig


def build_feature_bar(feats: dict) -> plt.Figure:
    colors = {"sobel": "#e74c3c", "canny": "#3498db", "hough": "#2ecc71", "ws": "#f39c12"}
    vals  = np.array(list(feats.values()), dtype=float)
    names = list(feats.keys())
    vmax  = np.abs(vals).max() + 1e-8
    norm  = vals / vmax
    idx   = np.argsort(np.abs(norm))[::-1][:15]
    top_n = [names[i] for i in idx]
    top_v = [norm[i]  for i in idx]
    bar_c = [next((c for p, c in colors.items() if n.startswith(p)), "#95a5a6") for n in top_n]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.barh(top_n[::-1], top_v[::-1], color=bar_c[::-1], edgecolor="none", height=0.65)
    ax.set_xlabel("Normalised value", color="#aaa", fontsize=9)
    ax.set_title("Top 15 Feature Values", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#aaa", labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#333")
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.grid(axis="x", color="#222", linewidth=0.5)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────
def main():
    model = load_model()

    st.markdown("""
        <h1 style='text-align:center;color:#f0f0f0;margin-bottom:0'>🦴 Bone Fracture Detector</h1>
        <p style='text-align:center;color:#888;margin-top:4px;font-size:15px'>
            Classical Computer Vision · No Deep Learning ·
            Sobel · Canny · Hough · Watershed · Random Forest
        </p>
        <hr style='border-color:#333;margin:16px 0 24px 0'>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        threshold = st.slider(
            "Classification threshold", 0.10, 0.90, 0.50, 0.05,
            help="Probability above this → 'not fractured'. Below → 'fractured'."
        )
        show_pipeline = st.checkbox("Show CV pipeline breakdown", value=True)
        show_features = st.checkbox("Show feature bar chart",     value=True)
        st.markdown("---")
        st.markdown("### 📋 Pipeline\n"
                    "1. **CLAHE** — local contrast boost\n"
                    "2. **Bilateral filter** — edge-preserving denoise\n"
                    "3. **Sobel** — gradient magnitude (14 features)\n"
                    "4. **Canny** — edge map (9 features)\n"
                    "5. **Hough** — line orientation (9 features)\n"
                    "6. **Watershed** — bone segmentation (10 features)\n"
                    "7. **Random Forest** — 42-feature classifier")
        st.markdown("---")
        st.markdown("### 📁 Model")
        if MODEL_PATH.exists():
            st.success(f"`Random_Forest.pkl` loaded\n{MODEL_PATH.stat().st_size/1024:.0f} KB")
        else:
            st.error("`Random_Forest.pkl` not found")

    # ── Upload ────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload an X-ray image (.png, .jpg, .jpeg)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded is None:
        st.info("👆 Upload a bone X-ray image to get started.", icon="ℹ️")
        return

    pil_image = Image.open(uploaded)
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### Uploaded X-ray")
        st.image(pil_image, use_column_width=True, clamp=True)
        st.caption(f"Size: {pil_image.size[0]}×{pil_image.size[1]} px  |  Mode: {pil_image.mode}")

    # ── Predict ───────────────────────────────────────────────
    with st.spinner("Running CV pipeline…"):
        try:
            feats  = extract_all_features(pil_image)
            X      = np.array(list(feats.values()), dtype=np.float32).reshape(1, -1)
            prob   = model.predict_proba(X)[0]
            p_frac = float(prob[0])
            p_nfrac= float(prob[1])
            pred   = 0 if p_frac >= threshold else 1
            label  = CLASS_NAMES[pred]
            conf   = p_frac if pred == 0 else p_nfrac
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

    # ── Result ────────────────────────────────────────────────
    with col_result:
        st.markdown("#### Prediction")
        if pred == 0:
            st.error(f"### 🔴 FRACTURED\nConfidence: **{conf:.1%}**")
        else:
            st.success(f"### 🟢 NOT FRACTURED\nConfidence: **{conf:.1%}**")

        c1, c2 = st.columns(2)
        c1.metric("Fractured",     f"{p_frac:.1%}")
        c2.metric("Not Fractured", f"{p_nfrac:.1%}")
        st.progress(p_frac, text=f"Fracture probability: {p_frac:.1%}")

        st.markdown("---")
        st.markdown("**Key signal features**")
        signal_feats = {
            "Sobel high-gradient ratio" : round(feats["sobel_high_ratio"],           4),
            "Canny edge density"        : round(feats["canny_edge_density"],         4),
            "Canny small contours"      : round(feats["canny_small_contour_ratio"],  4),
            "Hough perpendicular ratio" : round(feats["hough_perpendicular_ratio"],  4),
            "Hough angle std"           : round(feats["hough_std_angle"],            4),
            "Watershed region count"    : round(feats["ws_region_count"],            0),
            "Watershed small regions"   : round(feats["ws_small_region_ratio"],      4),
        }
        for k, v in signal_feats.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:3px 0;border-bottom:1px solid #222'>"
                f"<span style='color:#aaa;font-size:13px'>{k}</span>"
                f"<span style='color:#f0f0f0;font-size:13px;font-weight:600'>{v}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── CV pipeline viz ───────────────────────────────────────
    if show_pipeline:
        st.markdown("---")
        st.markdown("#### 🔬 CV Pipeline Breakdown")
        with st.spinner("Rendering pipeline…"):
            fig_pipe = build_pipeline_figure(pil_image)
        st.pyplot(fig_pipe, use_container_width=True)
        plt.close(fig_pipe)

    # ── Feature bar ───────────────────────────────────────────
    if show_features:
        st.markdown("---")
        col_bar, col_info = st.columns([2, 1], gap="large")
        with col_bar:
            st.markdown("#### 📊 Feature Values")
            with st.spinner("Rendering features…"):
                fig_feat = build_feature_bar(feats)
            st.pyplot(fig_feat, use_container_width=True)
            plt.close(fig_feat)
        with col_info:
            st.markdown("#### 🏷️ Feature Groups")
            st.markdown("""
            <div style='font-size:13px;line-height:2.2'>
                <span style='color:#e74c3c'>■</span> Sobel — gradient magnitude<br>
                <span style='color:#3498db'>■</span> Canny — edge structure<br>
                <span style='color:#2ecc71'>■</span> Hough — line orientation<br>
                <span style='color:#f39c12'>■</span> Watershed — bone segments
            </div>""", unsafe_allow_html=True)

    # ── Download features ─────────────────────────────────────
    st.markdown("---")
    st.download_button(
        label="⬇️ Download raw feature vector (JSON)",
        data=json.dumps({k: float(v) for k, v in feats.items()}, indent=2),
        file_name="features.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()