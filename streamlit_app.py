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
IMG_SIZE          = (256, 256)
CANNY_LOW         = 30
CANNY_HIGH        = 100
HOUGH_THRESHOLD   = 60
HOUGH_MIN_LEN     = 40
HOUGH_MAX_GAP     = 10
CLASS_NAMES       = ["fractured", "not fractured"]
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
    colors = {
        "sobel": "#e74c3c",
        "canny": "#3498db",
        "hough": "#2ecc71",
    }
    feats = {k: v for k, v in feats.items() if not k.startswith("ws_")}
    vals  = np.array(list(feats.values()), dtype=float)
    names = list(feats.keys())
    vmax  = np.abs(vals).max() + 1e-8
    norm  = vals / vmax
    idx   = np.argsort(np.abs(norm))[::-1][:15]
    top_n = [names[i] for i in idx]
    top_v = [norm[i]  for i in idx]
    bar_c = [
        next((c for p, c in colors.items() if n.startswith(p)), "#95a5a6")
        for n in top_n
    ]
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
# Sample image helpers
# ─────────────────────────────────────────────────────────────
def get_sample_files(folder: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for ext in exts:
        files.extend(sorted(folder.glob(ext)))
    return files


# ─────────────────────────────────────────────────────────────
# Algorithm explanation section
# ─────────────────────────────────────────────────────────────
def show_algorithm_explanations():
    st.markdown("---")
    st.markdown("## 📖 Algorithm Explanations")
    st.markdown(
        "Each classical computer vision technique below contributes a group of features "
        "that the Random Forest classifier uses to decide whether a bone is fractured."
    )

    # ── Sobel ────────────────────────────────────────────────
    with st.expander("🔴 Sobel — Gradient Edge Detection  *(14 features)*", expanded=False):
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            st.markdown("""
**What it does**

The Sobel operator slides two small 3×3 kernels across the image — one that
detects *horizontal* intensity changes (Gx) and one that detects *vertical*
changes (Gy). Combining them gives the **gradient magnitude**:

> **M = √(Gx² + Gy²)**

Bright pixels in the magnitude map mean a large, sudden change in brightness —
i.e., an edge, boundary, or surface discontinuity.

**Why it matters for fractures**

A fractured bone shows a sharp break in the cortical (outer) shell that
creates very high gradient values along the fracture line. Intact bone has
smooth, gently-varying brightness. The pipeline summarises the entire magnitude
map into 14 statistics to capture this difference.

**Key signal → `sobel_high_ratio`**
The fraction of pixels with magnitude above *mean + 2σ*. Fractured bones
typically show a higher proportion of strong-gradient pixels because the
fracture line acts like an additional strong edge inside the bone.

**All 14 features**
`sobel_mean` · `sobel_std` · `sobel_max`
· `sobel_p25/50/75/90/95`
· `sobel_energy` · `sobel_entropy`
· `sobel_high_ratio` · `sobel_dir_std`
· `sobel_horiz_energy` · `sobel_vert_energy`
            """)
        with col2:
            st.markdown("""
**The two Sobel kernels**

```
Gx (horizontal)    Gy (vertical)
 -1   0  +1         -1  -2  -1
 -2   0  +2          0   0   0
 -1   0  +1         +1  +2  +1
```

Each output pixel = dot product of the
kernel with the 3×3 neighbourhood.

**Where it sits in the pipeline**

```
Original image
  ↓ CLAHE
  ↓ Bilateral filter
  ↓ Gaussian blur   ← reduces noise
  ↓ Sobel           ← applied here
```

The Gaussian blur before Sobel is
critical — without it, noise produces
many false high-gradient pixels.
            """)
            st.info(
                "**Colour in pipeline view:** hot colourmap — "
                "white/yellow = high gradient, black = flat region.",
                icon="🎨",
            )

    # ── Canny ────────────────────────────────────────────────
    with st.expander("🔵 Canny — Multi-Stage Edge Detection  *(9 features)*", expanded=False):
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            st.markdown("""
**What it does**

Canny builds on Sobel but adds two extra steps to produce clean,
**thin, single-pixel-wide edges** rather than thick blobs:

1. **Gaussian smoothing** — suppress noise before gradient computation
2. **Sobel gradients** — compute Gx, Gy, magnitude, and direction
3. **Non-maximum suppression** — keep only the pixel with the *highest* gradient
   in the direction perpendicular to the edge, discarding its neighbours
4. **Double-threshold hysteresis** — pixels above `HIGH` are definite edges;
   pixels below `LOW` are discarded; pixels in between are kept only if they
   touch a definite edge

Parameters used in this pipeline: **low = 30, high = 100**.

**Why it matters for fractures**

The resulting binary edge map is fed to `findContours`, which traces connected
groups of edge pixels. A fractured bone produces **many short, disconnected
contours** (fragments of the fracture line) rather than the few long, smooth
contours you see around intact bone shafts.

**Key signal → `canny_small_contour_ratio`**
The fraction of contours with perimeter < 10 px. Fractures scatter edge
pixels into tiny isolated fragments, raising this ratio significantly.

**All 9 features**
`canny_edge_density` · `canny_edge_count`
· `canny_contour_count`
· `canny_mean/std/max_contour_len`
· `canny_small_contour_ratio`
· `canny_convexity_defect` · `canny_edge_variance`
            """)
        with col2:
            st.markdown("""
**Threshold logic**

```
gradient ≥ HIGH (100)
  → definite edge ✓

gradient < LOW  (30)
  → not an edge  ✗

LOW ≤ gradient < HIGH
  → edge only if connected
    to a definite edge ↔
```

Hysteresis prevents weak-gradient
edges from creating noise while
still letting them extend along
real boundaries.

**After Canny**

The binary edge map is passed
directly into the Hough Transform
for line detection.
            """)
            st.info(
                "**Colour in pipeline view:** grayscale — "
                "white pixels = detected edges, black = background.",
                icon="🎨",
            )

    # ── Hough ────────────────────────────────────────────────
    with st.expander("🟢 Hough Transform — Line Orientation Analysis  *(9 features)*", expanded=False):
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            st.markdown("""
**What it does**

The **Probabilistic Hough Transform** (`cv2.HoughLinesP`) looks at the Canny
edge map and votes for line segments in polar parameter space (ρ, θ).
Every edge pixel "votes" for all lines it could belong to; peaks in the
accumulator reveal actual line segments.

Only segments that pass minimum criteria are returned:
- Accumulator threshold: **60** votes
- Minimum line length: **40 px**
- Maximum allowed gap inside a line: **10 px**

**Why it matters for fractures**

Long bones naturally have edges running roughly *parallel* to the bone's
long axis. A fracture creates additional edges that run **perpendicular**
(or at odd angles) to the long axis — the fracture plane itself. This
raises both the variance of detected angles and the fraction of lines
that are nearly perpendicular to the dominant direction.

**Key signals**
- **`hough_perpendicular_ratio`** — fraction of lines deviating ≥ 45° from
  the dominant angle; strongly elevated in fractured images.
- **`hough_std_angle`** — high angular spread suggests disordered edges
  consistent with a fracture disrupting normal bone structure.
- **`hough_angle_entropy`** — uniform angle distribution (high entropy)
  means lines point in many directions, a fracture indicator.

**All 9 features**
`hough_line_count` · `hough_mean/std_length`
· `hough_mean/std_angle` · `hough_angle_entropy`
· `hough_dominant_angle` · `hough_perpendicular_ratio`
· `hough_short_line_ratio`
            """)
        with col2:
            st.markdown("""
**Parameter trade-offs**

```
threshold ↑
  → fewer but more
    confident lines

min_length ↑
  → eliminates short stubs
    (noise artefacts)

max_gap ↓
  → won't bridge
    over real breaks
```

**Angle convention**

Angles are in **[0°, 90°]** —
the absolute angle of each
segment relative to horizontal,
regardless of direction.

**Dominant angle**

The histogram bin with the most
lines. All other lines are compared
against this to compute
`hough_perpendicular_ratio`.
            """)
            st.info(
                "**Colour in pipeline view:** detected lines drawn in green "
                "over the CLAHE-enhanced image.",
                icon="🎨",
            )

    # ── Watershed ────────────────────────────────────────────
    with st.expander("🟠 Watershed — Bone Region Segmentation  *(10 features)*", expanded=False):
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            st.markdown("""
**What it does**

Watershed treats the **distance-transformed** binary image as a topographic
landscape — pixels far from any background become mountain peaks, pixels near
the background become valleys. The algorithm then "floods" water from local
peaks (seeds) until two flood fronts collide, drawing a **watershed boundary**
at the collision point.

Steps in this pipeline:

1. **Otsu thresholding** on bilateral-filtered image → binary bone/background mask
2. **Morphological opening** (erosion + dilation) → remove tiny noise specks
3. **Distance transform** → each foreground pixel gets its distance to the nearest
   background pixel (peaks = bone centres)
4. **Sure foreground** — pixels with distance > 0.5 × max → definite bone cores
5. **Connected components** → label each bone core as a separate seed marker
6. **`cv2.watershed()`** → grow each seed outward until boundaries are found;
   boundary pixels are marked **−1**

**Why it matters for fractures**

An intact bone typically forms **one or two large, compact regions**. A fracture
physically separates the bone into pieces, causing the segmentation to produce
**more, smaller regions** with irregular (less compact) shapes. The boundary
pixels (marked −1, shown in red) also increase in density at fracture sites.

**Key signals**
- **`ws_region_count`** — more regions → more fragmentation → likely fracture.
- **`ws_small_region_ratio`** — fraction of regions with area < 200 px²; small
  fragments accumulate around fracture lines.
- **`ws_compactness_mean`** — compact (round) shapes score near 1; irregular
  fracture fragments score much higher.

**All 10 features**
`ws_region_count`
· `ws_mean/std/max/min_region_area`
· `ws_area_ratio` · `ws_boundary_mean`
· `ws_compactness_mean` · `ws_small_region_ratio`
· `ws_region_entropy`
            """)
        with col2:
            st.markdown("""
**Distance transform intuition**

```
Binary mask (bone = white):
  0 0 0 0 0
  0 1 1 1 0
  0 1 1 1 0   → distance from
  0 1 1 1 0      background
  0 0 0 0 0

Distance values:
  0 0 0 0 0
  0 1 1 1 0
  0 1 2 1 0   ← peak = 2
  0 1 1 1 0
  0 0 0 0 0
```

The peak becomes a seed — the
algorithm grows outward from it.

**Colour in pipeline view**

```
Each segmented region → random colour
Watershed boundary   → red
Background           → black
```

**Note on sensitivity**

Watershed is sensitive to noise.
CLAHE + bilateral filter before
this step are essential to avoid
over-segmentation (too many tiny
spurious regions).
            """)
            st.info(
                "**Colour in pipeline view:** each bone region is a unique random colour; "
                "red pixels mark the watershed boundaries.",
                icon="🎨",
            )


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
        show_explanations = st.checkbox("Show algorithm explanations", value=True)
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

    # ── Image input ───────────────────────────────────────────
    tab_upload, tab_sample = st.tabs(["📤 Upload Your Own Image", "🖼️ Use a Sample Image"])

    pil_image   = None
    image_label = ""

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload an X-ray image (.png, .jpg, .jpeg)",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded is not None:
            pil_image   = Image.open(uploaded)
            image_label = f"Uploaded: {uploaded.name}"

    with tab_sample:
        fractured_files     = get_sample_files(FRACTURED_DIR)
        not_fractured_files = get_sample_files(NOT_FRACTURED_DIR)

        if not fractured_files and not not_fractured_files:
            st.warning(
                "No sample images found. Make sure the `Fractured/` and "
                "`Not Fractured/` folders exist next to `streamlit_app.py`."
            )
        else:
            col_cat, col_sel = st.columns([1, 2], gap="large")
            with col_cat:
                category = st.radio(
                    "Category",
                    ["Fractured", "Not Fractured"],
                    help="Pick the type of sample you want to simulate.",
                )
            folder_files = fractured_files if category == "Fractured" else not_fractured_files

            if not folder_files:
                st.warning(f"No images found in the `{category}` folder.")
            else:
                with col_sel:
                    selected_sample = st.selectbox(
                        "Select image",
                        folder_files,
                        format_func=lambda p: p.name,
                    )

                if selected_sample is not None:
                    col_thumb, col_info = st.columns([1, 2], gap="large")
                    with col_thumb:
                        st.image(str(selected_sample), caption=selected_sample.name, width=240)
                    with col_info:
                        badge_color = "#c0392b" if category == "Fractured" else "#27ae60"
                        badge_label = category.upper()
                        st.markdown(
                            f"<span style='background:{badge_color};color:white;"
                            f"padding:3px 10px;border-radius:4px;font-size:13px;"
                            f"font-weight:600'>{badge_label}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**File:** `{selected_sample.name}`  \n"
                            f"**Folder:** `{selected_sample.parent.name}/`"
                        )
                        st.markdown(
                            "_Switch to this tab and select an image to analyze it "
                            "automatically — no upload needed._"
                        )

                    # Only use sample if nothing was uploaded
                    if pil_image is None:
                        pil_image   = Image.open(str(selected_sample))
                        image_label = f"Sample ({category}): {selected_sample.name}"

    if pil_image is None:
        st.info("👆 Upload an X-ray image or pick a sample from the **Use a Sample Image** tab.", icon="ℹ️")
        if show_explanations:
            show_algorithm_explanations()
        return

    # ── Image preview + prediction ────────────────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        source_icon = "🗂️" if image_label.startswith("Sample") else "📤"
        st.markdown(f"#### {source_icon} X-ray Image")
        st.image(pil_image, use_column_width=True, clamp=True)
        st.caption(
            f"{image_label}  |  "
            f"Size: {pil_image.size[0]}×{pil_image.size[1]} px  |  Mode: {pil_image.mode}"
        )

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
            </div>""", unsafe_allow_html=True)

    # ── Algorithm explanations ────────────────────────────────
    if show_explanations:
        show_algorithm_explanations()

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
