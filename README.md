# 🦴 Bone Fracture Detector

A classical computer vision web app for bone fracture detection — **no deep learning**.  
Built with OpenCV, scikit-learn, and Streamlit.

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bone-fracture-detection4.streamlit.app)

---

## How It Works

Every uploaded X-ray passes through a 4-stage CV pipeline that extracts **42 hand-crafted features**, which are fed into a **Random Forest classifier**.

```
X-ray image
    │
    ├─ CLAHE  →  Bilateral Filter  →  Gaussian Blur
    │
    ├── [1] Sobel Operators      → 14 gradient magnitude features
    ├── [2] Canny Edge Detector  →  9 edge structure features
    ├── [3] Hough Transform      →  9 line orientation features
    └── [4] Watershed Algorithm  → 10 bone segmentation features
                                        │
                                  Random Forest
                                        │
                             fractured / not fractured
```

| Feature Block | What it detects |
|---|---|
| **Sobel** | Sharp intensity changes at fracture lines |
| **Canny** | Thin edge fragments around crack sites |
| **Hough** | Lines perpendicular to the bone axis (cracks) |
| **Watershed** | Fragmented bone segments after a break |

---

## Repository Structure

```
├── streamlit_app.py                 # Main Streamlit application
├── pipeline_viewer.html             # Stage-by-stage image viewer embedded in the app
├── Random_Forest.pkl                # Trained classifier (4.7 MB, committed)
├── bone-fracture-detection.ipynb    # Feature extraction, training and model comparison (Kaggle)
├── test_canny_features.py           # Checks for the Canny feature block
├── test_pipeline_ui.py              # Checks for the pipeline viewer
├── assets/app-trailer.mp4           # Trailer shown on first visit
└── requirements.txt
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/geraldadli/bone_fracture.git
cd bone_fracture

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
streamlit run streamlit_app.py
```

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (`Random_Forest.pkl` is already committed).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and set **Main file** to `streamlit_app.py`.
4. Click **Deploy**.

---

## Model Training

The classifier was trained on the  
[Bone Fracture Multi-Region X-ray dataset](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data) on Kaggle.

Training notebook: `bone-fracture-detection.ipynb`. It extracts the 42 features from 9,246 training
images and compares three classifiers on the same features:

| Model | CV accuracy (5-fold) | Test accuracy | Test AUC |
|-------|----------------------|---------------|----------|
| SVM (RBF) | 0.9925 ± 0.0014 | 0.9960 | 0.9980 |
| **Random Forest** (used in the app) | **0.9931 ± 0.0015** | **1.0000** | **1.0000** |
| Gradient Boosting | 0.9852 ± 0.0016 | 0.9822 | 0.9994 |

Test set: 506 images (238 fractured, 268 not). These scores are on this dataset's own splits, and
the dataset contains rotated copies of the same X-rays, so expect lower accuracy on X-rays from
elsewhere. Research project, not a diagnostic tool.

---

## Tech Stack

- **OpenCV** — image preprocessing & feature extraction  
- **scikit-learn** — Random Forest, SVM, Gradient Boosting  
- **Streamlit** — web interface  
- **SciPy** — entropy computation  
- **Matplotlib** — pipeline visualisations