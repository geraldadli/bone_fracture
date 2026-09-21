# 🦴 Bone Fracture Detector

A classical computer vision web app for bone fracture detection — **no deep learning**.  
Built with OpenCV, scikit-learn, and Streamlit.

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

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
├── streamlit_app.py      # Main Streamlit application
├── Random_Forest.pkl     # Trained classifier (download from Kaggle run)
├── requirements.txt      # Python dependencies
└── README.md
```

> **Note:** `Random_Forest.pkl` is not tracked by Git (see `.gitignore`).  
> Download it from your Kaggle notebook output and place it in the root directory.

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the model file
# Copy Random_Forest.pkl into the project root

# 4. Launch
streamlit run streamlit_app.py
```

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (make sure `Random_Forest.pkl` is included).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and set **Main file** to `streamlit_app.py`.
4. Click **Deploy**.

> ⚠️ `Random_Forest.pkl` must be committed to the repo for Streamlit Cloud to find it.  
> If the file exceeds GitHub's 100 MB limit, use [Git LFS](https://git-lfs.github.com/).

---

## Model Training

The classifier was trained on the  
[Bone Fracture Multi-Region X-ray dataset](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data) on Kaggle.

Training notebook: `bone_fracture_opencv.ipynb`

| Metric | Value |
|--------|-------|
| CV Accuracy (5-fold) | reported in notebook |
| Test Accuracy | reported in notebook |
| Test AUC | reported in notebook |

---

## Tech Stack

- **OpenCV** — image preprocessing & feature extraction  
- **scikit-learn** — Random Forest, SVM, Gradient Boosting  
- **Streamlit** — web interface  
- **SciPy** — entropy computation  
- **Matplotlib** — pipeline visualisations