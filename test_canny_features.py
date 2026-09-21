"""Run with: python test_canny_features.py"""
from unittest.mock import patch

import numpy as np

from streamlit_app import extract_canny_features


def test_defect_shapes():
    contour = np.array([[0, 0], [8, 0], [4, 4], [8, 8], [0, 8]], dtype=np.int32)
    defects = np.array([[0, 2, 1, 256], [2, 4, 3, 768]], dtype=np.int32)
    with patch("cv2.findContours", return_value=([contour], None)):
        for result, expected in [(defects, 2.0), (defects[:, None, :], 2.0), (None, 0.0)]:
            with patch("cv2.convexityDefects", return_value=result):
                features, _ = extract_canny_features(np.zeros((16, 16), dtype=np.uint8))
                assert features["canny_convexity_defect"] == expected


if __name__ == "__main__":
    test_defect_shapes()
    print("Canny defect shape checks passed")
