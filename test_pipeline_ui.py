"""Run with: python test_pipeline_ui.py"""
import base64
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from streamlit_app import build_pipeline_images, extract_all_features, FRACTURED_DIR, get_sample_files


def test_reference_colors():
    """Fixed display values from the original pipeline, independent of image analysis."""
    gray = np.array([[0, 64, 128], [192, 224, 255]], dtype=np.uint8)
    edges = np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8)
    markers = np.array([[-1, 0, 1], [2, 2, 3]], dtype=np.int32)
    with (
        patch("streamlit_app.preprocess_image", return_value=dict.fromkeys(["gray", "clahe", "blurred", "bilateral"], gray)),
        patch("streamlit_app.extract_sobel_features", return_value=({}, gray.astype(float))),
        patch("streamlit_app.extract_canny_features", return_value=({}, edges)),
        patch("streamlit_app.extract_hough_features", return_value=({}, np.array([[0, 0, 0, 0]]))),
        patch("streamlit_app.extract_watershed_features", return_value=({}, markers)),
    ):
        maps = build_pipeline_images(Image.new("RGB", (3, 2)))
    pixels = [np.array(Image.open(io.BytesIO(base64.b64decode(m["src"].split(",")[1])))) for m in maps]
    np.testing.assert_array_equal(pixels[0], gray)
    np.testing.assert_array_equal(pixels[1], [
        [[0, 0, 0], [56, 55, 77], [112, 123, 143]],
        [[168, 199, 199], [212, 227, 227], [255, 255, 255]]])
    np.testing.assert_array_equal(pixels[2], [
        [[10, 0, 0], [178, 0, 0], [255, 91, 0]],
        [[255, 255, 6], [255, 255, 132], [255, 255, 255]]])
    np.testing.assert_array_equal(pixels[3], edges)
    hough = np.repeat(gray[..., None], 3, axis=2)
    hough[0, 0] = [80, 220, 0]
    np.testing.assert_array_equal(pixels[4], hough)
    np.testing.assert_array_equal(pixels[5], [
        [[255, 50, 50], [0, 0, 0], [0, 0, 0]],
        [[225, 184, 159], [225, 184, 159], [112, 120, 67]]])


def test_pipeline_ui():
    sample = get_sample_files(FRACTURED_DIR)[0]
    for image in (Image.new("RGB", (64, 64)), Image.open(sample)):
        features = extract_all_features(image)
        maps = build_pipeline_images(image)
        assert len(features) == 42 and np.isfinite(list(features.values())).all()
        assert [stage["name"] for stage in maps] == [
            "Original", "CLAHE", "Sobel gradient", "Canny edges", "Hough lines", "Watershed"]
        for stage in maps:
            decoded = Image.open(io.BytesIO(base64.b64decode(stage["src"].split(",")[1])))
            assert decoded.size == (256, 256)
    app = AppTest.from_file(str(Path(__file__).with_name("streamlit_app.py"))).run(timeout=30)
    assert not app.exception, app.exception
    assert len(app.get("iframe")) == 1
    trailer = app.get("video")  # Plays on its own below the hero, no button needed.
    assert len(trailer) == 1 and trailer[0].proto.autoplay and trailer[0].proto.muted
    assert any(".play()" in html.proto.body and html.proto.unsafe_allow_javascript for html in app.get("html"))
    app.run()  # The trailer stays on the page alongside the workspace across reruns.
    assert not app.exception and len(app.get("video")) == 1
    assert not app.get("video")[0].proto.loop
    app.toggle(key="loop_trailer").set_value(True).run()
    assert not app.exception and app.get("video")[0].proto.loop
    app.toggle(key="loop_trailer").set_value(False).run()
    assert not app.exception and not app.get("video")[0].proto.loop
    assert app.toggle(key="animate_pipeline").value
    app.toggle(key="animate_pipeline").set_value(False).run()
    assert not app.exception
    assert any("animation-play-state:paused!important" in item.value for item in app.markdown)
    app.toggle(key="animate_pipeline").set_value(True).run()
    assert any("animation-play-state:running!important" in item.value for item in app.markdown)
    assert app.slider[0].label == "Fracture threshold" and app.slider[0].value == 50
    assert "Analysis settings" not in [expander.label for expander in app.expander]
    with patch("sklearn.ensemble.RandomForestClassifier.predict_proba", return_value=np.array([[0.6, 0.4]])):
        for threshold, label in [(55, "Fractured"), (60, "Fractured"), (65, "Not fractured")]:
            app.slider[0].set_value(threshold).run()
            assert not app.exception
            result = next(item.value for item in app.markdown if 'class="result"' in item.value)
            assert f"<h2>{label}</h2>" in result
            assert "60.0%" in result  # The cutoff changes the decision, not the model probability.
        app.button(key="reset_threshold").click().run()
        assert not app.exception and app.slider[0].value == 50
        result = next(item.value for item in app.markdown if 'class="result"' in item.value)
        assert result.index("<h2>Fractured</h2>") < result.index("Model prediction")
        assert "60.0%" in result
    original_frame = app.get("iframe")[0].proto.srcdoc
    app.selectbox[0].select("Not Fractured").run()
    assert not app.exception
    assert app.get("iframe")[0].proto.srcdoc != original_frame
    app.radio[0].set_value("Upload").run()
    assert not app.exception
    assert len(app.get("iframe")) == 0  # No hidden sample shown as an uploaded image.
    for data, valid in [(sample.read_bytes(), True), (b"invalid PNG", False)]:
        upload = io.BytesIO(data)
        upload.name = "uploaded.png"
        with patch("streamlit.file_uploader", return_value=upload):
            app.run()
        assert not app.exception
        assert len(app.get("iframe")) == int(valid)
        assert bool(app.error) != valid
    app.radio[0].set_value("Sample").run()
    assert not app.exception and len(app.get("iframe")) == 1


if __name__ == "__main__":
    test_reference_colors()
    test_pipeline_ui()
    print("Reference colors, pipeline maps, uploads and source-switching checks passed")
