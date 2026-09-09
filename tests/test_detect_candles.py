# tests/test_detect_candles.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "feature_extraction"))

from detect_candles import detect_candles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_IMAGE = os.path.join(BASE_DIR, "..", "dataset", "images", "Hammer")


def get_any_sample_image():
    files = [f for f in os.listdir(SAMPLE_IMAGE) if f.endswith(".png")]
    assert files, "No sample images found -- did you run generate_images.py?"
    return os.path.join(SAMPLE_IMAGE, files[0])


def test_detects_expected_candle_count():
    image_path = get_any_sample_image()
    candles = detect_candles(image_path)
    assert len(candles) == 6, f"Expected 6 candles, got {len(candles)}"


def test_body_and_wicks_sum_to_total_range():
    image_path = get_any_sample_image()
    candles = detect_candles(image_path)
    for c in candles:
        computed_total = c["body_height_px"] + c["upper_wick_px"] + c["lower_wick_px"]
        # allow small pixel rounding tolerance
        assert abs(computed_total - c["total_range_px"]) <= 2