# tests/test_performance.py
import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "prediction"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "feature_extraction"))

from predict import load_model_and_scaler, predict_from_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_IMAGE = os.path.join(BASE_DIR, "..", "dataset", "images", "Hammer")


def test_prediction_completes_within_time_limit():
    model, scaler = load_model_and_scaler("RandomForest")
    files = [f for f in os.listdir(SAMPLE_IMAGE) if f.endswith(".png")]
    image_path = os.path.join(SAMPLE_IMAGE, files[0])

    start = time.time()
    predict_from_image(image_path, model, scaler)
    elapsed = time.time() - start

    print(f"\nPrediction took {elapsed:.3f} seconds")
    assert elapsed < 2.0, f"Prediction too slow: {elapsed:.3f}s (NFR1 requires < 2s)"