# tests/test_predict_edge_cases.py
import sys
import os
import pytest
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "prediction"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "feature_extraction"))

from predict import load_model_and_scaler, predict_from_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_test_files")
os.makedirs(TEMP_DIR, exist_ok=True)


@pytest.fixture(scope="module")
def model_and_scaler():
    return load_model_and_scaler("RandomForest")


def test_missing_file_raises_error(model_and_scaler):
    model, scaler = model_and_scaler
    fake_path = os.path.join(TEMP_DIR, "does_not_exist.png")
    with pytest.raises(FileNotFoundError):
        predict_from_image(fake_path, model, scaler)


def test_corrupted_file_raises_error(model_and_scaler):
    model, scaler = model_and_scaler
    corrupted_path = os.path.join(TEMP_DIR, "corrupted.png")
    with open(corrupted_path, "w") as f:
        f.write("this is not actually an image")  # garbage bytes

    with pytest.raises(FileNotFoundError):
        predict_from_image(corrupted_path, model, scaler)


def test_blank_image_raises_error(model_and_scaler):
    model, scaler = model_and_scaler
    blank_path = os.path.join(TEMP_DIR, "blank.png")
    blank_img = np.ones((300, 300, 3), dtype=np.uint8) * 255  # pure white image
    cv2.imwrite(blank_path, blank_img)

    with pytest.raises(ValueError):
        predict_from_image(blank_path, model, scaler)