# feature_extraction/extract_features.py
import os
import glob
import pandas as pd
from detect_candles import detect_candles

PATTERNS = ["Doji", "Hammer", "ShootingStar", "SpinningTop",
            "BullishEngulfing", "BearishEngulfing"]

IMAGE_DIR = "../dataset/images"
OUTPUT_CSV = "../dataset/features.csv"


def compute_ratio_features(candle):
    """
    Converts one candle's raw pixel measurements into scale-invariant ratios,
    mirroring the same formulas used in pattern_rules.py -- but computed from
    detected pixels instead of exact OHLC numbers.
    """
    total_range = candle["total_range_px"]
    body = candle["body_height_px"]
    upper_wick = candle["upper_wick_px"]
    lower_wick = candle["lower_wick_px"]

    body_ratio = body / total_range if total_range > 0 else 0
    upper_wick_ratio = upper_wick / total_range if total_range > 0 else 0
    lower_wick_ratio = lower_wick / total_range if total_range > 0 else 0
    wick_to_body_ratio = (upper_wick + lower_wick) / body if body > 0 else 0

    is_bullish = 1 if candle["is_bullish"] else 0

    return {
        "body_ratio": round(body_ratio, 4),
        "upper_wick_ratio": round(upper_wick_ratio, 4),
        "lower_wick_ratio": round(lower_wick_ratio, 4),
        "wick_to_body_ratio": round(wick_to_body_ratio, 4),
        "is_bullish": is_bullish,
    }


def extract_features_from_image(image_path):
    """
    Runs full detection on one image, then computes ratio features for the
    last TWO candles (prev, curr). Using this same 2-candle schema for every
    class -- even single-candle patterns -- keeps feature vectors a fixed
    length across all 6 classes, which classical ML models require.
    """
    candles = detect_candles(image_path)
    if len(candles) < 2:
        return None  # malformed detection -- skip this image

    prev_candle = candles[-2]
    curr_candle = candles[-1]

    prev_features = compute_ratio_features(prev_candle)
    curr_features = compute_ratio_features(curr_candle)

    row = {}
    for key, value in prev_features.items():
        row[f"prev_{key}"] = value
    for key, value in curr_features.items():
        row[f"curr_{key}"] = value

    return row


def build_dataset():
    rows = []
    skipped = 0

    for pattern in PATTERNS:
        folder = os.path.join(IMAGE_DIR, pattern)
        image_paths = glob.glob(os.path.join(folder, "*.png"))
        print(f"Processing {pattern}: {len(image_paths)} images")

        for image_path in image_paths:
            features = extract_features_from_image(image_path)
            if features is None:
                skipped += 1
                continue
            features["label"] = pattern
            rows.append(features)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
    print(f"Skipped {skipped} malformed images")
    print("\nClass distribution:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    build_dataset()

