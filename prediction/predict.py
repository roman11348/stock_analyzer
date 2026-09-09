# prediction/predict.py
import sys
import os
import joblib
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

# Make sibling folders importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "feature_extraction"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from detect_candles import detect_candles
from extract_features import compute_ratio_features
from fetch_data import fetch_ohlc

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
WINDOW_SIZE = 6

PATTERN_TO_SIGNAL = {
    "Hammer": "Bullish",
    "BullishEngulfing": "Bullish",
    "ShootingStar": "Bearish",
    "BearishEngulfing": "Bearish",
    "Doji": "Neutral",
    "SpinningTop": "Neutral",
}


def load_model_and_scaler(model_name="RandomForest"):
    """Loads a trained model and the scaler used during training."""
    model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    return model, scaler


def build_feature_row(candles):
    """
    Given detected candles, builds the same prev_/curr_ prefixed feature
    row used during training (Phase 5) -- from the LAST TWO candles only.
    """
    if len(candles) < 2:
        raise ValueError("Need at least 2 detected candles to build features.")

    prev_features = compute_ratio_features(candles[-2])
    curr_features = compute_ratio_features(candles[-1])

    row = {}
    for key, value in prev_features.items():
        row[f"prev_{key}"] = value
    for key, value in curr_features.items():
        row[f"curr_{key}"] = value

    return row, candles[-1]

def predict_from_image(image_path, model, scaler):
    candles = detect_candles(image_path)
    row, curr_candle = build_feature_row(candles)

    feature_df = pd.DataFrame([row])
    if hasattr(scaler, "feature_names_in_"):
        feature_df = feature_df[scaler.feature_names_in_]

    X_scaled = scaler.transform(feature_df)
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    confidence = round(float(max(probabilities)), 3)
    signal = PATTERN_TO_SIGNAL.get(prediction, "Neutral")

    return {
        "pattern": prediction,
        "signal": signal,
        "confidence": confidence,
        "recommendation": get_recommendation(signal, confidence),   # <- new
        "curr_candle_bullish": curr_candle["is_bullish"],
    }


def render_latest_chart(df, window_size=WINDOW_SIZE, save_path="latest_chart.png"):
    """
    Renders the most recent `window_size` candles in the SAME small, clean
    style as training data -- this is what actually feeds the CV pipeline.
    Takes an already-fetched DataFrame (no network call here).
    """
    if len(df) < window_size:
        raise ValueError(f"Not enough data to build a {window_size}-candle window.")

    window = df.tail(window_size).copy()
    window["date"] = pd.to_datetime(window["date"])
    window = window.set_index("date")

    mpf.plot(
        window,
        type="candle",
        style="charles",
        axisoff=True,
        savefig=dict(fname=save_path, dpi=100, bbox_inches="tight", pad_inches=0.1),
        figsize=(4, 4),
    )
    plt.close("all")
    return save_path



def predict_from_ticker(ticker, model, scaler, window_size=WINDOW_SIZE):
    df = fetch_ohlc(ticker, period="6mo", interval="1d")
    image_path = render_latest_chart(df, window_size=window_size)
    result = predict_from_image(image_path, model, scaler)
    result["ticker"] = ticker
    result["chart_image_path"] = image_path
    result["display_chart_path"] = render_display_chart(df, ticker)
    return result

def render_display_chart(df, ticker, num_candles=30, save_path="latest_chart_display.png"):
    """
    Renders a larger, more readable chart PURELY for display in the GUI.
    Uses more candles, visible axes, and a bigger figure -- never fed into
    the CV/ML pipeline, so it's free to look however is most readable.
    """
    num_candles = min(num_candles, len(df))
    window = df.tail(num_candles).copy()
    window["date"] = pd.to_datetime(window["date"])
    window = window.set_index("date")

    mpf.plot(
        window,
        type="candle",
        style="charles",
        axisoff=False,
        title=f"\n{ticker.upper()} — last {num_candles} sessions",
        ylabel="Price",
        figsize=(9, 5),
        savefig=dict(fname=save_path, dpi=130, bbox_inches="tight"),
    )
    plt.close("all")
    return save_path

STRONG_CONFIDENCE_THRESHOLD = 0.75

def get_recommendation(signal, confidence, strong_threshold=STRONG_CONFIDENCE_THRESHOLD):
    """
    Combines the qualitative signal with the model's confidence to produce
    a graded recommendation. Direction comes from the pattern's signal;
    strength comes from how confident the model was in its prediction.
    """
    if signal == "Bullish":
        return "Strong Buy" if confidence >= strong_threshold else "Buy"
    elif signal == "Bearish":
        return "Strong Sell" if confidence >= strong_threshold else "Sell"
    else:
        return "Hold"

if __name__ == "__main__":
    model, scaler = load_model_and_scaler("RandomForest")

    print("=== Prediction from an existing image ===")
    test_image = "../dataset/images/Hammer/AAPL_147.png"  # change as needed
    result = predict_from_image(test_image, model, scaler)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n=== Prediction from a live ticker ===")
    result2 = predict_from_ticker("M&M.NS", model, scaler)
    for k, v in result2.items():
        print(f"  {k}: {v}")