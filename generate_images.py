# generate_images.py
import os
import glob
import random
import pandas as pd
import mplfinance as mpf
from pattern_rules import (
    is_doji, is_hammer, is_shooting_star, is_spinning_top,
    is_bullish_engulfing, is_bearish_engulfing
)

WINDOW_SIZE = 6           # candles shown per image (context + target candle)
TARGET_PER_CLASS = 200    # final balanced count per pattern
IMAGE_DIR = "dataset/images"
RAW_DIR = "dataset/raw_ohlc"
PATTERNS = ["Doji", "Hammer", "ShootingStar", "SpinningTop",
            "BullishEngulfing", "BearishEngulfing"]


def find_candidates(csv_path):
    """Scan one ticker's data, return {pattern: [(ticker, end_index), ...]}"""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    ticker = os.path.basename(csv_path).replace(".csv", "")
    candidates = {p: [] for p in PATTERNS}

    for i in range(WINDOW_SIZE - 1, len(df)):
        row = df.iloc[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        if is_doji(o, h, l, c):
            candidates["Doji"].append((ticker, i))
        if is_hammer(o, h, l, c):
            candidates["Hammer"].append((ticker, i))
        if is_shooting_star(o, h, l, c):
            candidates["ShootingStar"].append((ticker, i))
        if is_spinning_top(o, h, l, c):
            candidates["SpinningTop"].append((ticker, i))

        prev_row = df.iloc[i - 1]
        if is_bullish_engulfing(prev_row, row):
            candidates["BullishEngulfing"].append((ticker, i))
        if is_bearish_engulfing(prev_row, row):
            candidates["BearishEngulfing"].append((ticker, i))

    return df, candidates


def render_window(df, end_index, save_path):
    """Render WINDOW_SIZE candles ending at end_index as a clean chart image."""
    window = df.iloc[end_index - WINDOW_SIZE + 1 : end_index + 1].copy()
    window = window.set_index("date")

    mpf.plot(
        window,
        type="candle",
        style="charles",
        axisoff=True,
        savefig=dict(fname=save_path, dpi=100, bbox_inches="tight", pad_inches=0.1),
        figsize=(4, 4),
    )


def main():
    random.seed(42)  # reproducibility — same sample every run

    all_candidates = {p: [] for p in PATTERNS}
    dataframes = {}

    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    for csv_path in csv_files:
        df, candidates = find_candidates(csv_path)
        ticker = os.path.basename(csv_path).replace(".csv", "")
        dataframes[ticker] = df
        for pattern in PATTERNS:
            all_candidates[pattern].extend(candidates[pattern])

    for pattern in PATTERNS:
        os.makedirs(os.path.join(IMAGE_DIR, pattern), exist_ok=True)
        pool = all_candidates[pattern]
        sample_size = min(TARGET_PER_CLASS, len(pool))
        sampled = random.sample(pool, sample_size)

        print(f"{pattern}: {len(pool)} candidates -> sampling {sample_size}")

        for ticker, end_index in sampled:
            df = dataframes[ticker]
            save_path = os.path.join(IMAGE_DIR, pattern, f"{ticker}_{end_index}.png")
            render_window(df, end_index, save_path)

        print(f"  done: {pattern}")


if __name__ == "__main__":
    main()