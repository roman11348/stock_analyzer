# count_patterns.py
import os
import glob
import pandas as pd
from pattern_rules import (
    is_doji, is_hammer, is_shooting_star, is_spinning_top,
    is_bullish_engulfing, is_bearish_engulfing
)

def count_patterns(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)

    counts = {
        "Doji": 0,
        "Hammer": 0,
        "ShootingStar": 0,
        "SpinningTop": 0,
        "BullishEngulfing": 0,
        "BearishEngulfing": 0,
    }

    prev_row = None  # reset for every new ticker — important!

    for i, row in df.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        if is_doji(o, h, l, c):
            counts["Doji"] += 1
        if is_hammer(o, h, l, c):
            counts["Hammer"] += 1
        if is_shooting_star(o, h, l, c):
            counts["ShootingStar"] += 1
        if is_spinning_top(o, h, l, c):
            counts["SpinningTop"] += 1

        if prev_row is not None:
            if is_bullish_engulfing(prev_row, row):
                counts["BullishEngulfing"] += 1
            if is_bearish_engulfing(prev_row, row):
                counts["BearishEngulfing"] += 1

        prev_row = row

    return counts


if __name__ == "__main__":
    folder = "dataset/raw_ohlc"
    csv_files = glob.glob(os.path.join(folder, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {folder}. Did you run fetch_data.py first?")

    total_counts = {
        "Doji": 0, "Hammer": 0, "ShootingStar": 0,
        "SpinningTop": 0, "BullishEngulfing": 0, "BearishEngulfing": 0,
    }

    for csv_path in csv_files:
        ticker_name = os.path.basename(csv_path).replace(".csv", "")
        result = count_patterns(csv_path)

        print(f"\nPattern counts for {ticker_name}:")
        for pattern, count in result.items():
            print(f"  {pattern}: {count}")
            total_counts[pattern] += count

    print("\n" + "=" * 40)
    print("TOTAL across all tickers:")
    for pattern, count in total_counts.items():
        print(f"  {pattern}: {count}")