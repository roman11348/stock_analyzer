# feature_extraction/validate_features.py
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

df = pd.read_csv("../dataset/features.csv")

print("Mean feature values per pattern:\n")
summary = df.groupby("label")[[
    "curr_body_ratio", "curr_upper_wick_ratio",
    "curr_lower_wick_ratio", "curr_wick_to_body_ratio"
]].mean().round(3)

print(summary)