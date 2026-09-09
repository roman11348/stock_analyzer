# Candlestick Pattern Recognition using Computer Vision and Machine Learning

A system that detects candlestick patterns from stock chart images (or a live stock ticker) using OpenCV for detection and classical Machine Learning for classification — built as a summer training project.

## Overview

Manually spotting candlestick patterns on a stock chart is slow and error-prone. This project automates it: given a chart image **or** a stock ticker symbol, the system detects the relevant candlestick(s), extracts geometric features via computer vision, classifies the pattern with a trained ML model, and returns a **Bullish / Bearish / Neutral** trading signal with a confidence score.

## Patterns Supported (V1)

- Doji
- Hammer
- Shooting Star
- Bullish Engulfing
- Bearish Engulfing
- Spinning Top

## Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV
- **Data:** NumPy, Pandas, yfinance
- **Charting:** mplfinance, Matplotlib
- **Machine Learning:** scikit-learn (SVM, Random Forest, Decision Tree, KNN)
- **GUI:** Streamlit
- **Testing:** pytest
- **IDE:** PyCharm

## Folder Structure

```
CandlestickPatternRecognition/
    dataset/
        raw_ohlc/            # cached OHLC CSVs from yfinance
        images/               # generated, labeled chart images (6 pattern folders)
        features.csv          # extracted feature dataset used for training
    preprocessing/
        cv_preprocessing.py    # grayscale, threshold, morphology, contours
    feature_extraction/
        detect_candles.py      # contour-based body/wick detection
        extract_features.py    # converts detections into ratio features
    models/
        RandomForest.pkl        # final trained model
        scaler.pkl               # fitted StandardScaler
    prediction/
        predict.py               # unified prediction pipeline (image + ticker input)
    tests/
        test_pattern_rules.py
        test_detect_candles.py
        test_predict_edge_cases.py
        test_performance.py
    fetch_data.py                # OHLC data fetching (yfinance)
    pattern_rules.py             # rule-based pattern definitions used for labeling
    generate_images.py           # synthetic labeled dataset generation
    train.py                     # trains and compares all 4 ML models
    app.py                       # Streamlit GUI
    requirements.txt
    README.md
```

## Pipeline / Workflow

```
 Upload Chart Image ─┐
                      ├──▶ Preprocessing ──▶ Candlestick Detection ──▶ Feature Extraction ──▶ Random Forest ──▶ Prediction + Signal
 Stock Ticker ─▶ Fetch OHLC ─▶ Render Chart ─┘
```

Both input paths converge into the exact same detection and classification pipeline — the ticker path simply fetches the latest available data and renders it in the same visual style as the training images before handing it off.

## Installation

```bash
git clone <your-repo-url>
cd CandlestickPatternRecognition
python -m venv .venv
.venv\Scripts\activate        # on Windows
pip install -r requirements.txt
```

**Key packages:** `opencv-python`, `numpy`, `pandas`, `matplotlib`, `mplfinance`, `yfinance`, `scikit-learn`, `joblib`, `streamlit`, `pytest`

## Usage

### Run the web app
```bash
streamlit run app.py
```
Choose one of two input modes:
- **Upload Chart Image** — upload a candlestick chart screenshot directly.
- **Enter Stock Ticker** — type a symbol (e.g. `AAPL`, `TCS.NS`, `RELIANCE.NS`); the app fetches the latest available data, displays a larger readable chart, and predicts the pattern.

### Regenerate the dataset (optional)
```bash
python fetch_data.py         # fetch OHLC data
python generate_images.py    # render labeled candlestick images
```

### Extract features and retrain the model
```bash
python feature_extraction/extract_features.py
python train.py
```

### Run the test suite
```bash
pytest tests/ -v
```

## Dataset

The dataset is generated, not manually labeled. Historical OHLC data (13 tickers, NSE + US markets) is scanned using precise mathematical definitions of each pattern (e.g. a Doji requires `body / range < 0.1`). Matching candle windows are rendered as 6-candle chart images via `mplfinance` and randomly sampled to ~200 balanced images per class (1,200 total), avoiding the natural over-representation of common patterns like Doji.

## Feature Engineering

Each chart image is processed through OpenCV (grayscale → threshold → morphology → contour detection) to isolate individual candles. For the **last two candles** in each image, four scale-invariant ratios are computed: `body_ratio`, `upper_wick_ratio`, `lower_wick_ratio`, and `wick_to_body_ratio`, plus a bullish/bearish color flag — an 11-column fixed-length feature vector regardless of whether the pattern involves one or two candles.

## Model & Results

Four classical ML models were trained and compared on an 80/20 stratified split:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest (final)** | 0.963 | 0.964 | 0.962 | **0.963** |
| Decision Tree | 0.921 | 0.925 | 0.921 | 0.921 |
| SVM | 0.908 | 0.910 | 0.908 | 0.908 |
| KNN | 0.892 | 0.892 | 0.892 | 0.890 |

**Random Forest** was selected as the final model. Tree-based models outperformed SVM/KNN, consistent with the fact that the features were engineered from threshold-based rules in the first place. Doji was the most-confused class across all models, due to pixel-measurement noise near its tight `body_ratio < 0.1` boundary against Spinning Top.

## Testing

A `pytest` suite covers:
- **Unit tests** for the rule-based pattern definitions (`pattern_rules.py`)
- **Consistency checks** on detected candle geometry (body + wicks summing to total range)
- **Edge cases** — missing files, corrupted files, and blank images all raise clear exceptions rather than failing silently
- **Performance** — confirms predictions complete in under 2 seconds

All tests pass (`pytest tests/ -v`).

## Known Limitations

- **Dark-themed screenshots are not supported.** The thresholding step assumes a light background with darker candles, matching the synthetic training data. A dark-theme chart (common in apps like TradingView) inverts this assumption and produces incorrect detection. Planned fix: adaptive background-brightness detection.
- **Ticker-fetch data is not truly real-time.** It reflects the latest available data from `yfinance`, which may be end-of-day or delayed, not live streaming ticks.
- **Trend context is not considered.** Patterns like Hammer are classified purely on candle shape, without checking whether they occur after a genuine downtrend, as strict technical analysis would require.
- **Train/test split is random, not grouped by ticker**, so a small amount of near-duplicate leakage between overlapping candle windows is possible.

## Future Scope

- Adaptive thresholding to support both light and dark chart themes
- Incorporating trend context before classifying reversal patterns
- Group-based (per-ticker) train/test splitting
- Expanding to multi-candle patterns (e.g. Three White Soldiers, Morning/Evening Star)
- Benchmarking against a CNN-based deep learning approach
- A hand-labeled real-screenshot validation set to measure real-world accuracy

## Author

[Your Name] — B.Tech Computer Science
Summer Training Project, CDAC, 2026
