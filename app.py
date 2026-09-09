# app.py  (project root)
import os
import sys
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "prediction"))
sys.path.append(os.path.join(BASE_DIR, "feature_extraction"))

from predict import load_model_and_scaler, predict_from_image, predict_from_ticker

TEMP_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

PATTERN_DISPLAY_NAMES = {
    "Doji": "Doji",
    "Hammer": "Hammer",
    "ShootingStar": "Shooting Star",
    "SpinningTop": "Spinning Top",
    "BullishEngulfing": "Bullish Engulfing",
    "BearishEngulfing": "Bearish Engulfing",
}

SIGNAL_COLORS = {
    "Bullish": "green",
    "Bearish": "red",
    "Neutral": "gray",
}
RECOMMENDATION_COLORS = {
    "Strong Buy": "#0F9D58",
    "Buy": "#6FCF97",
    "Strong Sell": "#D93025",
    "Sell": "#F2A19C",
    "Hold": "gray",
}


@st.cache_resource
def get_model_and_scaler():
    return load_model_and_scaler("RandomForest")


def display_result(result):
    pattern_name = PATTERN_DISPLAY_NAMES.get(result["pattern"], result["pattern"])
    signal = result["signal"]
    color = SIGNAL_COLORS.get(signal, "black")
    rec = result["recommendation"]
    rec_color = RECOMMENDATION_COLORS.get(rec, "black")

    st.markdown(f"### Detected Pattern: **{pattern_name}**")
    st.markdown(
        f"### Signal: <span style='color:{color}; font-weight:bold'>{signal}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"### Recommendation: <span style='color:{rec_color}; font-weight:bold'>{rec}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Confidence:** {result['confidence'] * 100:.1f}%")
    st.progress(result["confidence"])


def main():
    st.set_page_config(page_title="Candlestick Pattern Recognition", layout="centered")
    st.title("📈 Candlestick Pattern Recognition")
    st.write("Detect candlestick patterns using Computer Vision and Machine Learning.")

    model, scaler = get_model_and_scaler()

    mode = st.radio("Choose input method:", ["Upload Chart Image", "Enter Stock Ticker"])

    if mode == "Upload Chart Image":
        uploaded_file = st.file_uploader("Upload a candlestick chart image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.image(temp_path, caption="Uploaded chart", use_container_width=True)

            if st.button("Predict Pattern"):
                with st.spinner("Analyzing chart..."):
                    try:
                        result = predict_from_image(temp_path, model, scaler)
                        display_result(result)
                    except Exception as e:
                        st.error(f"Could not analyze this image: {e}")

    else:  # Enter Stock Ticker
        ticker = st.text_input("Enter stock ticker (e.g. AAPL, TCS.NS, RELIANCE.NS)")

        if st.button("Fetch & Predict"):
            if not ticker.strip():
                st.warning("Please enter a ticker symbol.")
            else:
                with st.spinner(f"Fetching latest data for {ticker}..."):
                    try:
                        result = predict_from_ticker(ticker.strip().upper(), model, scaler)
                        st.image(
                            result["display_chart_path"],   # <- fixed: was chart_image_path
                            caption=f"Latest chart: {ticker.upper()}",
                            use_container_width=True,
                        )
                        display_result(result)
                        st.caption("Note: uses latest available market data, not live streaming ticks.")
                    except Exception as e:
                        st.error(f"Could not fetch or analyze data for '{ticker}': {e}")


if __name__ == "__main__":
    main()