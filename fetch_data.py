# fetch_data.py
import yfinance as yf
import pandas as pd
import os


def fetch_ohlc(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Downloads historical OHLC data for a given stock ticker.
    period: how far back (e.g. '5y' = 5 years)
    interval: candle timeframe (e.g. '1d' = daily candles)
    """
    df = yf.download(ticker, period=period, interval=interval)
    df = df.reset_index()  # turns the Date index into a normal column
    df = df[["Date", "Open", "High", "Low", "Close"]]
    df.columns = ["date", "open", "high", "low", "close"]
    return df

if __name__ == "__main__":
    os.makedirs("dataset/raw_ohlc", exist_ok=True)
    tickers = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "AAPL", "MSFT",
        "HDFCBANK.NS", "ICICIBANK.NS", "WIPRO.NS",
        "GOOGL", "AMZN", "NVDA", "TSLA", "M&M.NS"
    ]  # mix of NSE + US for volume
    for t in tickers:
        df = fetch_ohlc(t)
        df.to_csv(f"dataset/raw_ohlc/{t.replace('.', '_')}.csv", index=False)
        print(f"Saved {t}: {len(df)} rows")