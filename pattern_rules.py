# pattern_rules.py

def candle_metrics(open_, high, low, close):
    body = abs(close - open_)
    rng = high - low
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    return body, rng, upper_wick, lower_wick

def is_doji(open_, high, low, close):
    body, rng, _, _ = candle_metrics(open_, high, low, close)
    return rng > 0 and (body / rng) < 0.1

def is_hammer(open_, high, low, close):
    body, rng, upper_wick, lower_wick = candle_metrics(open_, high, low, close)
    if rng == 0: return False
    return (lower_wick >= 2 * body) and (upper_wick <= 0.1 * rng) and (body / rng < 0.3)

def is_shooting_star(open_, high, low, close):
    body, rng, upper_wick, lower_wick = candle_metrics(open_, high, low, close)
    if rng == 0: return False
    return (upper_wick >= 2 * body) and (lower_wick <= 0.1 * rng) and (body / rng < 0.3)

def is_spinning_top(open_, high, low, close):
    body, rng, upper_wick, lower_wick = candle_metrics(open_, high, low, close)
    if rng == 0: return False
    small_body = 0.1 <= (body / rng) <= 0.3
    balanced_wicks = abs(upper_wick - lower_wick) < 0.15 * rng
    return small_body and balanced_wicks and upper_wick > 0 and lower_wick > 0

def is_bullish_engulfing(prev, curr):
    prev_bearish = prev["close"] < prev["open"]
    curr_bullish = curr["close"] > curr["open"]
    engulfs = curr["open"] < prev["close"] and curr["close"] > prev["open"]
    return prev_bearish and curr_bullish and engulfs

def is_bearish_engulfing(prev, curr):
    prev_bullish = prev["close"] > prev["open"]
    curr_bearish = curr["close"] < curr["open"]
    engulfs = curr["open"] > prev["close"] and curr["close"] < prev["open"]
    return prev_bullish and curr_bearish and engulfs