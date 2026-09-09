# tests/test_pattern_rules.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pattern_rules import (
    is_doji, is_hammer, is_shooting_star, is_spinning_top,
    is_bullish_engulfing, is_bearish_engulfing
)


def test_doji_detects_tiny_body():
    # open ~= close, wide range -> should be Doji
    assert is_doji(open_=100, high=110, low=90, close=100.2) is True


def test_doji_rejects_large_body():
    # large body relative to range -> should NOT be Doji
    assert is_doji(open_=100, high=110, low=90, close=108) is False


def test_hammer_detects_long_lower_wick():
    # small body near top, long lower wick, tiny upper wick
    assert is_hammer(open_=95, high=96, low=80, close=96) is True


def test_hammer_rejects_long_upper_wick():
    # this shape is a Shooting Star, NOT a Hammer -- important negative case
    assert is_hammer(open_=84, high=100, low=83, close=85) is False


def test_shooting_star_detects_long_upper_wick():
    assert is_shooting_star(open_=84, high=100, low=83, close=85) is True


def test_spinning_top_detects_balanced_wicks():
    assert is_spinning_top(open_=92.5, high=105, low=85, close=97.5) is True


def test_bullish_engulfing_detects_correctly():
    prev = {"open": 100, "close": 95}   # bearish candle
    curr = {"open": 94, "close": 102}   # bullish candle engulfing prev
    assert is_bullish_engulfing(prev, curr) is True


def test_bullish_engulfing_rejects_non_engulfing():
    prev = {"open": 100, "close": 95}
    curr = {"open": 96, "close": 99}    # bullish, but does NOT fully engulf
    assert is_bullish_engulfing(prev, curr) is False


def test_bearish_engulfing_detects_correctly():
    prev = {"open": 95, "close": 100}   # bullish candle
    curr = {"open": 101, "close": 93}   # bearish candle engulfing prev
    assert is_bearish_engulfing(prev, curr) is True