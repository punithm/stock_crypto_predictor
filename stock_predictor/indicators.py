"""
Technical indicators for feature engineering.
"""
import pandas as pd
import numpy as np


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def returns(series: pd.Series) -> pd.Series:
    return series.pct_change().fillna(0)


def volatility(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change().rolling(window).std().fillna(0)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=window - 1, adjust=False).mean()
    ma_down = down.ewm(com=window - 1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-8)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def prepare_indicators(df: pd.DataFrame, lookback: int = 60, sentiment_series: pd.Series = None) -> pd.DataFrame:
    """
    Add a set of technical indicators to the dataframe.
    Returns a DataFrame with indicator columns aligned to the original index.
    """
    close = df['Close']
    ind = pd.DataFrame(index=df.index)
    ind['close'] = close
    ind['ret'] = returns(close)
    ind['sma_10'] = sma(close, 10)
    ind['sma_30'] = sma(close, 30)
    ind['ema_12'] = ema(close, 12)
    ind['ema_26'] = ema(close, 26)
    ind['volatility_10'] = volatility(close, 10)
    ind['volatility_30'] = volatility(close, 30)
    ind['rsi_14'] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    ind['macd'] = macd_line
    ind['macd_sig'] = signal_line
    ind['macd_hist'] = hist
    ind['vol'] = df['Volume'] if 'Volume' in df.columns else 0
    # Optionally include daily sentiment (align by date)
    if sentiment_series is not None and not sentiment_series.empty:
        # sentiment_series indexed by date (datetime.date or datetime)
        # convert to same index type and align
        s = sentiment_series.copy()
        # ensure datetime index
        s.index = pd.to_datetime(s.index)
        s = s.reindex(ind.index, method='ffill').fillna(0)
        ind['sentiment'] = s.values
    else:
        ind['sentiment'] = 0

    # Fill any remaining NaNs
    ind = ind.fillna(method='bfill').fillna(method='ffill').fillna(0)
    return ind
