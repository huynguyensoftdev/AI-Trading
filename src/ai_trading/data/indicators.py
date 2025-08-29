from __future__ import annotations
import pandas as pd
import numpy as np

def compute_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['return'] = df['close'].pct_change().fillna(0.0)
    df['ema_fast'] = df['close'].ewm(span=12).mean()
    df['ema_slow'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['rsi'] = _rsi(df['close'], 14)
    return df.dropna()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))
