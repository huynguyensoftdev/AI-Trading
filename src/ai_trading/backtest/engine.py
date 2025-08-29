from __future__ import annotations
import pandas as pd
import numpy as np

def run_signals(df: pd.DataFrame, positions: pd.Series, fee: float = 1e-4) -> pd.DataFrame:
    pos = positions.shift(1).fillna(0)
    ret = df['close'].pct_change().fillna(0.0)
    pnl = pos * ret - np.where(positions != positions.shift(1), fee, 0)
    eq = (1 + pnl).cumprod()
    out = df.copy()
    out['position'] = pos
    out['pnl'] = pnl
    out['equity'] = eq
    return out

def metrics(equity: pd.Series) -> dict:
    ret = equity.pct_change().fillna(0.0)
    sharpe = np.sqrt(252) * ret.mean() / (ret.std() + 1e-12)
    dd = (equity / equity.cummax() - 1).min()
    return {'sharpe': float(sharpe), 'max_drawdown': float(dd)}
