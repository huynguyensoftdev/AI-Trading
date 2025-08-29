from __future__ import annotations
import pandas as pd

def load_csv(path: str, time_col: str = 'time') -> pd.DataFrame:
    df = pd.read_csv(path)
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
    required = ['open','high','low','close','volume']
    for c in required:
        if c not in df.columns:
            raise ValueError(f'Missing column: {c}')
    return df
