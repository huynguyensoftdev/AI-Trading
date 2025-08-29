import pandas as pd, numpy as np
from ai_trading.envs.single_asset_env import SingleAssetEnv
from ai_trading.data.indicators import compute_basic_indicators

def test_env_reset_step():
    idx = pd.date_range('2020-01-01', periods=200, freq='H')
    price = 1.0 + 0.001 * np.cumsum(np.random.randn(200))
    df = pd.DataFrame({'open': price, 'high': price*1.001, 'low': price*0.999, 'close': price, 'volume': 100}, index=idx)
    df = compute_basic_indicators(df)
    env = SingleAssetEnv(df=df, window_size=16)
    obs, _ = env.reset()
    assert obs.shape[0] == 16 * 6
