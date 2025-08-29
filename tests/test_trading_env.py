import numpy as np, pandas as pd
from ai_trading.envs.trading_env import TradingEnv
from ai_trading.utils import seed_everything

def test_tradingenv_single_discrete_reset_step():
    seed_everything(0)
    idx = pd.date_range("2020-01-01", periods=120, freq="H")
    price = 1.0 + 0.001 * np.cumsum(np.random.randn(len(idx)))
    df = pd.DataFrame({"open": price, "high": price*1.001, "low": price*0.999, "close": price, "volume": 100}, index=idx)
    env = TradingEnv(df, window_size=16, action_type="discrete")
    obs, _ = env.reset()
    assert obs.shape[0] == 16 * len(env.features)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
