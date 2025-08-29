from ai_trading.utils import seed_everything
from ai_trading.data.indicators import compute_basic_indicators
from ai_trading.envs.single_asset_env import SingleAssetEnv
from ai_trading.agents.sac_agent import train_sac
import pandas as pd, numpy as np

def main(quick=True):
    seed_everything(42)
    idx = pd.date_range('2020-01-01', periods=1000, freq='H')
    price = 1.0 + 0.001 * np.cumsum(np.random.randn(1000))
    df = pd.DataFrame({'open': price, 'high': price*1.001, 'low': price*0.999, 'close': price, 'volume': 100}, index=idx)
    df = compute_basic_indicators(df)
    def env_fn():
        return SingleAssetEnv(df=df, window_size=64)
    model = train_sac(env_fn, timesteps=2000 if quick else 100000)
    model.save('sac_demo')
    print('Trained SAC (demo)')

if __name__ == '__main__':
    main(True)
