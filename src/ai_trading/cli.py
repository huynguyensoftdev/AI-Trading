from __future__ import annotations
import typer
from typing import Optional
import pandas as pd, numpy as np
from ai_trading.utils import seed_everything
from ai_trading.data.indicators import compute_basic_indicators
from ai_trading.envs.single_asset_env import SingleAssetEnv
from ai_trading.agents.sac_agent import train_sac

app = typer.Typer()

@app.command()
def quickstart(agent: str = 'sac', quick: bool = True):
    seed_everything(42)
    idx = pd.date_range('2020-01-01', periods=800, freq='H')
    price = 1.0 + 0.001 * np.cumsum(np.random.randn(800))
    df = pd.DataFrame({'open': price, 'high': price*1.001, 'low': price*0.999, 'close': price, 'volume': 100}, index=idx)
    df = compute_basic_indicators(df)
    def env_fn():
        return SingleAssetEnv(df=df, window_size=32)
    if agent == 'sac':
        model = train_sac(env_fn, timesteps=2000 if quick else 100000)
    else:
        from ai_trading.agents.ppo_agent import train_ppo
        model = train_ppo(env_fn, timesteps=2000 if quick else 100000)
    print('Done.')

if __name__ == '__main__':
    app()
