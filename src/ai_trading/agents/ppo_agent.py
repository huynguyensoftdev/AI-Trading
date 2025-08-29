from __future__ import annotations
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Callable

def train_ppo(env_fn: Callable, timesteps: int = 100_000, **kwargs):
    env = DummyVecEnv([env_fn])
    model = PPO('MlpPolicy', env, verbose=1, **kwargs)
    model.learn(total_timesteps=timesteps)
    return model
