from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class SingleAssetEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, df: pd.DataFrame, window_size: int = 32, fee: float = 1e-4):
        super().__init__()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError('df must have DatetimeIndex')
        self.df = df.copy()
        self.window_size = window_size
        self.ptr = window_size
        self.fee = fee
        feats = ['close','return','ema_fast','ema_slow','macd','rsi']
        self._features = self.df[feats].values.astype('float32')
        n_feat = self._features.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size * n_feat,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.position = 0
    def reset(self, seed=None, options=None):
        self.ptr = self.window_size
        self.position = 0
        return self._get_obs(), {}
    def step(self, action):
        prev_price = self._features[self.ptr -1, 0]
        price = self._features[self.ptr, 0]
        reward = 0.0
        if action == 1:
            self.position = 1
            reward -= self.fee
        elif action == 2:
            self.position = -1
            reward -= self.fee
        reward += self.position * ((price - prev_price) / (prev_price + 1e-12))
        self.ptr += 1
        done = self.ptr >= len(self._features) -1
        return self._get_obs(), float(reward), done, False, {'position': self.position}
    def _get_obs(self):
        start = self.ptr - self.window_size
        return self._features[start:self.ptr].flatten()
