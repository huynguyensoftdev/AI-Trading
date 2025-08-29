from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, data: dict, window_size: int = 32, fee: float = 1e-4):
        super().__init__()
        common = None
        for df in data.values():
            common = df.index if common is None else common.intersection(df.index)
        if len(common) < window_size + 2:
            raise ValueError('not enough overlapping timestamps')
        self.symbols = list(data.keys())
        self.data = {s: data[s].loc[common] for s in self.symbols}
        feats = ['close','return','ema_fast','ema_slow','macd','rsi']
        self._features = {s: self.data[s][feats].values.astype('float32') for s in self.symbols}
        feat_dim = sum([f.shape[1] for f in self._features.values()])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size * feat_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3]*len(self.symbols))
        self.window_size = window_size
        self.ptr = window_size
        self.position = [0]*len(self.symbols)
        self.fee = fee
    def reset(self, seed=None, options=None):
        self.ptr = self.window_size
        self.position = [0]*len(self.symbols)
        return self._get_obs(), {}
    def step(self, action):
        prev = [self._features[s][self.ptr-1,0] for s in self.symbols]
        price = [self._features[s][self.ptr,0] for s in self.symbols]
        reward = 0.0
        new_pos = [1 if a==1 else (-1 if a==2 else p) for a,p in zip(action, self.position)]
        for i in range(len(self.symbols)):
            if new_pos[i]==1:
                reward += (price[i]-prev[i])/(prev[i]+1e-12)
            elif new_pos[i]==-1:
                reward += (prev[i]-price[i])/(prev[i]+1e-12)
            if new_pos[i]!=self.position[i]:
                reward -= self.fee
        self.position = new_pos
        self.ptr += 1
        done = self.ptr >= len(list(self._features.values())[0]) -1
        return self._get_obs(), float(reward), done, False, {'position': self.position}
    def _get_obs(self):
        parts = []
        start = self.ptr - self.window_size
        for s in self.symbols:
            parts.append(self._features[s][start:self.ptr].flatten())
        return np.concatenate(parts)
