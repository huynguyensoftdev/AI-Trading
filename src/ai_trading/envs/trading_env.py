"""
TradingEnv - OpenAI Gym / Gymnasium compatible trading environment.

Features:
- Single-asset (pd.DataFrame) or multi-asset (dict[symbol] -> pd.DataFrame).
- Observation: flattened window of features per asset (window_size x n_features).
- Action:
    - discrete: MultiDiscrete([3]*n_assets) with mapping {0:hold,1:long,2:short}
    - continuous: Box(low=-1, high=1, shape=(n_assets,)) representing allocation [-1,1]
- Reward: portfolio return (sum of per-asset returns weighted by positions/allocations) minus fees/slippage.
- Gymnasium-compatible step/reset signature: (obs, reward, terminated, truncated, info)

Usage:
    env = TradingEnv(df)  # single-asset
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Author: Generated for AI-Trading project
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# gymnasium preferred; fallback to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces  # type: ignore

ArrayLike = Union[np.ndarray, List[float]]

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        window_size: int = 32,
        features: Optional[List[str]] = None,
        action_type: str = "discrete",  # "discrete" or "continuous"
        normalize_obs: bool = True,
        fee: float = 1e-4,  # fee per trade (proportional)
        slippage: float = 0.0,
        reward_mode: str = "portfolio_return",  # or 'pnl' (instant PnL)
        use_log_return: bool = False,
        max_position: float = 1.0,
        dtype: np.dtype = np.float32,
    ):
        """
        Parameters
        ----------
        data : DataFrame or dict of DataFrames
            If DataFrame: single-asset with DatetimeIndex and OHLCV columns.
            If dict: key -> symbol, value -> DataFrame (must share overlapping DatetimeIndex).
        window_size : int
            Number of timesteps included in observation window.
        features : list[str] or None
            List of column names to use as features (defaults to close, return, ema_fast,... if present).
        action_type : str
            "discrete" (MultiDiscrete) or "continuous" (Box allocations in [-1,1]).
        normalize_obs : bool
            If True, observations are normalized per feature using running mean/std within the window.
        fee : float
            Transaction cost applied when position changes.
        slippage : float
            Additional proportional cost to simulate market slippage.
        reward_mode : str
            'portfolio_return' or 'pnl' (difference insignificant for single-step).
        use_log_return : bool
            If True, use log returns for reward calculations.
        max_position : float
            Clamp continuous actions to [-max_position, max_position].
        """
        super().__init__()

        # Basic checks
        if not isinstance(data, (pd.DataFrame, dict)):
            raise ValueError("data must be a pandas DataFrame or dict of DataFrames")

        # Normalize input to dict of DataFrames
        if isinstance(data, pd.DataFrame):
            self.symbols = ["_ASSET"]
            self.data = {"_ASSET": data.copy()}
        else:
            self.symbols = list(data.keys())
            self.data = {s: data[s].copy() for s in self.symbols}

        # Align indices by inner join intersection
        common_idx = None
        for df in self.data.values():
            if common_idx is None:
                common_idx = df.index
            else:
                common_idx = common_idx.intersection(df.index)
        if common_idx is None or len(common_idx) == 0:
            raise ValueError("No overlapping timestamps between asset DataFrames")

        # Reindex to common index and drop rows with NaN
        for s in self.symbols:
            self.data[s] = self.data[s].loc[common_idx].dropna()
        # Require minimal columns
        for s, df in self.data.items():
            if "close" not in df.columns:
                raise ValueError(f"DataFrame for {s} must contain 'close' column")

        self.window_size = int(window_size)
        self.fee = float(fee)
        self.slippage = float(slippage)
        self.reward_mode = reward_mode
        self.normalize_obs = bool(normalize_obs)
        self.use_log_return = bool(use_log_return)
        self.max_position = float(max_position)
        self.dtype = dtype

        # Infer default features
        default_feats = ["close", "return", "ema_fast", "ema_slow", "macd", "rsi"]
        if features is None:
            # include existing default features if present
            feats = []
            for c in default_feats:
                # include if any asset has it
                if any(c in df.columns for df in self.data.values()):
                    feats.append(c)
            # always include close
            if "close" not in feats:
                feats = ["close"] + feats
            self.features = feats
        else:
            self.features = features

        # Precompute feature arrays per asset
        self._feature_arrays = {}
        for s in self.symbols:
            df = self.data[s]
            # Ensure required derived features exist
            if "return" not in df.columns:
                df["return"] = df["close"].pct_change().fillna(0.0)
            # compute missing ema/macd/rsi only if asked and missing
            if "ema_fast" in self.features and "ema_fast" not in df.columns:
                df["ema_fast"] = df["close"].ewm(span=12).mean()
            if "ema_slow" in self.features and "ema_slow" not in df.columns:
                df["ema_slow"] = df["close"].ewm(span=26).mean()
            if "macd" in self.features and "macd" not in df.columns:
                df["macd"] = df.get("ema_fast", df["close"].ewm(span=12).mean()) - df.get(
                    "ema_slow", df["close"].ewm(span=26).mean()
                )
            if "rsi" in self.features and "rsi" not in df.columns:
                df["rsi"] = self._rsi(df["close"], 14)
            # keep only needed features (in order)
            missing = [f for f in self.features if f not in df.columns]
            if missing:
                raise ValueError(f"Missing feature columns in data[{s}]: {missing}")
            arr = df[self.features].values.astype(dtype)
            self._feature_arrays[s] = arr

        # Determine env length (must be same across all assets because we intersected indices)
        # Use length of first symbol
        first_sym = self.symbols[0]
        self._n_steps = self._feature_arrays[first_sym].shape[0]

        # Compute per-asset feature dims and total observation dim
        self.n_assets = len(self.symbols)
        self.n_features_per_asset = self._feature_arrays[first_sym].shape[1]
        self.obs_shape = (self.window_size * self.n_assets * self.n_features_per_asset,)

        # Action space
        if action_type not in ("discrete", "continuous"):
            raise ValueError("action_type must be 'discrete' or 'continuous'")
        self.action_type = action_type
        if action_type == "discrete":
            # MultiDiscrete with 3 choices per asset: 0 hold,1 long,2 short
            self.action_space = spaces.MultiDiscrete([3] * self.n_assets)
        else:
            # continuous allocations per asset in [-max_position, max_position]
            low = -self.max_position * np.ones(self.n_assets, dtype=self.dtype)
            high = self.max_position * np.ones(self.n_assets, dtype=self.dtype)
            self.action_space = spaces.Box(low=low, high=high, dtype=self.dtype)

        # Observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=self.dtype)

        # Internal pointers/state
        # ptr points to the current index (0..n_steps-1). We start at window_size index.
        self.ptr = int(self.window_size)
        # positions: for discrete: -1/0/1 per asset; for continuous: allocation float per asset
        self.positions = np.zeros(self.n_assets, dtype=self.dtype)
        # cash/equity tracking (not required but useful)
        self.equity = 1.0  # normalized
        # done flag for truncation/termination
        self.terminated = False
        self.truncated = False

    # --------- Helpers ----------
    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            return
        np.random.seed(int(seed))

    def _get_obs(self) -> np.ndarray:
        """
        Stack windows for each asset in symbol order.
        Output is flattened 1D array of shape (window_size * n_assets * n_features_per_asset,)
        """
        parts = []
        start = self.ptr - self.window_size
        for s in self.symbols:
            arr = self._feature_arrays[s][start : self.ptr, :]  # (window, features)
            if self.normalize_obs:
                # Normalize per feature across the window
                mean = np.nanmean(arr, axis=0, keepdims=True)
                std = np.nanstd(arr, axis=0, keepdims=True) + 1e-9
                arr = (arr - mean) / std
            parts.append(arr.flatten())
        obs = np.concatenate(parts).astype(self.dtype)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset env. Returns (obs, info) to be compatible with Gymnasium.
        """
        if seed is not None:
            self.seed(seed)
        self.ptr = int(self.window_size)
        self.positions = np.zeros(self.n_assets, dtype=self.dtype)
        self.equity = 1.0
        self.terminated = False
        self.truncated = False
        obs = self._get_obs()
        return obs, {}

    def step(self, action: ArrayLike) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        action: MultiDiscrete array (discrete mode) or np.array allocations (continuous)
        returns: obs, reward, terminated, truncated, info
        """
        if self.terminated or self.truncated:
            raise RuntimeError("Step called after env termination. Call reset().")

        # Normalize action shape
        if self.action_type == "discrete":
            # convert MultiDiscrete to positions [-1,0,1]
            act_arr = np.asarray(action, dtype=int)
            if act_arr.shape == ():
                # single scalar, broadcast
                act_arr = np.array([int(act_arr)] * self.n_assets)
            if act_arr.shape != (self.n_assets,):
                raise ValueError(f"Discrete action must be shape ({self.n_assets},), got {act_arr.shape}")
            new_positions = np.where(act_arr == 1, 1.0, np.where(act_arr == 2, -1.0, self.positions))
        else:
            new_positions = np.asarray(action, dtype=self.dtype).flatten()
            if new_positions.shape != (self.n_assets,):
                raise ValueError(f"Continuous action must be shape ({self.n_assets},), got {new_positions.shape}")
            # clip to bounds
            new_positions = np.clip(new_positions, -self.max_position, self.max_position)

        # Compute price change from t-1 to t
        prev_prices = np.array([self._feature_arrays[s][self.ptr - 1, 0] for s in self.symbols], dtype=self.dtype)
        cur_prices = np.array([self._feature_arrays[s][self.ptr, 0] for s in self.symbols], dtype=self.dtype)

        if self.use_log_return:
            returns = np.log(cur_prices + 1e-12) - np.log(prev_prices + 1e-12)
        else:
            returns = (cur_prices - prev_prices) / (prev_prices + 1e-12)

        # compute reward: sum(position * returns)
        # For discrete mode we interpret positions as -1/0/1
        # For continuous mode positions are allocations in [-1,1]
        reward = float(np.sum(new_positions * returns))

        # trade costs for position changes
        pos_changes = new_positions != self.positions
        n_changes = float(np.sum(pos_changes))
        cost = (self.fee + self.slippage) * n_changes
        reward -= cost

        # update equity (multiplicative)
        self.equity = float(self.equity * (1.0 + reward))

        # update positions
        self.positions = new_positions.copy()

        # advance pointer
        self.ptr += 1
        # termination if reach the end
        terminated = self.ptr >= (self._n_steps - 1)
        truncated = False

        obs = self._get_obs()
        info = {
            "prices": cur_prices,
            "returns": returns,
            "positions": self.positions.copy(),
            "equity": self.equity,
            "trade_cost": cost,
        }
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode: str = "human"):
        # Minimal render: print pointer, positions, equity
        dt_index = None
        try:
            # get datetime if available
            s = self.symbols[0]
            # find original dataframe index value at ptr-1
            idx = self.ptr - 1
            dt_index = self.data[s].index[idx]
        except Exception:
            dt_index = None
        print(f"Step {self.ptr}/{self._n_steps} | Time {dt_index} | Equity {self.equity:.4f} | Positions {self.positions}")

    def close(self):
        pass
