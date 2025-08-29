from __future__ import annotations
import itertools
from typing import Callable, Dict, List, Any

def simple_grid_search(env_fn: Callable, train_fn: Callable, param_grid: Dict[str, List[Any]], timesteps: int = 2000):
    best = None
    best_score = -1e9
    for combo in _product(param_grid):
        model = train_fn(env_fn, timesteps=timesteps, **combo)
        env = env_fn()
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total += reward
        if total > best_score:
            best_score = total
            best = combo
    return best, best_score

def _product(d):
    keys = list(d.keys())
    for vals in itertools.product(*(d[k] for k in keys)):
        yield dict(zip(keys, vals))
