import torch
import random
import numpy as np
import gymnasium as gym
from pathlib import Path
from typing import Callable, List, Any

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, SubprocVecEnv

from config import Config
from data_processing.data_processing import load_instance 
from gym_jssp_env import GymJSSPEnv


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unwrap_env(e: gym.Env) -> Any:
    """Unwrap possible layers of wrappers and return the bottom layer env."""
    cur: Any = e
    while hasattr(cur, "env") or hasattr(cur, "venv"):
        cur = getattr(cur, "env", getattr(cur, "venv", cur))
    return cur


def _mask_fn(e: gym.Env) -> np.ndarray:
    """
    Masking function for ActionMasker:
    - First calls the underlying env.action_masks() method
    - Next, tries the action_mask property
    - If no mask exists, returns a fallback value of True (no error, 
      but the constraint will be lost)
    """
    base = _unwrap_env(e)
    if hasattr(base, "action_masks"):
        return base.action_masks()
    if hasattr(base, "action_mask"):
        return base.action_mask
    n = getattr(base, "action_space").n
    return np.ones(n, dtype=bool)


def make_env(file_path: Path, cfg: Config, worker_id: int) -> Callable[[], gym.Env]:
    """A single environment factory for training: with mask and random seed on reset"""
    def _thunk() -> gym.Env:
        # each subprocess gets unique but deterministic seed
        seed = cfg.seed + worker_id
        seed_everything(seed)

        jobs, opt_ms, _ = load_instance(str(file_path))
        base_env = GymJSSPEnv(
            jobs, # type: ignore
            render_mode=None,
            optimal_ms=opt_ms,
            reward_alpha=cfg.reward_alpha,
            reward_mode=cfg.reward_mode,
            feature_type=cfg.feature_type,
        )
        base_env.action_space.seed(seed)
        base_env.reset(seed=seed)

        env = ActionMasker(base_env, _mask_fn)
        return env
    return _thunk


def build_train_env(files: List[Path], cfg: Config) -> VecMonitor:
    """Multi-process + monitored vectorized training environment."""
    n_envs = cfg.n_envs
    assert n_envs <= len(files), "n_envs should not be larger than the number of available instances"
    env_fns = [make_env(files[i], cfg, i) for i in range(n_envs)]
    raw_vec = SubprocVecEnv(env_fns)         # parallel Collection
    vec = VecMonitor(raw_vec)                # record statistics for each episode
    return vec


def build_eval_env(files: List[Path], cfg: Config) -> VecMonitor:
    """
    Evaluation environment: Use DummyVecEnv to package multiple Monitor(env)s 
    so that evaluation statistics can be obtained in one step.
    """
    def make_thunk(fp: Path, idx: int) -> Callable[[], gym.Env]:
        def _thunk() -> gym.Env:
            # Ensure the same seed strategy as training
            seed = cfg.seed + idx
            seed_everything(seed)
            # The env returned by make_env is already ActionMasker(base_env)
            env = make_env(fp, cfg, worker_id=idx)()
            # Monitor is used to record the length of each episode, rewards, etc.
            return Monitor(env, filename=None)
        return _thunk

    # Throw all thunks to DummyVecEnv, and then use VecMonitor
    env_fns = [make_thunk(fp, i) for i, fp in enumerate(files)]
    vec = DummyVecEnv(env_fns)
    return VecMonitor(vec)
