import logging
import os
import time
import random
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Union, Any

import gymnasium as gym
import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from data_processing.data_processing import load_instance  
from utils.env_utils import make_env  


# ======================= Batch Switch Callback =======================
class InstanceSwitchCallback(BaseCallback):
    """
    Batch switching based on a dual guarantee of gap improvement using a sliding window 
    and a maximum episode count.
    - Stagnation is determined based on whether the average gap improvement 
      over the most recent episodes falls below a threshold.
    - A "maximum episode count" is added as a safety net.
    - When switching, the old VecEnv is safely shut down and a new training VecEnv is rebuilt.
    """
    def __init__(self, train_files: List[Path], cfg, verbose: int = 0):
        super().__init__(verbose)
        self.all_files: List[Path] = list(train_files)
        self.current_files: List[Path] = list(train_files)
        self.cfg = cfg

        self.batch_idx = 0
        self.episode_count = 0
        self.next_switch_ep = int(cfg.switch_episodes)
        self.last_switch_ts = 0

        self.gap_history: List[float] = []
        self.last_window_mean: Optional[float] = None
        self.no_improve_count = 0

        self._switching_lock = False

        # Cache instance optimal value
        self.opt_cache: Dict[str, float] = {}

    # ---------- util ----------
    def _safe_env_close(self, env):
        """Safely close the vector context to avoid leaks."""
        if env is None:
            return
        try:
            if hasattr(env, "close"):
                env.close()
            if hasattr(env, "env_method"):
                try:
                    env.env_method("close")
                except Exception:
                    pass
            if hasattr(env, "envs"):
                for e in env.envs:
                    try:
                        if hasattr(e, "close"):
                            e.close()
                    except Exception:
                        continue
        except Exception as e:
            logging.warning(f"[Switch] safe close warning: {e}")

    def _read_opt_ms(self, inst_path: Path) -> float:
        """Read the optimal makespan of the instance. 
        Return 1.0 if failure occurs to prevent division by zero."""
        key = str(inst_path)
        if key in self.opt_cache:
            return self.opt_cache[key]
        try:
            _, opt_ms, _ = load_instance(key)
            val = float(opt_ms) if opt_ms is not None else 1.0
            if val <= 0:
                val = 1.0
            self.opt_cache[key] = val
            return val
        except Exception as e:
            logging.error(f"[Switch] read opt ms failed for {key}: {e}")
            return 1.0

    # ---------- hooks ----------
    def _on_step(self) -> bool:
        # Count episodes: Read done from infos (SB3 VecEnv semantics)
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("terminal_observation") is not None or info.get("done", False):
                self.episode_count += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._switching_lock:
            return

        # Collect the gap of this rollout (using env info: makespan/optimal_ms)
        infos = self.locals.get("infos", [])
        new_gaps: List[float] = []
        for info in infos:
            if "makespan" in info and "optimal_ms" in info and info["optimal_ms"] > 0:
                new_gaps.append(info["makespan"] / info["optimal_ms"] - 1.0)
        if new_gaps:
            self.gap_history.extend(new_gaps)
            window = int(self.cfg.switch_gap_window)
            if len(self.gap_history) > window:
                self.gap_history = self.gap_history[-window:]
            cur_mean = float(np.mean(self.gap_history))
            if self.last_window_mean is not None:
                improve = self.last_window_mean - cur_mean
                if improve < float(self.cfg.switch_gap_threshold):
                    self.no_improve_count += 1
                else:
                    self.no_improve_count = 0
            self.last_window_mean = cur_mean

        # Trigger condition: episode limit or gap stagnation patience
        need_switch = False
        reason = ""
        if self.episode_count >= self.next_switch_ep:
            need_switch, reason = True, "max_episodes"
        if self.no_improve_count >= int(self.cfg.switch_patience):
            need_switch, reason = True, "gap_stagnation"

        logging.info(f"[Switch] ts={self.num_timesteps} epi={self.episode_count} "
                     f"no_improve={self.no_improve_count} need={need_switch} reason={reason}")

        if not need_switch:
            return

        self._switching_lock = True
        try:
            self._perform_switch_impl()
        except Exception as e:
            logging.error(f"[Switch] batch switch failed: {e}")
        finally:
            self._switching_lock = False

    def _perform_switch_impl(self) -> None:
        start_ts = self.last_switch_ts
        end_ts = self.num_timesteps
        for inst in self.current_files:
            logging.info(
                "[Switch] batch=%s instance=%s opt_ms=%s start_ts=%s end_ts=%s",
                self.batch_idx,
                inst,
                self._read_opt_ms(Path(inst)),
                start_ts,
                end_ts,
            )

        # Close the old environment
        old_env = self.model.get_env()
        self._safe_env_close(old_env)

        # Sampling a new batch
        k = min(len(self.all_files), int(self.cfg.n_envs))
        new_files = random.sample(self.all_files, k)

        # Rebuild a new VecEnv
        from utils.env_utils import build_train_env  
        new_env = build_train_env(new_files, self.cfg)
        self.model.set_env(new_env)

        # 纭繚 _last_obs 灏辩华
        reset_out = new_env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self.model._last_obs = obs  # type: ignore

        # Update internal state
        self.batch_idx += 1
        self.last_switch_ts = end_ts
        self.current_files = list(new_files)
        self.episode_count = 0
        self.no_improve_count = 0
        self.gap_history.clear()
        self.last_window_mean = None
        self.next_switch_ep = int(self.cfg.switch_episodes)

    def _on_training_end(self) -> None:
        """Log the interval of the last batch to the training log."""
        final_ts = self.num_timesteps
        if final_ts > self.last_switch_ts and self.current_files:
            for inst in self.current_files:
                logging.info(
                    "[Switch] final_batch=%s instance=%s opt_ms=%s start_ts=%s end_ts=%s",
                    self.batch_idx,
                    inst,
                    self._read_opt_ms(Path(inst)),
                    self.last_switch_ts,
                    final_ts,
                )


# ======================= Entropy Decay Callback =======================
class EntropyDecay(BaseCallback):
    """
    Two modes are supported:
    - power_decay: Power decay from start_coef to end_coef
    - warmup_decay: The warmup_ratio remains constant at start_coef, 
      then power decays thereafter
    """
    def __init__(
        self,
        total_timesteps: int,
        start_coef: float,
        end_coef: float = 0.005,
        power: float = 2.0,
        log_every: int = 512,
        verbose: int = 0,
        mode: str = "power_decay",
        warmup_ratio: float = 0.0,
    ):
        super().__init__(verbose)
        self.total_timesteps = float(total_timesteps)
        self.start_coef = float(start_coef)
        self.end_coef = float(end_coef)
        self.power = float(power)
        self.log_every = int(log_every)
        self.mode = str(mode)
        self.warmup_ratio = float(warmup_ratio)
        if self.mode not in {"power_decay", "warmup_decay"}:
            raise ValueError(f"Unknown entropy decay mode: {self.mode}")

    def _on_step(self) -> bool:
        progress = self.num_timesteps / max(1.0, self.total_timesteps)
        progress = min(progress, 1.0)
        new_coef = self.start_coef
        if self.mode == "power_decay":
            new_coef = self.end_coef + (self.start_coef - self.end_coef) * ((1.0 - progress) ** self.power)
        else:
            if progress < self.warmup_ratio:
                new_coef = self.start_coef
            else:
                dp = (progress - self.warmup_ratio) / (1.0 - self.warmup_ratio)
                dp = min(dp, 1.0)
                new_coef = self.end_coef + (self.start_coef - self.end_coef) * ((1.0 - dp) ** self.power)

        self.model.ent_coef = float(new_coef)  # type: ignore
        if self.n_calls % self.log_every == 0:
            try:
                wandb.log({"train/ent_coef": new_coef}, step=self.num_timesteps)
            except Exception:
                pass
        return True


# ======================= Step Callback =======================
class StepMetricsCallback(BaseCallback):
    """Lightweight step-level PPO metrics."""
    def __init__(self, freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.freq = int(freq)
        self._last_time = None
        self._last_step = None
        self._ema_sps: Optional[float] = None
        self._convex_branch_param: Optional[torch.nn.Parameter] = None
        self._convex_hook: Optional[Any] = None
        self._last_convex_grad: Optional[np.ndarray] = None
        self._last_convex_grad_total_norm: Optional[float] = None

    def _capture_convex_grad(self, grad: torch.Tensor) -> torch.Tensor:
        grad_cpu = grad.detach().float().cpu()
        self._last_convex_grad = grad_cpu.numpy().copy()
        self._last_convex_grad_total_norm = float(grad_cpu.norm().item())
        return grad

    def _attach_convex_hook(self) -> None:
        policy = getattr(self.model, "policy", None)
        encoder = getattr(policy, "encoder", None)
        fusion = getattr(encoder, "fusion", None)
        branch_logits = getattr(fusion, "branch_logits", None)
        if isinstance(branch_logits, torch.nn.Parameter):
            self._convex_branch_param = branch_logits
            self._convex_hook = branch_logits.register_hook(self._capture_convex_grad)
            logging.info("[Convex] enabled branch logits monitoring")

    def _detach_convex_hook(self) -> None:
        if self._convex_hook is not None:
            try:
                self._convex_hook.remove()
            except Exception:
                pass
        self._convex_hook = None

    def _on_training_start(self) -> None:
        self._attach_convex_hook()

    def _append_convex_metrics(self, light: Dict[str, Union[float, str]]) -> None:
        if self._convex_branch_param is None:
            return

        logits = self._convex_branch_param.detach().float().cpu()
        weights = torch.softmax(logits, dim=0)
        branch_names = ("job", "mach", "global")
        for idx, name in enumerate(branch_names):
            light[f"train/convex/logits/{name}"] = float(logits[idx].item())
            light[f"train/convex/weights/{name}"] = float(weights[idx].item())

        if self._last_convex_grad is None:
            return

        light["train/convex/grad_norm/total"] = float(
            self._last_convex_grad_total_norm if self._last_convex_grad_total_norm is not None
            else np.linalg.norm(self._last_convex_grad)
        )
        for idx, name in enumerate(branch_names):
            light[f"train/convex/grad/{name}"] = float(self._last_convex_grad[idx])
            light[f"train/convex/grad_norm/{name}"] = float(abs(self._last_convex_grad[idx]))

    def _on_step(self) -> bool:
        if self.n_calls % self.freq != 0:
            return True

        step = self.num_timesteps
        light: Dict[str, Union[float, str]] = {}

        lg = getattr(self.model.logger, "name_to_value", {})
        for key in [
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/entropy_loss",
            "train/clip_fraction",
            "train/approx_kl",
            "train/loss",
        ]:
            v = lg.get(key, None)
            if v is not None:
                light[key] = float(v)

        try:
            light["train/learning_rate"] = float(self.model.policy.optimizer.param_groups[0]["lr"])
        except Exception:
            pass

        # steps/s锛圗MA锛?
        now = time.time()
        if self._last_time is not None and self._last_step is not None:
            dt = now - self._last_time
            dstep = step - self._last_step
            if dt > 0 and dstep >= 0:
                inst = dstep / dt
                self._ema_sps = inst if self._ema_sps is None else 0.1 * inst + 0.9 * self._ema_sps
                light["perf/steps_per_sec"] = self._ema_sps
        self._last_time = now
        self._last_step = step

        self._append_convex_metrics(light)
        light["global/timestep"] = step
        wandb.log(light, step=step)
        return True

    def _on_training_end(self) -> None:
        self._detach_convex_hook()


# ======================= Episode Callbacks =======================
class TrainEpisodeCallback(BaseCallback):
    """Per-episode monitoring during training."""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [])
        if dones is not None:
            for idx, done in enumerate(dones):
                if done:
                    info = infos[idx]
                    wandb.log(
                        {
                            "train/episode/makespan": float(info["makespan"]),
                            "train/episode/optimality_gap": float(info["gap"]),
                            "global/timestep": self.num_timesteps,
                            "global/episode_index": self.episode_count,
                        },
                        step=self.num_timesteps,
                    )
                    self.episode_count += 1
        return True


class EvalTableLogger(BaseCallback):
    """
    Periodically perform a "single-instance, item-by-item evaluation" on a fixed eval set.
    Record the values 鈥嬧€?eval_step, instance, final_ms, optimal_ms, gap) in an appendable W&B table.
    Maintain a best_model.zip file based on the mean gap.
    """
    def __init__(self, eval_instances: List[Path], cfg, eval_freq: int, table_name: str = "eval_table", verbose: int = 0):
        super().__init__(verbose)
        self.eval_instances = list(eval_instances)
        self.cfg = cfg
        self.eval_freq = int(eval_freq)
        self.table_name = str(table_name)

        self.best_model_dir = Path(cfg.model_dir) / cfg.run_slug / "best_model"
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.best_mean_gap = float("inf")

        # List of prebuilt single environments (non-Vec)
        self.eval_envs: List[gym.Env] = [make_env(fp, cfg, i)() for i, fp in enumerate(self.eval_instances)]
        self.eval_columns = ["eval_step", "instance", "final_makespan", "optimal_makespan", "optimality_gap"]
        self._last_eval_step = 0

    @staticmethod
    def _safe_close_env(env: Optional[gym.Env]) -> None:
        if env is None:
            return
        try:
            if hasattr(env, "close"):
                env.close()
        except Exception as e:
            logging.warning(f"[Eval] safe close warning: {e}")

    @staticmethod
    def _new_gate_store() -> Dict[str, Union[float, np.ndarray]]:
        zeros = np.zeros(3, dtype=np.float64)
        return {
            "step_count": 0.0,
            "valid_token_count": 0.0,
            "valid_gate_sum": zeros.copy(),
            "valid_gate_sq_sum": zeros.copy(),
            "valid_gate_entropy_sum": 0.0,
            "valid_contrib_sum": zeros.copy(),
            "chosen_gate_sum": zeros.copy(),
            "chosen_gate_sq_sum": zeros.copy(),
            "chosen_gate_entropy_sum": 0.0,
            "chosen_contrib_sum": zeros.copy(),
            "chosen_minus_valid_sum": zeros.copy(),
        }

    @staticmethod
    def _as_numpy(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _accumulate_gate_store(self, store: Dict[str, Union[float, np.ndarray]], diag: Dict[str, torch.Tensor]) -> None:
        valid_count = self._as_numpy(diag["valid_count"]).reshape(-1).astype(np.float64)
        valid_gate_mean = self._as_numpy(diag["valid_gate_mean"]).reshape(-1, 3).astype(np.float64)
        valid_gate_sq_mean = self._as_numpy(diag["valid_gate_sq_mean"]).reshape(-1, 3).astype(np.float64)
        valid_gate_entropy_mean = self._as_numpy(diag["valid_gate_entropy_mean"]).reshape(-1).astype(np.float64)
        valid_contrib_mean = self._as_numpy(diag["valid_contrib_mean"]).reshape(-1, 3).astype(np.float64)
        chosen_gate = self._as_numpy(diag["chosen_gate"]).reshape(-1, 3).astype(np.float64)
        chosen_gate_entropy = self._as_numpy(diag["chosen_gate_entropy"]).reshape(-1).astype(np.float64)
        chosen_contrib = self._as_numpy(diag["chosen_contrib"]).reshape(-1, 3).astype(np.float64)
        chosen_minus_valid = self._as_numpy(diag["chosen_minus_valid"]).reshape(-1, 3).astype(np.float64)

        batch_steps = float(valid_count.shape[0])
        total_valid = float(valid_count.sum())
        store["step_count"] = float(store["step_count"]) + batch_steps
        store["valid_token_count"] = float(store["valid_token_count"]) + total_valid
        store["valid_gate_sum"] = np.asarray(store["valid_gate_sum"]) + (valid_gate_mean * valid_count[:, None]).sum(axis=0)
        store["valid_gate_sq_sum"] = np.asarray(store["valid_gate_sq_sum"]) + (valid_gate_sq_mean * valid_count[:, None]).sum(axis=0)
        store["valid_gate_entropy_sum"] = float(store["valid_gate_entropy_sum"]) + float((valid_gate_entropy_mean * valid_count).sum())
        store["valid_contrib_sum"] = np.asarray(store["valid_contrib_sum"]) + (valid_contrib_mean * valid_count[:, None]).sum(axis=0)
        store["chosen_gate_sum"] = np.asarray(store["chosen_gate_sum"]) + chosen_gate.sum(axis=0)
        store["chosen_gate_sq_sum"] = np.asarray(store["chosen_gate_sq_sum"]) + (chosen_gate ** 2).sum(axis=0)
        store["chosen_gate_entropy_sum"] = float(store["chosen_gate_entropy_sum"]) + float(chosen_gate_entropy.sum())
        store["chosen_contrib_sum"] = np.asarray(store["chosen_contrib_sum"]) + chosen_contrib.sum(axis=0)
        store["chosen_minus_valid_sum"] = np.asarray(store["chosen_minus_valid_sum"]) + chosen_minus_valid.sum(axis=0)

    @staticmethod
    def _finalize_gate_store(store: Dict[str, Union[float, np.ndarray]]) -> Dict[str, float]:
        branch_names = ("job", "mach", "global")
        metrics: Dict[str, float] = {}
        valid_token_count = float(store["valid_token_count"])
        step_count = float(store["step_count"])

        if valid_token_count > 0:
            valid_gate_mean = np.asarray(store["valid_gate_sum"]) / valid_token_count
            valid_gate_sq_mean = np.asarray(store["valid_gate_sq_sum"]) / valid_token_count
            valid_gate_std = np.sqrt(np.maximum(valid_gate_sq_mean - valid_gate_mean ** 2, 0.0))
            valid_contrib_mean = np.asarray(store["valid_contrib_sum"]) / valid_token_count
            metrics["eval/gate/valid_entropy_mean"] = float(store["valid_gate_entropy_sum"]) / valid_token_count
            for idx, name in enumerate(branch_names):
                metrics[f"eval/gate/valid_mean/{name}"] = float(valid_gate_mean[idx])
                metrics[f"eval/gate/valid_std/{name}"] = float(valid_gate_std[idx])
                metrics[f"eval/fusion/contrib_norm/valid_mean/{name}"] = float(valid_contrib_mean[idx])

        if step_count > 0:
            chosen_gate_mean = np.asarray(store["chosen_gate_sum"]) / step_count
            chosen_gate_sq_mean = np.asarray(store["chosen_gate_sq_sum"]) / step_count
            chosen_gate_std = np.sqrt(np.maximum(chosen_gate_sq_mean - chosen_gate_mean ** 2, 0.0))
            chosen_contrib_mean = np.asarray(store["chosen_contrib_sum"]) / step_count
            chosen_minus_valid_mean = np.asarray(store["chosen_minus_valid_sum"]) / step_count
            metrics["eval/gate/chosen_entropy_mean"] = float(store["chosen_gate_entropy_sum"]) / step_count
            metrics["eval/diagnostics/decision_steps"] = step_count
            metrics["eval/diagnostics/valid_token_count"] = valid_token_count
            for idx, name in enumerate(branch_names):
                metrics[f"eval/gate/chosen_mean/{name}"] = float(chosen_gate_mean[idx])
                metrics[f"eval/gate/chosen_std/{name}"] = float(chosen_gate_std[idx])
                metrics[f"eval/gate/chosen_minus_valid/{name}"] = float(chosen_minus_valid_mean[idx])
                metrics[f"eval/fusion/contrib_norm/chosen_mean/{name}"] = float(chosen_contrib_mean[idx])

        return metrics

    @staticmethod
    def _unwrap(env):
        u = env
        while hasattr(u, "env"):
            u = getattr(u, "env")
        return u

    def _run_one(self, env: gym.Env, gate_store: Optional[Dict[str, Union[float, np.ndarray]]] = None):
        base = self._unwrap(env)
        try:
            obs, _ = env.reset()
            done = False
            while not done:
                mask = env.action_masks()  # type: ignore
                policy = getattr(self.model, "policy", None)
                if policy is not None and hasattr(policy, "collect_step_diagnostics"):
                    diag = policy.collect_step_diagnostics(obs, action_masks=mask, deterministic=True)
                    if gate_store is not None:
                        self._accumulate_gate_store(gate_store, diag)
                    action = int(self._as_numpy(diag["actions"]).reshape(-1)[0])
                else:
                    action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)  # type: ignore
                obs, _, terminated, truncated, _ = env.step(int(action))
                done = bool(terminated or truncated)

            final_ms = float(base.get_makespan())
            optimal_ms = float(getattr(base, "optimal_ms"))
            gap = final_ms / optimal_ms - 1.0
            return final_ms, optimal_ms, gap
        except Exception:
            wandb.log({"eval/errors": 1}, commit=False)
            try:
                wandb.summary["last_eval_exception"] = traceback.format_exc()[-1000:]  # type: ignore
            except Exception:
                pass
            return None

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step - self._last_eval_step < self.eval_freq:
            return True

        gaps_this_eval: List[float] = []
        current_rows: List[List[Union[int, str, float]]] = []
        gate_store = self._new_gate_store()
        for fp, env in zip(self.eval_instances, self.eval_envs):
            res = self._run_one(env, gate_store=gate_store)
            if res is None:
                continue
            final_ms, optimal_ms, gap = res
            current_rows.append([step, str(fp), final_ms, optimal_ms, gap])
            gaps_this_eval.append(gap)

        if current_rows:
            try:
                if wandb.run is not None:
                    media_dir = os.path.join(wandb.run.dir, "media", "table")
                    os.makedirs(media_dir, exist_ok=True)
                table = wandb.Table(columns=self.eval_columns, data=current_rows)
                wandb.log({self.table_name: table}, step=step)
            except Exception:
                try:
                    wandb.log({self.table_name + "_rows": current_rows}, step=step)
                except Exception:
                    pass

        mean_gap = float(np.nanmean(gaps_this_eval)) if gaps_this_eval else float("nan")
        eval_metrics = {"eval/online_mean_gap": mean_gap}
        eval_metrics.update(self._finalize_gate_store(gate_store))
        wandb.log(eval_metrics, step=step)
        if np.isfinite(mean_gap) and mean_gap < self.best_mean_gap:
            self.best_mean_gap = mean_gap
            best_path = self.best_model_dir / "best_model.zip"
            self.model.save(best_path)
            try:
                wandb.summary["best/mean_gap"] = mean_gap  # type: ignore
                wandb.summary["best/step"] = step          # type: ignore
            except Exception:
                pass
            wandb.log({"eval/best_mean_gap": mean_gap}, step=step)

        self._last_eval_step = step
        return True

    def close(self) -> None:
        for env in self.eval_envs:
            self._safe_close_env(env)
        self.eval_envs.clear()

    def _on_training_end(self) -> None:
        self.close()


# ======================= Rollout Metrics =======================
class RolloutMetricsCallback(BaseCallback):
    """Collect the distribution and trend of a rollout, as well as some training signals for PPO."""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._gaps: List[float] = []

    def _on_rollout_start(self) -> None:
        self._gaps.clear()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [])
        if dones is not None:
            for idx, done in enumerate(dones):
                if done:
                    self._gaps.append(float(infos[idx]["gap"]))
        return True

    def _on_rollout_end(self) -> None:
        step = self.num_timesteps
        if not self._gaps:
            return

        buf = self.model.rollout_buffer  # type: ignore
        vals = torch.as_tensor(buf.values).float().view(-1)
        rets = torch.as_tensor(buf.returns).float().view(-1)
        mse = torch.mean((rets - vals) ** 2).item()

        wandb.log(
            {
                "train/rollout/gap_histogram": wandb.Histogram(self._gaps),
                "train/rollout/mean_gap": float(np.mean(self._gaps)),
                "train/rollout/value_mse": mse,
            },
            step=step,
        )


# ======================= Callback 宸ュ巶 =======================
class CallbackListWithEval(CallbackList):
    """Thin wrapper around CallbackList kept for compatibility with training scripts."""

    def close(self) -> None:
        for callback in self.callbacks:
            close_fn = getattr(callback, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as e:
                    logging.warning(f"[CallbackList] close warning: {e}")


def build_callbacks(cfg, train_instances: List[Path], eval_instances: List[Path]) -> CallbackListWithEval:
    switch_cb = InstanceSwitchCallback(train_instances, cfg)
    ent_cb = EntropyDecay(
        total_timesteps=cfg.total_timesteps,
        start_coef=cfg.entropy_coef,
        end_coef=cfg.end_entropy_coef,
        power=cfg.entropy_power,
        log_every=cfg.eval_freq,
        mode=cfg.entropy_mode,
        warmup_ratio=cfg.entropy_warmup_ratio,
    )

    ckpt_dir = Path(cfg.model_dir) / cfg.run_slug / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = CheckpointCallback(save_freq=cfg.save_freq, save_path=str(ckpt_dir), name_prefix="ckpt")

    step_cb = StepMetricsCallback(freq=cfg.log_every)
    train_epi_cb = TrainEpisodeCallback()
    train_rollout_cb = RolloutMetricsCallback()
    eval_table_cb = EvalTableLogger(eval_instances=eval_instances, cfg=cfg, eval_freq=cfg.eval_freq,
                                    table_name="eval_table")

    cb_list = CallbackListWithEval([switch_cb, ent_cb, ckpt_cb, eval_table_cb,
                                    step_cb, train_epi_cb, train_rollout_cb])
    return cb_list

