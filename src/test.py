"""
Evaluation script for the JSSP MaskablePPO agent.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import wandb
from gymnasium import spaces
from tqdm import tqdm

from sb3_contrib import MaskablePPO
from stable_baselines3.common.save_util import load_from_zip_file

from custom_policy import JSSPPolicy
from data_processing.data_processing import load_instance, parse_benchmark_by_name
from gym_jssp_env import GymJSSPEnv
from utils.bench_optimality import OPTIMAL_MAKESPAN_LOOKUP
from utils.render_utils import render_gantt


def _sync_device(device: torch.device):
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
    except Exception:
        pass


def _reset_peak_mem(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return float("nan")


def _safe_close_env(env: Optional[gym.Env]):
    if env is None:
        return
    try:
        if hasattr(env, "close"):
            env.close()
    except Exception:
        pass


def _safe_close_model(model: Optional[MaskablePPO]):
    if model is None:
        return
    try:
        env = model.get_env()
    except Exception:
        env = None
    _safe_close_env(env)


def _release_device_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _obs_to_torch(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in obs.items():
        t = torch.as_tensor(v, device=device)
        if t.dim() == v.ndim:
            t = t.unsqueeze(0)
        out[k] = t
    return out


def make_env_from_file(file_path: Path, feature_type="full", strict_opt: bool = False):
    known = {"ABZ", "DMU", "FT", "ORB", "SWV", "TA", "YN"}

    family = None
    for anc in file_path.parents:
        name = anc.name.upper()
        if name in known:
            family = name
            break

    if family is None:
        jobs_data, opt_ms_hdr, _ = load_instance(str(file_path))
        if strict_opt and not (opt_ms_hdr and float(opt_ms_hdr) > 0):
            raise ValueError(f"[STRICT] Missing valid opt= in header for synthetic instance: {file_path}")
        opt_ms = (
            float(opt_ms_hdr)
            if (opt_ms_hdr and float(opt_ms_hdr) > 0)
            else sum(d for job in jobs_data for _, d in job)
        )
        jobs = [[(int(m), float(d)) for (m, d) in job] for job in jobs_data]
        dataset_family = "SYNTHETIC"
    else:
        jobs = parse_benchmark_by_name(str(file_path), family)
        opt_ms = OPTIMAL_MAKESPAN_LOOKUP.get(file_path.stem)
        if strict_opt and opt_ms is None:
            raise KeyError(
                f"[STRICT] No optimal makespan found in lookup for benchmark instance: {family}/{file_path.stem}"
            )
        if opt_ms is None:
            opt_ms = sum(d for job in jobs for _, d in job)
        dataset_family = family

    n_jobs = len(jobs)
    n_machines = len(jobs[0]) if jobs else 0
    env = GymJSSPEnv(jobs, optimal_ms=float(opt_ms), render_mode=None, feature_type=feature_type)
    return env, float(opt_ms), dataset_family, n_jobs, n_machines


def create_model_for_env(
    env: gym.Env,
    policy_kwargs: Dict[str, Any],
    policy_params: Dict[str, Any],
) -> MaskablePPO:
    model = MaskablePPO(
        policy=JSSPPolicy,
        env=env,
        device="auto",
        policy_kwargs=policy_kwargs,
    )
    policy_params.pop("action_net.weight", None)
    policy_params.pop("action_net.bias", None)
    model.policy.load_state_dict(policy_params, strict=False)
    return model


def evaluate_model_on_files(
    model: MaskablePPO,
    files: List[Path],
    render: bool = False,
    gantt_save_dir: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    feature_type: str = "full",
    strict_opt: bool = False,
) -> Dict[str, Any]:
    device = model.device
    results: List[Dict[str, Any]] = []

    for p in tqdm(files, desc=f"Evaluating {dataset_name or ''}".strip()):
        env = None
        try:
            env, opt_ms, family, n_jobs, n_machines = make_env_from_file(
                p,
                feature_type=feature_type,
                strict_opt=strict_opt,
            )
            dataset_tag = f"{family}_{n_jobs}x{n_machines}"

            _sync_device(device)
            _reset_peak_mem(device)
            t0 = time.perf_counter()

            with torch.inference_mode():
                obs, info = env.reset()
                done = False
                while not done:
                    obs_t = _obs_to_torch(obs, device=device)
                    mask = env.action_masks() if hasattr(env, "action_masks") else env.action_mask
                    mask_np = np.asarray(mask, dtype=bool).reshape(1, -1)

                    dist = model.policy.get_distribution(obs_t, action_masks=mask_np)
                    action = dist.get_actions(deterministic=True)
                    action_np = action.detach().cpu().numpy().reshape(-1)

                    obs, _, terminated, truncated, info = env.step(int(action_np[0]))
                    done = bool(terminated or truncated)

            _sync_device(device)
            elapsed = time.perf_counter() - t0
            peak_mem = _peak_mem_mb(device)

            makespan = float(info.get("makespan", env.get_makespan()))
            gap = float(info.get("gap", makespan / float(opt_ms) - 1.0))

            if render and gantt_save_dir is not None:
                try:
                    dest = gantt_save_dir / dataset_tag
                    dest.mkdir(parents=True, exist_ok=True)
                    render_gantt(env, save_path=str(dest / f"{p.stem}.png"))
                except Exception:
                    pass

            results.append(
                {
                    "file": str(p),
                    "instance": p.stem,
                    "optimal_ms": float(opt_ms),
                    "makespan": makespan,
                    "gap": gap,
                    "solve_time_ms": elapsed * 1000.0,
                    "peak_mem_mb": peak_mem,
                    "dataset": dataset_name,
                    "dataset_tag": dataset_tag,
                    "n_jobs": n_jobs,
                    "n_machines": n_machines,
                }
            )
        finally:
            _safe_close_env(env)

    if len(results) == 0:
        return {"summary": {}, "details": []}

    arr_gap = np.array([r["gap"] for r in results], dtype=float)
    arr_ms = np.array([r["makespan"] for r in results], dtype=float)
    arr_t = np.array([r["solve_time_ms"] for r in results], dtype=float)
    pct_opt = float((arr_gap <= 1e-8).mean())
    vals = np.array([r["peak_mem_mb"] for r in results], dtype=float)
    peak_mem_mb_max = float(np.nanmax(vals)) if np.isfinite(vals).any() else float("nan")

    summary = {
        "mean_gap": float(arr_gap.mean()),
        "median_gap": float(np.median(arr_gap)),
        "std_gap": float(arr_gap.std(ddof=0)),
        "pct_opt": pct_opt,
        "cmax_mean": float(arr_ms.mean()),
        "cmax_median": float(np.median(arr_ms)),
        "solve_time_ms_mean": float(arr_t.mean()),
        "solve_time_ms_median": float(np.median(arr_t)),
        "solve_time_ms_std": float(arr_t.std(ddof=0)),
        "peak_mem_mb_max": peak_mem_mb_max,
    }
    return {"summary": summary, "details": results}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Greedy evaluation for JSSP MaskablePPO")
    p.add_argument("--model_path", type=str, required=True, help="Path to .zip (SB3) model.")
    p.add_argument("--test_dirs", type=str, nargs="+", required=True, help="One or more dataset directories.")
    p.add_argument(
        "--feature_type",
        type=str,
        default=None,
        help="Manually override feature_type for the env, e.g., 'full' or 'static_only'.",
    )
    p.add_argument("--project", type=str, default="JSSP", help="W&B project name.")
    p.add_argument("--entity", type=str, default=None, help="W&B entity (optional).")
    p.add_argument("--name", type=str, default=None, help="Run name in W&B.")
    p.add_argument("--note", type=str, default="", help="Free-form note (e.g., 'per-variant tuned; reference only').")
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B logging.")
    p.add_argument("--render", action="store_true", help="Save gantt charts if env supports.")
    p.add_argument(
        "--strict-opt",
        action="store_true",
        help="Require true optimal makespan: synthetic uses file header opt=; benchmarks use lookup table. Fail if missing.",
    )
    p.add_argument("--gantt_dir", type=str, default=None)
    p.set_defaults(strict_opt=True)
    p.add_argument("--no-strict-opt", dest="strict_opt", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    test_dirs = [Path(d) for d in args.test_dirs]
    run_name = args.name or f"eval_{int(time.time())}"

    assert model_path.exists(), f"Model not found: {model_path}"
    for d in test_dirs:
        assert d.exists() and d.is_dir(), f"Test dir not found: {d}"

    wandb_mode = "disabled" if args.no_wandb else "online"
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        config=vars(args),
        job_type="evaluation",
        mode=wandb_mode,
    )

    if args.note:
        wandb.config.update({"note": args.note}, allow_val_change=True)
    if torch.cuda.is_available():
        wandb.config.update({"gpu_name": torch.cuda.get_device_name(0)}, allow_val_change=True)

    print(f"Loading model parameters from: {model_path}")
    data, params, _ = load_from_zip_file(model_path, device="auto")
    assert data is not None and params is not None, "Failed to load data/params from model file."

    if args.feature_type:
        feature_type_to_use = args.feature_type
        print(f"Using manually specified feature_type: '{feature_type_to_use}'")
    else:
        feature_type_to_use = data.get("feature_type", "full")
        print(f"Using feature_type from model file (fallback): '{feature_type_to_use}'")

    policy_kwargs = data["policy_kwargs"]
    policy_params = params["policy"]
    assert isinstance(policy_params, dict)
    print("Model parameters loaded successfully.")
    print("-" * 60)

    all_results_details = []

    for ds_dir in test_dirs:
        dataset_name = ds_dir.name
        txt_files = sorted([p for p in ds_dir.rglob("*.txt") if p.is_file()])
        if not txt_files:
            print(f"[WARN] No txt instances in: {ds_dir}")
            continue

        print(f"\n== {dataset_name} ==")
        print(f"Creating a new model instance for size of '{dataset_name}'...")

        temp_env_for_shape = None
        current_model = None
        try:
            temp_env_for_shape, *_ = make_env_from_file(
                txt_files[0],
                feature_type=feature_type_to_use,
                strict_opt=args.strict_opt,
            )
            current_model = create_model_for_env(temp_env_for_shape, policy_kwargs, policy_params.copy())
            assert isinstance(current_model.observation_space, spaces.Dict)
            print(f"Model instance created with observation space: {current_model.observation_space['features'].shape}")

            gantt_dir = Path("gantt_charts") / run_name / dataset_name
            eval_res = evaluate_model_on_files(
                current_model,
                txt_files,
                render=args.render,
                gantt_save_dir=gantt_dir,
                dataset_name=dataset_name,
                feature_type=feature_type_to_use,
                strict_opt=args.strict_opt,
            )

            print(f"Mean gap:  {eval_res['summary']['mean_gap']:.4f}")

            df_ds = pd.DataFrame(eval_res["details"])
            tag = str(df_ds["dataset_tag"].iloc[0]) if not df_ds.empty else dataset_name

            s = eval_res["summary"]
            wandb.log(
                {
                    f"eval/{tag}/mean_gap": s.get("mean_gap"),
                    f"eval/{tag}/median_gap": s.get("median_gap"),
                    f"eval/{tag}/std_gap": s.get("std_gap"),
                    f"eval/{tag}/pct_opt": s.get("pct_opt"),
                    f"eval/{tag}/cmax_mean": s.get("cmax_mean"),
                    f"eval/{tag}/cmax_median": s.get("cmax_median"),
                    f"eval/{tag}/solve_time_ms_mean": s.get("solve_time_ms_mean"),
                    f"eval/{tag}/solve_time_ms_median": s.get("solve_time_ms_median"),
                    f"eval/{tag}/solve_time_ms_std": s.get("solve_time_ms_std"),
                    f"eval/{tag}/peak_mem_mb_max": s.get("peak_mem_mb_max"),
                }
            )

            csv_ds = Path(f"evaluation_details__{run_name}__{tag}.csv")
            df_ds.to_csv(csv_ds, index=False)
            print(f"[Saved] {csv_ds.resolve()}")
            all_results_details.extend(eval_res["details"])
        finally:
            _safe_close_model(current_model)
            _safe_close_env(temp_env_for_shape)
            _release_device_cache()

    if all_results_details:
        results_df = pd.DataFrame(all_results_details)
        out_csv = Path(f"evaluation_details__{run_name}__ALL.csv")
        results_df.to_csv(out_csv, index=False)
        print(f"\nSaved all-details to {out_csv.resolve()}")

    wandb.finish()
    _release_device_cache()
    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()

