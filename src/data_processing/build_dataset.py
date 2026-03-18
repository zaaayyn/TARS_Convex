"""
Dataset builder entry.
Generates JSSP datasets with classic setting: each job visits every machine exactly once
in a permuted order; processing times are integers in [1, 15].

Only instances whose optimal makespan is proven by OR-Tools are kept. If CP-SAT
cannot prove optimality for a candidate instance within the time limit, that
candidate is discarded and a new one is generated instead.

Example:
    python build_dataset.py --train_size 1000 --test_size 200 --jobs 6 --machines 6 \
        --out jssp_dataset_6x6 --time_limit 30
"""

from __future__ import annotations
import argparse

if __package__:
    from .data_processing import generate_dataset
else:
    from data_processing import generate_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate JSSP dataset (classic).")
    p.add_argument("--train_size", type=int, default=10_000, help="Number of training instances (>0).")
    p.add_argument("--test_size", type=int, default=100, help="Number of testing instances (>=0).")
    p.add_argument("--jobs", type=int, default=6, help="Number of jobs (>0).")
    p.add_argument("--machines", type=int, default=6, help="Number of machines (>0).")
    p.add_argument("--out", type=str, default="jssp_dataset_6x6", help="Output directory.")
    p.add_argument("--seed", type=int, default=87, help="Base random seed.")
    p.add_argument("--min_time", type=int, default=1, help="Minimum processing time (must be 1).")
    p.add_argument("--max_time", type=int, default=15, help="Maximum processing time (must be 15).")
    p.add_argument(
        "--solve_opt",
        action="store_true",
        help="Deprecated. OR-Tools optimal solving is now always enabled.",
    )
    p.add_argument("--time_limit", type=float, default=30.0, help="CP-SAT time limit per instance (seconds).")
    p.add_argument("--workers", type=int, default=None, help="CP-SAT num_search_workers (None => CPU count).")
    p.add_argument("--solver_seed", type=int, default=None, help="CP-SAT random_seed (optional).")
    return p.parse_args()


def main():
    args = parse_args()

    # Enforce the requirement: artificial processing times must be in [1, 15]
    if not (args.min_time == 1 and args.max_time == 15):
        raise ValueError("For synthetic data, processing times MUST be in [1, 15]. "
                         "Please use --min_time 1 --max_time 15.")
    if args.solve_opt:
        print("[note] --solve_opt is no longer required; optimal solving is always enabled.")

    generate_dataset(
        train_size=args.train_size,
        test_size=args.test_size,
        num_jobs=args.jobs,
        num_machines=args.machines,
        output_dir=args.out,
        seed=args.seed,
        min_time=args.min_time,
        max_time=args.max_time,
        solve_opt=True,
        time_limit_s=args.time_limit,
        workers=args.workers,
        solver_seed=args.solver_seed,
    )


if __name__ == "__main__":
    main()
