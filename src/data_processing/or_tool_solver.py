"""
OR-Tools CP-SAT solver for JSSP instances stored in the text format:
    "<J> <M> [opt=<ms>] [is_opt=<0/1>]"
    followed by J lines, each with 2*M integers: (machine_id, duration)...

- Auto-detects and normalizes 1-based machine ids to 0-based.
- Builds classic NoOverlap interval model with job precedence constraints.
- Returns (makespan, is_proven_optimal) with robust handling for FEASIBLE status.

This module is intentionally self-contained (parses the instance itself) to avoid
circular imports with data_processing.py.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import os
from ortools.sat.python import cp_model


JobsData = List[List[Tuple[int, int]]]


def _parse_instance(file_path: str) -> JobsData:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError(f"Empty file: {file_path}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid header: {lines[0]}")
    try:
        n_jobs = int(header[0])
        n_machines = int(header[1])
    except Exception as e:
        raise ValueError(f"Invalid numeric fields in header: {lines[0]}") from e

    data = lines[1:]

    raw_tokens: List[str] = []
    for ln in data:
        raw_tokens.extend(ln.split())

    if len(raw_tokens) != n_jobs * 2 * n_machines:
        raise ValueError("Body size mismatch: expected J lines of 2*M integers.")

    nums = [int(round(float(x))) for x in raw_tokens]

    machine_ids = nums[::2]
    is_one_based = (max(machine_ids) == n_machines)

    jobs_data: JobsData = []
    idx = 0
    for _ in range(n_jobs):
        job_ops: List[Tuple[int, int]] = []
        for _ in range(n_machines):
            m = nums[idx]; d = nums[idx + 1]
            idx += 2
            if is_one_based:
                m -= 1
            if not (0 <= m < n_machines):
                raise ValueError(f"Machine id {m} out of range.")
            if d <= 0:
                raise ValueError("Durations must be positive integers.")
            job_ops.append((m, d))
        jobs_data.append(job_ops)

    return jobs_data


def solve_jssp_with_ortools(
    instance_path: str,
    time_limit_s: float = 600.0,
    workers: Optional[int] = None,
    random_seed: Optional[int] = None,
    require_optimal: bool = False,
) -> Tuple[int, bool]:
    """
    Solve a JSSP instance with CP-SAT and return (makespan, is_proven_optimal).
    If status == FEASIBLE, we return best found makespan and is_proven_optimal=False
    unless the relative gap to best bound is ~0 with a positive bound.

    Parameters
    ----------
    instance_path : str
        Path to instance file.
    time_limit_s : float
        Solver time limit per instance (seconds).
    workers : Optional[int]
        num_search_workers. None => os.cpu_count() or 1.
    random_seed : Optional[int]
        Random seed for CP-SAT. None => solver default.
    require_optimal : bool
        If True, raise RuntimeError unless CP-SAT proves optimality.

    Returns
    -------
    (makespan:int, is_proven_optimal:bool)

    Raises
    ------
    RuntimeError if no solution is found (INFEASIBLE / MODEL_INVALID / UNKNOWN without solution).
    """
    jobs_data = _parse_instance(instance_path)
    n_jobs = len(jobs_data)
    n_machines = len(jobs_data[0]) if n_jobs > 0 else 0

    # Horizon upper bound: sum of all durations (safe)
    horizon = sum(d for job in jobs_data for (_, d) in job)

    model = cp_model.CpModel()

    # Variables: for each op, create start, end, interval
    # Also keep per-machine intervals to add NoOverlap
    all_tasks = {}  # (j, o) -> (start, end, interval)
    machine_to_intervals: List[List[cp_model.IntervalVar]] = [[] for _ in range(n_machines)]

    for j, job in enumerate(jobs_data):
        for o, (m, d) in enumerate(job):
            start = model.NewIntVar(0, horizon, f"start_{j}_{o}")
            end = model.NewIntVar(0, horizon, f"end_{j}_{o}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{j}_{o}")
            all_tasks[(j, o)] = (start, end, interval)
            machine_to_intervals[m].append(interval)

    # Job precedence: op_k finishes before op_{k+1} starts
    for j, job in enumerate(jobs_data):
        for o in range(len(job) - 1):
            _, end_k, _ = all_tasks[(j, o)]
            start_next, _, _ = all_tasks[(j, o + 1)]
            model.Add(start_next >= end_k)

    # Machine no-overlap
    for m in range(n_machines):
        model.AddNoOverlap(machine_to_intervals[m])

    # Objective: minimize makespan = max end time of last operation for each job (or all ends)
    job_ends = [all_tasks[(j, len(jobs_data[j]) - 1)][1] for j in range(n_jobs)]
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = workers if workers is not None else (os.cpu_count() or 1)
    if random_seed is not None:
        solver.parameters.random_seed = int(random_seed)
    # quieter logs by default; feel free to change
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        return int(solver.ObjectiveValue()), True

    if status == cp_model.FEASIBLE:
        ms = int(solver.ObjectiveValue())
        bound = solver.BestObjectiveBound()
        # Only if a positive bound exists and the relative gap is ~0, consider it optimal.
        if bound is not None and bound > 0:
            rel_gap = (ms - bound) / bound
            if rel_gap <= 1e-6:
                return ms, True
        if require_optimal:
            raise RuntimeError(
                "CP-SAT found a feasible schedule but did not prove optimality "
                f"(makespan={ms}, best_bound={bound})."
            )
        return ms, False

    # No solution found
    raise RuntimeError(f"CP-SAT did not find a feasible solution (status={status}).")
