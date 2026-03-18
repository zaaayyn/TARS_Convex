"""
Data processing utilities for JSSP:
- Synthetic instance generation (classic: each job visits each machine exactly once in a permuted order)
- Robust load/save
- Optional benchmark parser with 0/1-based machine index normalization
- Dataset builder (train/test), supporting test_size=0

All durations are integers; for synthetic data, durations are enforced to [1, 15].
Synthetic datasets are kept only when OR-Tools proves an optimal makespan.
Header format:
    "<num_jobs> <num_machines> opt=<makespan> is_opt=1"
"""

from __future__ import annotations
import os
import re
import random
from typing import List, Tuple, Optional, Union


JobsData = List[List[Tuple[int, int]]]  # jobs_data[job][op] = (machine_id, duration)


# ---------------------------
# Core I/O helpers
# ---------------------------

def save_jssp_instance_to_txt(jobs_data: JobsData, filepath: str,
                              opt_ms: Optional[int] = None, is_opt: Optional[bool] = None) -> None:
    """
    Save an instance to text. If opt_ms/is_opt are given, they will be embedded in header.
    """
    num_jobs = len(jobs_data)
    if num_jobs == 0:
        raise ValueError("jobs_data is empty.")
    num_machines = len(jobs_data[0])
    if not all(len(job) == num_machines for job in jobs_data):
        raise ValueError("All jobs must have the same number of operations (num_machines).")

    header = f"{num_jobs} {num_machines}"
    if opt_ms is not None and is_opt is not None:
        header = f"{header} opt={int(opt_ms)} is_opt={1 if is_opt else 0}"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for job in jobs_data:
            # force ints, machine_id and duration must be ints
            line = " ".join(f"{int(m)} {int(d)}" for m, d in job)
            f.write(line + "\n")

HEADER_RE = re.compile(
    r"^\s*#*\s*(\d+)\s+(\d+)(?:\s+opt\s*=\s*(\d+))?(?:\s+is_opt\s*=\s*(\d+))?\s*$",
    re.IGNORECASE
)

def load_instance(file_path: str) -> Tuple[JobsData, Optional[int], Optional[bool]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [ln.strip() for ln in f if ln.strip()]

    if not raw_lines:
        raise ValueError(f"Empty file: {file_path}")

    # 1) Locate the actual header line (skipping comments/separator lines)
    header_idx = -1
    n_jobs = n_machines = None
    opt_ms: Optional[int] = None
    is_opt: Optional[bool] = None

    for i, ln in enumerate(raw_lines):
        # Skip comments or separator lines
        if ln.startswith("#") or set(ln) <= {"+", "-", "=", "*", " "}:
            continue
        m = HEADER_RE.match(ln)
        if m:
            n_jobs = int(m.group(1))
            n_machines = int(m.group(2))
            if m.group(3) is not None:
                opt_ms = int(m.group(3))
            if m.group(4) is not None:
                is_opt = (m.group(4) in {"1", "true", "t", "y", "yes"})
            header_idx = i
            break

    if header_idx < 0 or n_jobs is None or n_machines is None:
        raise ValueError(f"Could not find header like '<n_jobs> <n_machines> [opt=...] [is_opt=...]' in: {file_path}")

    # 2) Read n_jobs lines, each line contains 2*n_machines integer (machine, duration) pairs
    ptr = header_idx + 1
    jobs_data: JobsData = []
    for j in range(n_jobs):
        if ptr + j >= len(raw_lines):
            raise ValueError(f"Unexpected EOF: need {n_jobs} job-lines after header in {file_path}")
        line = raw_lines[ptr + j]
        # Skip possible empty/comment/separator lines
        while line.startswith("#") or set(line) <= {"+", "-", "=", "*", " "}:
            ptr += 1
            if ptr + j >= len(raw_lines):
                raise ValueError(f"Unexpected EOF while skipping comments in {file_path}")
            line = raw_lines[ptr + j]

        parts = [int(x) for x in line.split()]
        expected = 2 * n_machines
        if len(parts) != expected:
            raise ValueError(
                f"Line {ptr+j} expects {expected} ints (machine,duration pairs), "
                f"got {len(parts)} in {file_path!s}"
            )

        job: List[Tuple[int, int]] = []
        macs = parts[::2]
        is_one_based = (max(macs) == n_machines)  # 1..m?
        for k in range(0, expected, 2):
            mac = parts[k] - (1 if is_one_based else 0)
            dur = parts[k + 1]
            if not (0 <= mac < n_machines):
                raise ValueError(f"Machine id {mac} out of range after normalization (0..{n_machines-1}).")
            if dur <= 0:
                raise ValueError("Durations must be positive integers.")
            job.append((mac, dur))
        jobs_data.append(job)

    return jobs_data, opt_ms, is_opt


def _load_or_tool_solver():
    """Import the local OR-Tools helper in both script and package modes."""
    if __package__:
        from . import or_tool_solver as solver_module  # type: ignore
    else:
        import or_tool_solver as solver_module  # type: ignore
    return solver_module


def _ensure_empty_txt_dir(dir_path: str) -> None:
    """Avoid mixing a new guaranteed-optimal dataset with stale txt files."""
    existing = [name for name in os.listdir(dir_path) if name.lower().endswith(".txt")]
    if existing:
        raise FileExistsError(
            f"Output directory already contains {len(existing)} .txt files: {dir_path}. "
            "Use an empty output directory or remove the old dataset first."
        )


# ---------------------------
# Synthetic generator
# ---------------------------

def generate_batch_jssp_instances(
    num_instances: int,
    num_jobs: int,
    num_machines: int,
    output_dir: str,
    min_time: int = 1,
    max_time: int = 15,
    seed: Optional[int] = None,
    solve_opt: bool = True,
    time_limit_s: float = 30.0,
    workers: Optional[int] = None,
    solver_seed: Optional[int] = None,
) -> None:
    if not isinstance(num_instances, int) or num_instances < 0:
        raise ValueError("num_instances must be a non-negative integer.")
    if not isinstance(num_jobs, int) or num_jobs <= 0:
        raise ValueError("num_jobs must be a positive integer.")
    if not isinstance(num_machines, int) or num_machines <= 0:
        raise ValueError("num_machines must be a positive integer.")
    if min_time != 1 or max_time != 15:
        raise ValueError("Synthetic processing times MUST be in [1, 15].")

    if not solve_opt:
        raise ValueError(
            "Synthetic dataset generation now requires solve_opt=True so every saved instance "
            "has an OR-Tools-proven optimal makespan."
        )

    if seed is not None:
        random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    _ensure_empty_txt_dir(output_dir)
    solver_module = _load_or_tool_solver()

    kept = 0
    attempts = 0
    discarded = 0
    candidate_path = os.path.join(output_dir, "_candidate_instance.txt")

    try:
        while kept < num_instances:
            attempts += 1

            # Generate classic JSSP: each job covers all machines once
            jobs_data: JobsData = []
            for _ in range(num_jobs):
                machines = list(range(num_machines))
                random.shuffle(machines)
                durations = [random.randint(min_time, max_time) for _ in machines]
                job = list(zip(machines, durations))
                jobs_data.append(job)

            save_jssp_instance_to_txt(jobs_data, candidate_path)

            try:
                ms, proven_opt = solver_module.solve_jssp_with_ortools(
                    candidate_path,
                    time_limit_s=time_limit_s,
                    workers=workers,
                    random_seed=solver_seed,
                    require_optimal=True,
                )
            except Exception as exc:
                discarded += 1
                if discarded == 1 or discarded % 100 == 0:
                    print(f"[discard {discarded}] attempt={attempts} reason={exc}")
                continue

            if not proven_opt or ms <= 0:
                discarded += 1
                if discarded == 1 or discarded % 100 == 0:
                    print(
                        f"[discard {discarded}] attempt={attempts} reason="
                        "solver did not return a proven positive optimum"
                    )
                continue

            kept += 1
            final_path = os.path.join(output_dir, f"jssp_{kept}.txt")
            save_jssp_instance_to_txt(jobs_data, final_path, opt_ms=int(ms), is_opt=True)

            if kept % 500 == 0 or kept == num_instances:
                print(
                    f"[{kept}/{num_instances}] saved: {final_path} "
                    f"(attempts={attempts}, discarded={discarded})"
                )
    finally:
        if os.path.exists(candidate_path):
            try:
                os.remove(candidate_path)
            except OSError:
                pass


def generate_dataset(
    train_size: int = 100,
    test_size: int = 20,
    num_jobs: int = 3,
    num_machines: int = 3,
    output_dir: str = "jssp_dataset",
    seed: int = 42,
    min_time: int = 1,
    max_time: int = 15,
    solve_opt: bool = True,
    time_limit_s: float = 30.0,
    workers: Optional[int] = None,
    solver_seed: Optional[int] = None,
) -> None:
    """
    Build dataset directories:
        output_dir/train/...
        output_dir/test/... (if test_size > 0)
    """
    if train_size <= 0 or test_size < 0:
        raise ValueError("train_size must be > 0 and test_size must be >= 0")
    if min_time != 1 or max_time != 15:
        raise ValueError("Synthetic processing times MUST be in [1, 15].")
    if not solve_opt:
        raise ValueError(
            "generate_dataset now requires solve_opt=True so only OR-Tools-proven optimal "
            "instances are written to disk."
        )

    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    print("Generating training set...")
    generate_batch_jssp_instances(
        train_size, num_jobs, num_machines, train_dir,
        min_time=min_time, max_time=max_time, seed=seed,
        solve_opt=solve_opt, time_limit_s=time_limit_s,
        workers=workers, solver_seed=solver_seed,
    )

    if test_size > 0:
        test_dir = os.path.join(output_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        print("Generating testing set...")
        generate_batch_jssp_instances(
            test_size, num_jobs, num_machines, test_dir,
            min_time=min_time, max_time=max_time, seed=seed + 10000,
            solve_opt=solve_opt, time_limit_s=time_limit_s,
            workers=workers, solver_seed=solver_seed,
        )

    print("Dataset generation complete.")


# ---------------------------
# Optional: benchmark parser
# ---------------------------
Jobs = List[List[Tuple[int, Union[int, float]]]]

def _read_clean_lines(file_path: str) -> List[str]:
    """Read the file, remove blank lines and comments/delimiters, 
    and keep only lines containing numbers."""
    lines = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("#") or set(s) <= {"+", "-","=","*"," "}:
                # Comment or separator line
                continue
            # Strictly require that numbers appear on this line; otherwise skip
            if re.search(r"\d", s):
                lines.append(s)
    return lines

def _read_header_ints(clean_lines: List[str]) -> Tuple[int, int, List[str]]:
    """Read (n_jobs, n_machines) from the cleaned rows, returning the remaining rows."""
    if not clean_lines:
        raise ValueError("Empty file after stripping comments.")
    header = clean_lines[0]
    nums = list(map(int, re.findall(r"-?\d+", header)))
    if len(nums) < 2:
        # Some files will put n m on a separate line; try to fill it in from the subsequent lines
        merged = " ".join(clean_lines[:2])
        nums = list(map(int, re.findall(r"-?\d+", merged)))
        if len(nums) < 2:
            raise ValueError(f"Cannot find 'n m' header in: {header!r}")
        # If the header line is incomplete, simply merge the two lines together.
        # But for simplicity, still set the remaining lines to clean_lines[2:]
        return nums[0], nums[1], clean_lines[2:]
    return nums[0], nums[1], clean_lines[1:]

def _parse_pairlist(lines_after_header: List[str], n: int, m: int) -> List[List[Tuple[int, Union[int, float]]]]:
    jobs: List[List[Tuple[int, Union[int, float]]]] = []
    buf: List[int] = []

    def flush_row(row_tokens: List[int]) -> None:
        ops: List[Tuple[int, Union[int, float]]] = []
        macs = row_tokens[::2]
        
        if min(macs) < 0 or max(macs) > m:
            raise ValueError(
                f"Invalid machine ID detected (min={min(macs)}, max={max(macs)} for {m} machines). "
                "The file is likely not in the pair-list format."
            )

        is_one_based = max(macs) == m
        for k in range(0, 2*m, 2):
            mac = row_tokens[k]
            pt_val = float(row_tokens[k+1])   
            if is_one_based:
                mac -= 1
            ops.append((mac, pt_val))
        jobs.append(ops)

    for line in lines_after_header:
        ints = list(map(int, re.findall(r"-?\d+", line)))
        if not ints:
            continue
        buf.extend(ints)
        # Some benchmarks will wrap/break lines, here we will write a line when it reaches 2m
        while len(buf) >= 2*m and len(jobs) < n:
            row = buf[:2*m]
            buf = buf[2*m:]
            flush_row(row)

        if len(jobs) == n:
            break

    if len(jobs) != n:
        raise ValueError(f"Pair-list format: expected {n} rows of 2*{m} ints, got {len(jobs)}.")
    return jobs

# ---------- DMU style: two-matrices ----------
def _parse_two_matrices(lines_after_header: List[str], n: int, m: int) -> Jobs:
    """DMU style: first give n×m processing times, then give n×m machines (1..m)."""
    tokens = []
    for line in lines_after_header:
        tokens.extend(map(int, re.findall(r"-?\d+", line)))
    need = n * m
    if len(tokens) < 2 * need:
        raise ValueError(f"Two-matrix format: need {2*need} ints, got {len(tokens)}.")
    pt_block  = tokens[:need]
    mac_block = tokens[need: 2*need]

    is_one_based = (min(mac_block) >= 1) and (max(mac_block) == m)

    # reshape
    jobs: Jobs = []
    for i in range(n):
        ops: List[Tuple[int, Union[int, float]]] = []
        for j in range(m):
            pt  = float(pt_block[i*m + j])     
            mac = mac_block[i*m + j]
            if is_one_based:
                mac -= 1
            if not (0 <= mac < m):
                raise ValueError(f"Machine id {mac} out of [0,{m-1}] after normalization.")
            ops.append((mac, pt))
        jobs.append(ops)
    return jobs

# ---------- Automatic identification ----------
def parse_benchmark_auto(file_path: str) -> Jobs:
    """
    Automatic parsing: Try pair-list first, then two-matrices if that fails.
    This covers: ABZ/FT/ORB/SWV/TA/YN (pair-list), DMU (two-matrices).
    """
    lines = _read_clean_lines(file_path)
    n, m, rest = _read_header_ints(lines)

    try:
        return _parse_pairlist(rest, n, m)
    except Exception:
        pass

    return _parse_two_matrices(rest, n, m)

# ---------- Specify a dataset name ----------
def parse_benchmark_by_name(file_path: str, dataset: str) -> Jobs:
    """
    Explicitly resolve by dataset name. The dataset prefix is ​​capitalized: 
    'ABZ', 'DMU', 'FT', 'ORB', 'SWV', 'TA', 'YN'
    """
    lines = _read_clean_lines(file_path)
    n, m, rest = _read_header_ints(lines)
    ds = dataset.strip().upper()
    if ds == "DMU":
        return _parse_two_matrices(rest, n, m)
    elif ds in {"ABZ", "FT", "ORB", "SWV", "TA", "YN"}:
        return _parse_pairlist(rest, n, m)
    else:
        # Automatic fallback when not recognized
        return parse_benchmark_auto(file_path)
