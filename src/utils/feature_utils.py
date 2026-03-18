import torch
import numpy as np
from typing import List, Tuple, Optional


# ========== Initialization Features ==========

def init_static_features(jobs_data: List[List[Tuple[int, float]]]):
    """
    Construct static features and initialize the job/machine masks required by the encoder.

    Returns:
    static_rows: [n_ops, 4] -> [job_id, op_id, machine_id, proc_time_norm]
    job_mask_enc: [n_ops, n_ops], True indicates masked (jobs do not see each other).
    mach_mask_enc: [n_ops, n_ops], True indicates masked (machines do not see each other).
    """
    # Normalized processing time (min-max)
    durations = [dur for job in jobs_data for _, dur in job]
    if durations:
        dur_min, dur_max = min(durations), max(durations)
        denom = (dur_max - dur_min) if (dur_max > dur_min) else 1.0
    else:
        dur_min, denom = 0.0, 1.0

    static_rows: List[List[float]] = []
    for j, job in enumerate(jobs_data):
        for op_idx, (m_id, dur) in enumerate(job):
            pt_norm = (dur - dur_min) / denom
            static_rows.append([float(j), float(op_idx), float(m_id), float(pt_norm)])

    static_tensor = torch.tensor(static_rows, dtype=torch.float)
    same_job  = static_tensor[:, 0].unsqueeze(1) == static_tensor[:, 0].unsqueeze(0)
    same_mach = static_tensor[:, 2].unsqueeze(1) == static_tensor[:, 2].unsqueeze(0)
    job_mask_enc  = ~same_job.bool()
    mach_mask_enc = ~same_mach.bool()
    return static_rows, job_mask_enc, mach_mask_enc


def init_dynamic_features(jobs_data: List[List[Tuple[int, float]]]):
    """
    Initializes the dynamic features and decoder with the initial rows of actions.

    Returns:
    dynamic_rows: [n_ops, 5] -> [is_scheduled, is_schedulable, mach_ready, start_time, remaining]
    action_mask_dec: (Optional) Boolean mask of [num_jobs, max_ops] (optional for the first step)
    """
    num_jobs = len(jobs_data)
    max_ops = max((len(job) for job in jobs_data), default=0)

    dynamic_rows: List[List[float]] = []
    for job in jobs_data:
        job_total = sum(d for _, d in job) or 1.0
        cum_proc = 0.0
        for op_idx, (m_id, dur) in enumerate(job):
            remaining = (job_total - cum_proc) / job_total
            cum_proc += dur
            dynamic_rows.append([
                0.0,                       # is_scheduled
                1.0 if op_idx == 0 else 0.0,  # is_schedulable
                0.0,                       # mach_ready
                -1.0,                      # start_time
                float(remaining)           # remaining ratio
            ])

    # Optional initial mask for decoder (rows=job, columns=op_idx)
    scheduled = torch.zeros((num_jobs, max_ops), dtype=torch.bool)
    pointers = torch.zeros((num_jobs,), dtype=torch.long)  # 全是 0
    mask = torch.zeros_like(scheduled, dtype=torch.bool)
    if num_jobs > 0 and max_ops > 0:
        rows = torch.arange(num_jobs)
        mask[rows, pointers] = True
    action_mask_dec = (mask & ~scheduled)
    return dynamic_rows, action_mask_dec


# ========== Main interface: Constructing the observed feature matrix ==========

def build_input_features(
    jobs_data: List[List[Tuple[int, float]]],
    env: Optional[object] = None,
    return_numpy: bool = True,
    feature_type: str = "full"
):
    """
    Concatenates static and dynamic features, returning a feature matrix of the specified type.

    feature_type:
    "full" -> 9 dimensions: 4 static features + 5 dynamic features
    "static_only" -> 4 dimensions: static only
    "static_plus_start_time" -> 5 dimensions: 4 static features + start_time
    """
    static_rows, _, _ = init_static_features(jobs_data)

    if feature_type == "static_only":
        features = static_rows

    else:
        # Dynamic part sources:
        # - If env is provided, construct from the live state of env
        # - Otherwise use the initial state (all unscheduled processes, and optionally 
        # the first process of each job)
        num_jobs = len(jobs_data)
        num_machines = len(jobs_data[0]) if num_jobs else 0
        num_ops = num_jobs * num_machines

        if env is not None:
            # 1) is_scheduled
            scheduled_mask = getattr(env, "scheduled_mask", np.zeros((num_jobs, num_machines), dtype=bool))
            is_scheduled = scheduled_mask.flatten().astype(float)

            # 2) is_schedulable：current action_mask
            action_mask = getattr(env, "action_mask", np.zeros(num_ops, dtype=bool))
            is_schedulable = action_mask.astype(float)

            # 3) mach_ready：ready time of the machine to which each operation belongs
            mach_ready_time = getattr(env, "machine_ready_time", np.zeros(num_machines, dtype=float))
            mach_ready_vec = []
            for j in range(num_jobs):
                for op in range(num_machines):
                    m_id, _ = jobs_data[j][op]
                    mach_ready_vec.append(float(mach_ready_time[m_id]))

            # 4) start_time
            op_start_times = getattr(env, "op_start_times", np.full((num_jobs, num_machines), -1.0, dtype=float))
            start_times = op_start_times.flatten().astype(float).tolist()

            # 5) remaining：remaining working hours ratio (in job dimension)
            job_total = [sum(d for _, d in job) or 1.0 for job in jobs_data]
            job_elapsed = []
            for j in range(num_jobs):
                elapsed = 0.0
                for op in range(num_machines):
                    st = op_start_times[j, op]
                    if st >= 0:
                        _, d = jobs_data[j][op]
                        elapsed += d
                job_elapsed.append(elapsed)
            remaining_ratios = []
            for j in range(num_jobs):
                remain = max(job_total[j] - job_elapsed[j], 0.0) / job_total[j]
                for _ in range(num_machines):
                    remaining_ratios.append(float(remain))

            dynamic_rows = []
            for i in range(num_ops):
                dynamic_rows.append([
                    is_scheduled[i],
                    is_schedulable[i],
                    mach_ready_vec[i],
                    start_times[i],
                    remaining_ratios[i],
                ])
        else:
            dynamic_rows, _ = init_dynamic_features(jobs_data)

        # assembly features
        if feature_type == "full":
            features = [s + d for s, d in zip(static_rows, dynamic_rows)]
        elif feature_type == "static_plus_start_time":
            start_times = [d[3] for d in dynamic_rows]
            features = [s + [st] for s, st in zip(static_rows, start_times)]
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    feats_tensor = torch.tensor(features, dtype=torch.float)
    return feats_tensor.numpy() if return_numpy else feats_tensor
