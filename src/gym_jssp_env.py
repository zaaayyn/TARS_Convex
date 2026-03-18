from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

from utils.feature_utils import init_static_features, init_dynamic_features, build_input_features
from utils.render_utils import render_text, render_gantt, render_gantt_step

class GymJSSPEnv(gym.Env):
    """
    Gymnasium-compatible Job Shop Scheduling Problem environment.
    """
    metadata = {"render_modes": ["text", "gantt", "gantt_step"]}

    def __init__(
        self,
        jobs_data: List[List[Tuple[int, Union[int, float]]]],
        reward_alpha: float = 1.0,
        render_mode: Optional[str] = None,
        optimal_ms: Optional[float] = None,
        reward_mode = "shaped",
        feature_type: str = "full"
    ):
        super().__init__()
        # ----- Static configuration -----
        self.jobs_data = jobs_data
        self.num_jobs = len(jobs_data)
        self.num_machines = len(jobs_data[0]) if jobs_data else 0
        self.num_ops = self.num_jobs * self.num_machines
        self.job_lengths = np.array([len(j) for j in jobs_data], dtype=np.int32)
        self.max_ops = int(self.job_lengths.max()) if self.job_lengths.size > 0 else 0
        self.reward_alpha = reward_alpha
        self.render_mode = render_mode or "text"
        self.instance_path: Optional[Path] = None
        self.sum_proc_time = sum(d for job in jobs_data for (_, d) in job) or 1.0
        self.reward_mode = reward_mode
        self.feature_type = feature_type

        if optimal_ms is None:
            # Use the sum of all process durations as a rough upper bound to avoid division by 0
            optimal_ms = float(self.sum_proc_time)
        self.optimal_ms = float(optimal_ms)

        # Initialize static features and Encoder Mask
        self.static_rows, self.job_mask_enc, self.mach_mask_enc = init_static_features(jobs_data)

        # Action space
        self.action_space = spaces.Discrete(self.num_ops)

        # Observation space
        low  = np.array([0, 0, 0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float64)
        high = np.array([self.num_jobs-1, self.max_ops-1, self.num_machines-1, 1.0, 1.0, 1.0, 
                         np.inf, np.inf, 1.0], dtype=np.float64)

        if self.feature_type == "static_only":
            low, high = low[:4], high[:4]
        elif self.feature_type == "static_plus_start_time":
            low, high = np.r_[low[:4], low[7:8]], np.r_[high[:4], high[7:8]]

        low  = np.tile(low,  (self.num_ops, 1))
        high = np.tile(high, (self.num_ops, 1))

        self.observation_space = spaces.Dict({
            "features": spaces.Box(low, high, dtype=np.float64),
            "assigned_mask": spaces.MultiBinary(self.num_ops),
            "prev_op_idx": spaces.MultiBinary(self.num_ops),
            "job_mask": spaces.MultiBinary((self.num_ops, self.num_ops)),
            "mach_mask": spaces.MultiBinary((self.num_ops, self.num_ops)),
        })

        self.job_pointer = np.zeros(self.num_jobs, dtype=np.int32)
        self.scheduled_mask = np.zeros((self.num_jobs, self.num_machines), dtype=bool)
        self.machine_ready_time = np.zeros(self.num_machines, dtype=np.float32)
        self.op_start_times = np.full((self.num_jobs, self.num_machines), -1.0, dtype=np.float32)
        self.job_ready_time = np.zeros(self.num_jobs, dtype=np.float32)
        self.last_action = -1
        self.current_time = 0.0

        self.reset()

        assert self.job_mask_enc is not None, "job_mask_enc not initialized"
        assert self.mach_mask_enc is not None, "mach_mask_enc not initialized"

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
              ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        # Dynamic feature initialization
        dynamic_rows, _ = init_dynamic_features(self.jobs_data)
        dyn = np.array(dynamic_rows, dtype=np.float32)
        # Update scheduled_mask and job_pointer
        self.scheduled_mask = dyn[:, 0].reshape(self.num_jobs, self.num_machines).astype(bool)
        schedulable = dyn[:, 1].reshape(self.num_jobs, self.num_machines)
        self.job_pointer = np.argmax(schedulable, axis=1).astype(np.int32)
        self.prev_makespan = None

        # Reset other dynamic variables
        self.machine_ready_time.fill(0.0)
        self.op_start_times.fill(-1.0)
        self.job_ready_time.fill(0.0)
        self.last_action = -1
        self.current_time = 0.0

        obs = self._build_observation()
        info = {
            "job_pointer": self.job_pointer.copy(),
            "machine_ready_time": self.machine_ready_time.copy(),
        }
        return obs, info

    @property
    def action_mask(self) -> np.ndarray:
        """Returns a 1D boolean array of length = num_ops."""
        mask = np.zeros(self.num_ops, dtype=bool)
        for j in range(self.num_jobs):
            op = self.job_pointer[j]
            if op < self.num_machines:
                idx = j * self.num_machines + op
                # The next step of this job
                m_id, _ = self.jobs_data[j][op]
                # Whether the machine is idle is controlled by the calculation start time in step. 
                # Here I only do the mask of "whether it is the turn"
                mask[idx] = not self.scheduled_mask[j, op]
        return mask
    
    def get_action_mask(self) -> np.ndarray:
        return self.action_mask
    
    def action_masks(self) -> np.ndarray:
        return self.get_action_mask()

    def is_done(self) -> bool:
        """
        Determine whether all operations are completed.
        """
        return bool(self.scheduled_mask.all())

    def get_makespan(self) -> float:
        """
        Returns the current maximum completion time of the schedule.
        """
        return float(self.current_time)

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """
        Construct observations, including features, assigned masks, previous actions, 
        and encoder attention masks.
        """
        # Feature matrix, returned in NumPy
        features = build_input_features(
            jobs_data=self.jobs_data,
            env=self,
            return_numpy=True,
            feature_type=self.feature_type
        )
        # Assigned mask (1 means executed)
        assigned = self.scheduled_mask.flatten().astype(np.int8)

        # One-hot of the last action
        prev = np.zeros(self.num_ops, dtype=np.int8)
        if self.last_action >= 0:
            prev[self.last_action] = 1

        # Convert the encoder mask from Tensor to NumPy (and to int8 with 0/1 for better stability)
        job_mask = self.job_mask_enc
        mach_mask = self.mach_mask_enc

        if job_mask is not None and hasattr(job_mask, 'numpy'):
            job_mask = job_mask.numpy().astype(np.int8)
        elif job_mask is None:
            job_mask = np.zeros((self.num_ops, self.num_ops), dtype=np.int8)

        if mach_mask is not None and hasattr(mach_mask, 'numpy'):
            mach_mask = mach_mask.numpy().astype(np.int8)
        elif mach_mask is None:
            mach_mask = np.zeros((self.num_ops, self.num_ops), dtype=np.int8)

        return {
            "features": features,
            "assigned_mask": assigned,
            "prev_op_idx": prev,
            "job_mask": job_mask,
            "mach_mask": mach_mask,
        }  # type: ignore

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if action < 0 or action >= self.num_ops:
            raise IndexError("Action out of range")
        if not self.action_mask[action]:
            raise ValueError("Action not schedulable")
        self.last_action = action

        # decode action
        j = action // self.num_machines
        op = action % self.num_machines
        machine_id, dur = self.jobs_data[j][op]

        # Calculate start and end time
        prev_finish = self.job_ready_time[j]
        machine_free_at = self.machine_ready_time[machine_id]   
        start = max(self.machine_ready_time[machine_id], prev_finish)
        end = start + float(dur)

        # Update state variables
        self.op_start_times[j, op] = start
        self.scheduled_mask[j, op] = True
        self.job_pointer[j] += 1
        self.machine_ready_time[machine_id] = end
        self.job_ready_time[j] = end
        self.current_time = max(self.current_time, end)

        # Calculate reward
        done = self.is_done()
        curr_makespan = self.get_makespan()
        reward = 0.0

        if self.reward_mode == "shaped":
            # Penalize the idle machine introduced by this action
            idle_time = start - machine_free_at
            reward = - (idle_time / self.sum_proc_time)
            if done:
                gap = (curr_makespan - self.optimal_ms) / self.optimal_ms
                reward += -1.0 * gap

        elif self.reward_mode == "terminal":
            if done:
                gap = (curr_makespan - self.optimal_ms) / self.optimal_ms
                reward = - gap
            else:
                reward = 0.0

        elif self.reward_mode == "mixed":
            idle_time = start - machine_free_at
            shaped_reward = - (idle_time / self.sum_proc_time)
            reward = shaped_reward
            if done:
                gap = (curr_makespan - self.optimal_ms) / self.optimal_ms
                reward += -1.0 * gap
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        gap = (curr_makespan - self.optimal_ms) / self.optimal_ms
        truncated = False

        # Construct output
        obs = self._build_observation()
        info = {
            "makespan": curr_makespan,
            "gap": gap,
            "optimal_ms": self.optimal_ms,
            "last_action": action,
            "job_pointer": self.job_pointer.copy(),
            "machine_ready_time": self.machine_ready_time.copy(),
        }
        return obs, reward, done, truncated, info

    def render(self):
        """Perform different rendering depending on render_mode."""
        if self.render_mode == "text":
            render_text(self)
        elif self.render_mode == "gantt":
            render_gantt(self)
        elif self.render_mode == "gantt_step":
            render_gantt_step(self)
        else:
            raise ValueError(f"Unknown render_mode {self.render_mode}")