import matplotlib.pyplot as plt
import warnings
import matplotlib.patches as mpatches
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gym_jssp_env import GymJSSPEnv


def render_text(env: 'GymJSSPEnv') -> None:
    """Text rendering: Print job pointer and machine ready time."""
    print("=" * 40)
    for j, ptr in enumerate(env.job_pointer):
        printed = ptr - 1 if ptr > 0 else None
        print(f"Job {j}: up to op {printed}")
    print("Machine ready:", env.machine_ready_time)
    print("Last action:", env.last_action)
    print("=" * 40)

def _enable_constrained(fig):
    try:
        fig.set_layout_engine("constrained")  
        return True
    except Exception:
        try:
            fig.set_constrained_layout(True)   
            return True
        except Exception:
            return False

def _disable_constrained(fig):
    try:
        fig.set_layout_engine(None)            
    except Exception:
        try:
            fig.set_constrained_layout(False)  
        except Exception:
            pass


def render_gantt(env: 'GymJSSPEnv', save_path: Optional[str] = None):
    # === 0) Scale Statistics ===
    J = int(getattr(env, "num_jobs", len(env.jobs_data)))
    M = int(getattr(env, "num_machines", len(env.jobs_data[0]) if env.jobs_data else 0))

    # Estimate the length of the time axis based on the maximum scheduled end time
    max_end = 0
    for j in range(J):
        for o in range(M):
            if env.scheduled_mask[j, o]:
                mac, dur = env.jobs_data[j][o]
                st = env.op_start_times[j][o]
                if st + dur > max_end:
                    max_end = st + dur

    # === 1) Responsive canvas (scales up more conservatively) ===
    # The width increases with the "time range + number of jobs", with an upper limit of 64 
    # inches; the height increases with the number of machines, 
    # with an upper limit of 32 inches
    w = max(16.0, min(64.0, 0.035 * max_end + 0.30 * J))
    h = max(6.0,  min(32.0, 0.70 * M))

    # Automatically reduce the load on large instances 
    # (do not mark text in the box, do not draw legends)
    too_large = (J * M) > 700 or J > 60 or M > 25
    annotate = not too_large
    show_legend = not too_large

    # Use constrained_layout; if there is still a warning, there will be a fallback
    fig, ax = plt.subplots(figsize=(w, h))

    colors = plt.cm.tab20.colors  # type: ignore
    legend_handles = {}

    for j in range(J):
        for o in range(M):
            if env.scheduled_mask[j, o]:
                machine, duration = env.jobs_data[j][o]
                start = env.op_start_times[j][o]
                color = colors[j % len(colors)]
                ax.broken_barh(
                    [(start, duration)],
                    (machine - 0.4, 0.8),
                    facecolors=color,
                    edgecolor='black',
                    linewidth=0.6,
                )
                if annotate:
                    ax.text(
                        start + duration / 2.0,
                        machine,
                        f"J{j}O{o}",
                        va='center',
                        ha='center',
                        fontsize=7,
                        color='white',
                        clip_on=True,
                    )
                if show_legend and j not in legend_handles:
                    legend_handles[j] = mpatches.Patch(color=color, label=f"Job {j}")

    ax.set_yticks(range(M))
    ax.set_yticklabels([f"M{m}" for m in range(M)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title("JSSP Gantt Chart")
    ax.set_ylim(-0.5, M - 0.5)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5)
    ax.margins(x=0.01)

    if show_legend and legend_handles:
        ax.legend(
            handles=legend_handles.values(),
            bbox_to_anchor=(1.01, 1),
            loc='upper left',
            borderaxespad=0.,
            ncol=1,
            fontsize=8,
        )

    # === 2) Save (use constrained_layout first, and use it as a fallback if 
    # a user warning is issued) ===
    if save_path:
        _enable_constrained(fig)
        with warnings.catch_warnings(record=True) as wlog:
            warnings.simplefilter("always")
            fig.savefig(save_path, dpi=160, bbox_inches="tight")
        if any(issubclass(w.category, UserWarning) for w in wlog):
            _disable_constrained(fig)
            fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12)
            fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return fig


def render_gantt_step(env: 'GymJSSPEnv', delay: float = 0.5) -> None:
    """Gradual Gantt chart rendering: Wait for the user to press a key and 
    then display it gradually."""
    fig, ax = plt.subplots(figsize=(12, 0.6 * env.num_machines))
    colors = plt.cm.tab20.colors  # type: ignore
    legend_handles = {}
    for j in range(env.num_jobs):
        for o in range(env.num_machines):
            if env.scheduled_mask[j][o]:
                machine, duration = env.jobs_data[j][o]
                start = env.op_start_times[j][o]
                color = colors[j % len(colors)]
                ax.broken_barh([(start, duration)], (machine - 0.4, 0.8),
                               facecolors=color, edgecolor='black')
                ax.text(start + duration / 2, machine,
                        f"J{j}O{o}", va='center', ha='center', fontsize=8, color='white')
                if j not in legend_handles:
                    legend_handles[j] = mpatches.Patch(color=color, label=f"Job {j}")
    ax.set_yticks(range(env.num_machines))
    ax.set_yticklabels([f"M{m}" for m in range(env.num_machines)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title("JSSP Gantt Chart - Stepwise Rendering")
    ax.set_ylim(-0.5, env.num_machines - 0.5)           
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5)
    ax.legend(handles=legend_handles.values(), bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.show(block=False)
    input("Press Enter to schedule next operation... ")
    plt.close()
