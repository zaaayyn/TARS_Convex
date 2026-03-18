import sys
import subprocess
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

# =====================================================================
# --- Configure all your tests here ---
# =====================================================================
# You can define multiple test campaigns. Each campaign is a dictionary containing:
# - "name": The name of the test campaign, which will be used as the run name in W&B.
# - "model_path": The path to the .zip file containing the model you want to test.
# - "test_dirs": A list of the paths to the folders containing the datasets to be tested.
#
DEFAULT_WANDB_PROJECT = "JSSP_SOTA_Test"
FEATURE_TYPE_FOR_TESTS = "full"

EVALUATION_CAMPAIGNS: List[Dict[str, Any]] = [
    {
        "name": "Run_with_full_global_12",
        "model_path": "models\\withG_seed12_20250925-224718\\best_model\\best_model.zip",
        "test_dirs": [
            "jssp_dataset_test/6x6",
            "jssp_dataset_test/10x10",
            "jssp_dataset_test/15x15",
            "bench_data_txt/ABZ/10x10",
            "bench_data_txt/ABZ/20x15",
            "bench_data_txt/DMU/20x15",
            "bench_data_txt/DMU/20x20",
            "bench_data_txt/DMU/30x15",
            "bench_data_txt/DMU/30x20",
            "bench_data_txt/DMU/40x15",
            "bench_data_txt/DMU/40x20",
            "bench_data_txt/DMU/50x15",
            "bench_data_txt/DMU/50x20",
            "bench_data_txt/FT",
            "bench_data_txt/ORB",
            "bench_data_txt/SWV/20x10",
            "bench_data_txt/SWV/20x15",
            "bench_data_txt/SWV/50x10",
            "bench_data_txt/TA/15x15",
            "bench_data_txt/TA/20x15",
            "bench_data_txt/TA/20x20",
            "bench_data_txt/TA/30x15",
            "bench_data_txt/TA/30x20",
            "bench_data_txt/TA/50x15",
            "bench_data_txt/TA/50x20",
            "bench_data_txt/TA/100x20",
            "bench_data_txt/YN",
        ],
    },
    {
        "name": "Run_with_full_global_87",
        "model_path": "models\\withG_seed87_20250925-154110\\best_model\\best_model.zip",
        "test_dirs": [
            "jssp_dataset_test/6x6",
            "jssp_dataset_test/10x10",
            "jssp_dataset_test/15x15",
            "bench_data_txt/ABZ/10x10",
            "bench_data_txt/ABZ/20x15",
            "bench_data_txt/DMU/20x15",
            "bench_data_txt/DMU/20x20",
            "bench_data_txt/DMU/30x15",
            "bench_data_txt/DMU/30x20",
            "bench_data_txt/DMU/40x15",
            "bench_data_txt/DMU/40x20",
            "bench_data_txt/DMU/50x15",
            "bench_data_txt/DMU/50x20",
            "bench_data_txt/FT",
            "bench_data_txt/ORB",
            "bench_data_txt/SWV/20x10",
            "bench_data_txt/SWV/20x15",
            "bench_data_txt/SWV/50x10",
            "bench_data_txt/TA/15x15",
            "bench_data_txt/TA/20x15",
            "bench_data_txt/TA/20x20",
            "bench_data_txt/TA/30x15",
            "bench_data_txt/TA/30x20",
            "bench_data_txt/TA/50x15",
            "bench_data_txt/TA/50x20",
            "bench_data_txt/TA/100x20",
            "bench_data_txt/YN",
        ],
    },{
        "name": "Run_with_full_global_2025",
        "model_path": "models\\withG_seed2025_20250926-082826\\best_model\\best_model.zip",
        "test_dirs": [
            "jssp_dataset_test/6x6",
            "jssp_dataset_test/10x10",
            "jssp_dataset_test/15x15",
            "bench_data_txt/ABZ/10x10",
            "bench_data_txt/ABZ/20x15",
            "bench_data_txt/DMU/20x15",
            "bench_data_txt/DMU/20x20",
            "bench_data_txt/DMU/30x15",
            "bench_data_txt/DMU/30x20",
            "bench_data_txt/DMU/40x15",
            "bench_data_txt/DMU/40x20",
            "bench_data_txt/DMU/50x15",
            "bench_data_txt/DMU/50x20",
            "bench_data_txt/FT",
            "bench_data_txt/ORB",
            "bench_data_txt/SWV/20x10",
            "bench_data_txt/SWV/20x15",
            "bench_data_txt/SWV/50x10",
            "bench_data_txt/TA/15x15",
            "bench_data_txt/TA/20x15",
            "bench_data_txt/TA/20x20",
            "bench_data_txt/TA/30x15",
            "bench_data_txt/TA/30x20",
            "bench_data_txt/TA/50x15",
            "bench_data_txt/TA/50x20",
            "bench_data_txt/TA/100x20",
            "bench_data_txt/YN",
        ],
    },{
        "name": "Run_with_full_global_1287",
        "model_path": "models\\withG_seed1287_20250926-165151\\best_model\\best_model.zip",
        "test_dirs": [
            "jssp_dataset_test/6x6",
            "jssp_dataset_test/10x10",
            "jssp_dataset_test/15x15",
            "bench_data_txt/ABZ/10x10",
            "bench_data_txt/ABZ/20x15",
            "bench_data_txt/DMU/20x15",
            "bench_data_txt/DMU/20x20",
            "bench_data_txt/DMU/30x15",
            "bench_data_txt/DMU/30x20",
            "bench_data_txt/DMU/40x15",
            "bench_data_txt/DMU/40x20",
            "bench_data_txt/DMU/50x15",
            "bench_data_txt/DMU/50x20",
            "bench_data_txt/FT",
            "bench_data_txt/ORB",
            "bench_data_txt/SWV/20x10",
            "bench_data_txt/SWV/20x15",
            "bench_data_txt/SWV/50x10",
            "bench_data_txt/TA/15x15",
            "bench_data_txt/TA/20x15",
            "bench_data_txt/TA/20x20",
            "bench_data_txt/TA/30x15",
            "bench_data_txt/TA/30x20",
            "bench_data_txt/TA/50x15",
            "bench_data_txt/TA/50x20",
            "bench_data_txt/TA/100x20",
            "bench_data_txt/YN",
        ],
    },{
        "name": "Run_with_full_global_8712",
        "model_path": "models\\withG_seed8712_20250926-233145\\best_model\\best_model.zip",
        "test_dirs": [
            "jssp_dataset_test/6x6",
            "jssp_dataset_test/10x10",
            "jssp_dataset_test/15x15",
            "bench_data_txt/ABZ/10x10",
            "bench_data_txt/ABZ/20x15",
            "bench_data_txt/DMU/20x15",
            "bench_data_txt/DMU/20x20",
            "bench_data_txt/DMU/30x15",
            "bench_data_txt/DMU/30x20",
            "bench_data_txt/DMU/40x15",
            "bench_data_txt/DMU/40x20",
            "bench_data_txt/DMU/50x15",
            "bench_data_txt/DMU/50x20",
            "bench_data_txt/FT",
            "bench_data_txt/ORB",
            "bench_data_txt/SWV/20x10",
            "bench_data_txt/SWV/20x15",
            "bench_data_txt/SWV/50x10",
            "bench_data_txt/TA/15x15",
            "bench_data_txt/TA/20x15",
            "bench_data_txt/TA/20x20",
            "bench_data_txt/TA/30x15",
            "bench_data_txt/TA/30x20",
            "bench_data_txt/TA/50x15",
            "bench_data_txt/TA/50x20",
            "bench_data_txt/TA/100x20",
            "bench_data_txt/YN",
        ],
    },
    # ... Add more test activities here ...
]
# =====================================================================

def paired_tests(detail_csv_a: Path, detail_csv_b: Path):
    df_a = pd.read_csv(detail_csv_a)
    df_b = pd.read_csv(detail_csv_b)
    # align (dataset, instance)
    key = ["dataset", "instance"]
    merged = pd.merge(df_a[key + ["gap"]], df_b[key + ["gap"]],
                      on=key, suffixes=("_A", "_B"))
    if merged.empty:
        return None

    x = merged["gap_A"].to_numpy(float)
    y = merged["gap_B"].to_numpy(float)
    d = x - y  # WithG - NoG (negative value means WithG is better)

    # Paired t-test 
    n = d.size
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
    t = mean_d / (std_d / np.sqrt(n)) if std_d > 0 else np.nan
    # Approximate two-sided p-value (uses normal approximation instead of t distribution, 
    # reasonable when n ≥ 30)
    from math import erf
    p_t = float(2.0 * (1.0 - 0.5 * (1 + erf(abs(t)/np.sqrt(2))))) if np.isfinite(t) else np.nan

    # Sign test (no distribution assumption)
    wins = int(np.sum(d < 0))   #A(WithG) is better than B(NoG)
    loses = int(np.sum(d > 0))
    ties  = int(np.sum(d == 0))
    # Two-sided p (binomial tail sum under Bernoulli 0.5; n effective = wins+loses)
    n_eff = max(1, wins + loses)
    from math import comb
    tail = sum(comb(n_eff, k) for k in range(0, min(wins, loses)+1))
    p_sign = 2.0 * min(tail, 2**n_eff - tail) / (2**n_eff)

    return {
        "n_pairs": int(n),
        "mean_delta": mean_d,
        "t_stat": t, "p_pair_t": p_t,
        "wins(A< B)": wins, "loses(A> B)": loses, "ties": ties, "p_sign": float(p_sign),
    }

def main():
    """
    Iterate over the test activities defined above and launch `test.py` once for each activity.
    """
    script_dir = Path(__file__).parent.resolve()
    test_script_path = script_dir / "test.py"

    if not test_script_path.exists():
        print(f"❌ Error: 'test.py' not found at {test_script_path}")
        print("Please make sure 'run_evaluation.py' and 'test.py' are in the same directory.")
        return

    print("=" * 60)
    print(f"Found test script: {test_script_path}")
    print(f"Found {len(EVALUATION_CAMPAIGNS)} evaluation campaigns to run.")
    print("=" * 60)

    for i, campaign in enumerate(EVALUATION_CAMPAIGNS, 1):
        name = campaign.get("name", f"campaign_{i}")
        model_path = Path(campaign.get("model_path", ""))
        test_dirs = campaign.get("test_dirs", [])

        print(f"\n\n--- Starting Campaign {i}/{len(EVALUATION_CAMPAIGNS)}: {name} ---")

        # --- Preliminary inspection ---
        if not model_path or not test_dirs:
            print(f"  [SKIP] Campaign '{name}' is missing 'model_path' or 'test_dirs'.")
            continue
        
        if not model_path.exists():
            print(f"  [SKIP] Model not found for campaign '{name}': {model_path}")
            continue

        valid_test_dirs = []
        for d in test_dirs:
            dir_path = Path(d)
            if dir_path.exists() and dir_path.is_dir():
                valid_test_dirs.append(str(dir_path))
            else:
                print(f"  [WARN] Test directory not found, will be skipped: {d}")

        if not valid_test_dirs:
            print(f"  [SKIP] No valid test directories found for campaign '{name}'.")
            continue

        command = [
            sys.executable,
            str(test_script_path),
            "--model_path", str(model_path),
            "--test_dirs", *valid_test_dirs,
            "--project", campaign.get("project", DEFAULT_WANDB_PROJECT),
            "--name", name,
            "--note", campaign.get("note", ""),
            "--render",
        ]

        if FEATURE_TYPE_FOR_TESTS:
            command.extend(["--feature_type", FEATURE_TYPE_FOR_TESTS])

        print(f"\nRunning command for '{name}':")
        print(" ".join(f'"{c}"' if " " in c else c for c in command))
        print("-" * 60)

        # --- Execute Command ---
        try:
            # check=True means that if test.py reports an error, this script will also stop.
            subprocess.run(command, check=True)
            print(f"\n Campaign '{name}' finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\n Campaign '{name}' failed with exit code {e.returncode}.")
        except KeyboardInterrupt:
            print("\n Evaluation stopped by user.")
            return
        
        print("\n\n=== Significance tests (per dataset) ===")
        # Find the ALL details of the two runs in this directory
        csv_map = defaultdict(dict)
        for camp in EVALUATION_CAMPAIGNS:
            name = camp["name"]
            # Matches ALL details output by test.py
            for p in Path(".").glob(f"evaluation_details__{name}__ALL.csv"):
                df = pd.read_csv(p)
                for ds in sorted(df["dataset"].unique()):
                    sub = df[df["dataset"] == ds]
                    out = Path(f"evaluation_details__{name}__{ds}.csv")
                    sub.to_csv(out, index=False)
                    csv_map[ds][name] = out

        # Only compare when both variants exist
        if len(EVALUATION_CAMPAIGNS) >= 2:
            nameA = EVALUATION_CAMPAIGNS[0]["name"]
            nameB = EVALUATION_CAMPAIGNS[1]["name"]
            for ds, mp in sorted(csv_map.items()):
                if nameA in mp and nameB in mp:
                    res = paired_tests(mp[nameA], mp[nameB])
                    if res:
                        print(f"[{ds}] n={res['n_pairs']}, Δ(mean gap A-B)={res['mean_delta']:.4f}, "
                            f"t={res['t_stat']:.3f}, p_t={res['p_pair_t']:.3g}, p_sign={res['p_sign']:.3g}")

    print("\n\n All evaluation campaigns are complete.")


if __name__ == "__main__":
    main()