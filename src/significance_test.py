"""
Read evaluation_details__{run_name}__{dataset}.csv files and run paired
significance tests (with_global vs without_global / static vs full) automatically by seed.

File name pattern expected:
  evaluation_details__Run_with_global_12__6x6.csv
  evaluation_details__Run_without_global_12__6x6.csv
Columns expected in CSV:
  ['file','instance','optimal_ms','makespan','gap','solve_time_ms','peak_mem_mb','dataset']
"""

import os, re, glob, argparse
import numpy as np
import pandas as pd

DEFAULT_TAG_A = "with_global"         # full feature
DEFAULT_TAG_B = "with_static_global"  # static feature
DEFAULT_METRIC = "gap"                # or "makespan"

# ---- utils: p-values (scipy if available; otherwise normal approx) ----
def pval_paired_t(d):
    """two-sided p-value for paired t-test on differences d"""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return 1.0, 0.0
    m = d.mean()
    sd = d.std(ddof=1)
    t = 0.0 if sd == 0 else m / (sd / np.sqrt(n))
    try:
        from scipy.stats import t as student_t
        p = 2 * student_t.sf(abs(t), df=n - 1)
    except Exception:
        # normal approx if scipy not available
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return float(p), float(t)

def pval_sign_test(d):
    """two-sided sign test p-value: count how often with<without (wins) vs > (loses)"""
    d = np.asarray(d, dtype=float)
    wins = int((d < 0).sum())   # with_global better (gap smaller)
    loses = int((d > 0).sum())
    n_eff = wins + loses
    if n_eff == 0:
        return 1.0, wins, loses, int((d == 0).sum())
    try:
        from scipy.stats import binomtest
        k = min(wins, loses)
        p = 2 * binomtest(k, n=n_eff, p=0.5, alternative="two-sided").pvalue
        p = min(p, 1.0)
    except Exception:
        # normal approx for two-sided binomial with p=0.5 (continuity correction)
        from math import erf, sqrt
        z = (abs(wins - n_eff / 2) - 0.5) / np.sqrt(n_eff / 4)
        p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    return float(p), wins, loses, int((d == 0).sum())

# ---- parsing helpers ----
SEED_RE = re.compile(r'_(\d+)$')  # run name ends with "_{seed}"

def seed_from_run_name(name: str):
    m = SEED_RE.search(name)
    return int(m.group(1)) if m else None

CSV_RE = re.compile(r"evaluation_details__(?P<run>.+?)__(?P<ds>\d+x\d+)\.csv")

def read_pairs(root: str, tag_a: str, tag_b: str):
    """
    Return dict: ds -> {'A': {seed: path}, 'B': {seed: path}}
    A/B are decided by whether tag_a/tag_b substrings appear in run name.
    """
    mapping = {}
    for fp in glob.glob(os.path.join(root, "evaluation_details__*__*.csv")):
        base = os.path.basename(fp)
        m = CSV_RE.match(base)
        if not m:
            continue
        run = m.group("run")
        ds  = m.group("ds")      # e.g., 6x6, 10x10, 15x15; files like __ALL.csv are ignored by regex
        seed = seed_from_run_name(run)
        if seed is None:
            continue
        side = "A" if (tag_a in run) else ("B" if (tag_b in run) else None)
        if side is None:
            continue
        mapping.setdefault(ds, {"A": {}, "B": {}})
        mapping[ds][side][seed] = fp
    return mapping

def diffs_from_pair(csv_A, csv_B, metric="gap"):
    """
    Return per-instance differences d = metric(A) - metric(B)
    where A is tag_a run and B is tag_b run, inner-joined by 'instance'.
    """
    A = pd.read_csv(csv_A)
    B = pd.read_csv(csv_B)
    if metric not in A.columns or metric not in B.columns:
        raise KeyError(f"Metric '{metric}' not found in CSV columns.")
    m = pd.merge(A[["instance", metric]], B[["instance", metric]],
                 on="instance", suffixes=("_A", "_B"))
    d = (m[f"{metric}_A"] - m[f"{metric}_B"]).to_numpy(float)
    return d

# ---- main ----
def main():
    ap = argparse.ArgumentParser(description="Paired significance tests from evaluation_details CSVs.")
    ap.add_argument("--root", type=str, default=".", help="Directory containing evaluation_details__*.csv")
    ap.add_argument("--tag_a", type=str, default=None)
    ap.add_argument("--tag_b", type=str, default=None)
    ap.add_argument("--metric", type=str, default=None)
    ap.add_argument("--save_csv", type=str, default=None, help="Optional path to save a summary CSV")
    args = ap.parse_args()

    tag_a  = args.tag_a  or DEFAULT_TAG_A
    tag_b  = args.tag_b  or DEFAULT_TAG_B
    metric = args.metric or DEFAULT_METRIC

    mapping = read_pairs(args.root, tag_a, tag_b)
    rows = []

    print(f"=== Paired significance tests: {tag_a}  vs  {tag_b}  (metric = {metric}) ===")
    for ds in sorted(mapping.keys()):
        A_map = mapping[ds]["A"]
        B_map = mapping[ds]["B"]
        seeds = sorted(set(A_map.keys()) & set(B_map.keys()))
        if not seeds:
            continue
        print(f"[{ds}] matched seeds: {seeds}")

        # per-seed
        for s in seeds:
            d = diffs_from_pair(A_map[s], B_map[s], metric)     # A - B
            n = int(d.size)
            delta = float(d.mean()) if n else 0.0
            p_t, t = pval_paired_t(d)
            p_sign, wins, loses, ties = pval_sign_test(d)
            # wins: d<0 means A<B (if the metric is as small as possible, A wins)
            print(f"  seed={s:>5}  n={n:>4}, Δ(mean {metric}, A-B)={delta:+.4f}, "
                f"t={t:+.3f}, p_t={p_t:.3g}, p_sign={p_sign:.3g}, wins(A<B)={wins}, loses(A>B)={loses}, ties={ties}")
            rows.append(dict(dataset=ds, seed=s, pooled=False, n=n,
                            delta_mean=delta, t=t, p_t=p_t, p_sign=p_sign,
                            wins=wins, loses=loses, ties=ties,
                            tag_a=tag_a, tag_b=tag_b, metric=metric))

        # pooled across seeds
        d_all = np.concatenate([diffs_from_pair(A_map[s], B_map[s], metric) for s in seeds])
        n = int(d_all.size)
        delta = float(d_all.mean()) if n else 0.0
        p_t, t = pval_paired_t(d_all)
        p_sign, wins, loses, ties = pval_sign_test(d_all)
        print(f"  pooled      n={n:>4}, Δ(mean {metric}, A-B)={delta:+.4f}, "
            f"t={t:+.3f}, p_t={p_t:.3g}, p_sign={p_sign:.3g}, wins(A<B)={wins}, loses(A>B)={loses}, ties={ties}")
        rows.append(dict(dataset=ds, seed=None, pooled=True, n=n,
                        delta_mean=delta, t=t, p_t=p_t, p_sign=p_sign,
                        wins=wins, loses=loses, ties=ties,
                        tag_a=tag_a, tag_b=tag_b, metric=metric))

    if args.save_csv:
        out = pd.DataFrame(rows)
        out.to_csv(args.save_csv, index=False)
        print(f"\nSaved summary to {args.save_csv}")

if __name__ == "__main__":
    main()
