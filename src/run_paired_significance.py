import os
import json
import csv
import math
import itertools
from collections import defaultdict

import numpy as np

try:
    from scipy import stats
except Exception as exc:
    raise SystemExit(
        "SciPy is required. Install once with: pip install scipy\n"
        f"Original import error: {exc}"
    )

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

OUTPUT_ROOT = os.path.join(PROJECT_DIR, "outputs")
OUT_DIR = os.path.join(OUTPUT_ROOT, "statistical_significance")
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_FILES = [
    os.path.join(
        OUTPUT_ROOT,
        "full_ft_fresh_process",
        "full_ft_fresh_process_results.json",
    ),
    os.path.join(
        OUTPUT_ROOT,
        "exp2_peft_official",
        "exp2_peft_official_results.json",
    ),
    os.path.join(
        OUTPUT_ROOT,
        "ia3_official_5seed",
        "ia3_official_5seed_results.json",
    ),
]

DATASETS = ["sst2", "imdb"]
METHODS = ["full_ft", "hybrid", "lora", "adapter", "bitfit", "ia3"]
TARGET_METHOD = "hybrid"
METRICS = ["eval_accuracy", "eval_f1_macro"]

RESULTS_JSON = os.path.join(
    OUT_DIR, "paired_significance_results.json"
)
RESULTS_CSV = os.path.join(
    OUT_DIR, "paired_significance_results.csv"
)


def normalize_method(value):
    m = str(value).strip().lower()

    aliases = {
        "full": "full_ft",
        "full_ft": "full_ft",
        "full-ft": "full_ft",
        "full fine-tuning": "full_ft",
        "hybrid": "hybrid",
        "lora": "lora",
        "adapter": "adapter",
        "bitfit": "bitfit",
        "ia3": "ia3",
        "ia³": "ia3",
    }

    return aliases.get(m, m)


def load_rows():
    rows = []

    for path in INPUT_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing required results file:\n{path}"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                data = data["results"]
            else:
                data = [data]

        for row in data:
            if row.get("status", "success") != "success":
                continue

            dataset = str(row.get("dataset", "")).strip().lower()
            method = normalize_method(row.get("method", ""))

            if dataset not in DATASETS:
                continue
            if method not in METHODS:
                continue
            if "seed" not in row:
                continue

            clean = dict(row)
            clean["dataset"] = dataset
            clean["method"] = method
            clean["seed"] = int(row["seed"])
            rows.append(clean)

    return rows


def build_index(rows):
    index = {}
    for row in rows:
        key = (row["dataset"], row["method"], row["seed"])
        index[key] = row
    return index


def paired_values(index, dataset, method_a, method_b, metric):
    seeds_a = {
        seed
        for (d, m, seed) in index.keys()
        if d == dataset and m == method_a
    }
    seeds_b = {
        seed
        for (d, m, seed) in index.keys()
        if d == dataset and m == method_b
    }

    common = sorted(seeds_a & seeds_b)

    xa, xb, used = [], [], []

    for seed in common:
        ra = index[(dataset, method_a, seed)]
        rb = index[(dataset, method_b, seed)]

        if metric not in ra or metric not in rb:
            continue

        va = float(ra[metric])
        vb = float(rb[metric])

        if not (np.isfinite(va) and np.isfinite(vb)):
            continue

        xa.append(va)
        xb.append(vb)
        used.append(seed)

    return np.array(xa), np.array(xb), used


def exact_sign_flip_pvalue(diffs):
    """
    Exact two-sided paired randomization/sign-flip test.
    With n=5, only 2^5=32 sign assignments are needed.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)

    if n == 0:
        return float("nan")

    observed = abs(float(np.mean(diffs)))
    values = []

    for signs in itertools.product([-1.0, 1.0], repeat=n):
        signed = diffs * np.array(signs)
        values.append(abs(float(np.mean(signed))))

    values = np.asarray(values)
    return float(np.mean(values >= observed - 1e-15))


def paired_effect_size_dz(diffs):
    diffs = np.asarray(diffs, dtype=float)

    if len(diffs) < 2:
        return float("nan")

    sd = float(np.std(diffs, ddof=1))

    if sd == 0:
        if float(np.mean(diffs)) == 0:
            return 0.0
        return math.copysign(float("inf"), float(np.mean(diffs)))

    return float(np.mean(diffs) / sd)


def mean_diff_ci95(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)

    if n < 2:
        return float("nan"), float("nan")

    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    se = sd / math.sqrt(n)
    critical = float(stats.t.ppf(0.975, df=n - 1))

    return (
        mean - critical * se,
        mean + critical * se,
    )


def wilcoxon_pvalue(xa, xb):
    diffs = xa - xb

    if np.allclose(diffs, 0):
        return 1.0

    try:
        res = stats.wilcoxon(
            xa,
            xb,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto",
        )
        return float(res.pvalue)
    except Exception:
        return float("nan")


def holm_adjust(pvalues):
    """
    Holm-Bonferroni adjusted p-values.
    """
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)

    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return adjusted.tolist()

    ordered = valid[np.argsort(p[valid])]
    m = len(ordered)
    running = 0.0

    for rank, idx in enumerate(ordered):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        adjusted[idx] = min(running, 1.0)

    return adjusted.tolist()


def main():
    rows = load_rows()
    index = build_index(rows)

    print("=" * 88)
    print("PAIRED STATISTICAL SIGNIFICANCE ANALYSIS")
    print("Same training seeds are paired within each dataset.")
    print("Primary test: exact two-sided sign-flip permutation test.")
    print("Secondary: paired t-test and Wilcoxon; effect size = Cohen's dz.")
    print("=" * 88)

    results = []

    # Focused comparisons: Hybrid against every principal comparator.
    comparators = ["full_ft", "lora", "adapter", "bitfit", "ia3"]

    for dataset in DATASETS:
        for metric in METRICS:
            for comparator in comparators:
                xa, xb, seeds = paired_values(
                    index,
                    dataset,
                    TARGET_METHOD,
                    comparator,
                    metric,
                )

                if len(seeds) < 2:
                    print(
                        f"SKIP {dataset} {metric}: "
                        f"{TARGET_METHOD} vs {comparator} "
                        f"(paired n={len(seeds)})"
                    )
                    continue

                diffs = xa - xb

                t_res = stats.ttest_rel(xa, xb)
                perm_p = exact_sign_flip_pvalue(diffs)
                wil_p = wilcoxon_pvalue(xa, xb)
                dz = paired_effect_size_dz(diffs)
                ci_low, ci_high = mean_diff_ci95(diffs)

                result = {
                    "dataset": dataset,
                    "metric": metric,
                    "method_a": TARGET_METHOD,
                    "method_b": comparator,
                    "paired_n": len(seeds),
                    "seeds": ",".join(str(s) for s in seeds),
                    "method_a_mean": float(np.mean(xa)),
                    "method_b_mean": float(np.mean(xb)),
                    "mean_difference_a_minus_b": float(np.mean(diffs)),
                    "difference_std": float(np.std(diffs, ddof=1)),
                    "difference_ci95_low": float(ci_low),
                    "difference_ci95_high": float(ci_high),
                    "cohens_dz": float(dz),
                    "paired_t_statistic": float(t_res.statistic),
                    "paired_t_p": float(t_res.pvalue),
                    "wilcoxon_p": float(wil_p),
                    "exact_sign_flip_p": float(perm_p),
                }

                results.append(result)

    # Holm correction separately for each metric across the 10 dataset/comparator tests.
    for metric in METRICS:
        idxs = [
            i for i, r in enumerate(results)
            if r["metric"] == metric
        ]

        raw = [
            results[i]["exact_sign_flip_p"]
            for i in idxs
        ]

        adjusted = holm_adjust(raw)

        for i, adj in zip(idxs, adjusted):
            results[i]["holm_adjusted_exact_p"] = float(adj)
            results[i]["significant_raw_0_05"] = (
                results[i]["exact_sign_flip_p"] < 0.05
            )
            results[i]["significant_holm_0_05"] = (
                adj < 0.05
            )

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if results:
        fields = list(results[0].keys())
        with open(
            RESULTS_CSV,
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)

    print("\n" + "=" * 88)
    print("ACCURACY: HYBRID VS PRINCIPAL COMPARATORS")
    print("=" * 88)

    for r in results:
        if r["metric"] != "eval_accuracy":
            continue

        print(
            f"{r['dataset']:5s} | "
            f"Hybrid vs {r['method_b']:8s} | "
            f"n={r['paired_n']} | "
            f"diff={100*r['mean_difference_a_minus_b']:+.2f} pp | "
            f"dz={r['cohens_dz']:+.3f} | "
            f"perm_p={r['exact_sign_flip_p']:.4f} | "
            f"Holm_p={r['holm_adjusted_exact_p']:.4f}"
        )

    print("\nInterpretation rule:")
    print(
        "- Use the exact sign-flip permutation p-value as the primary inferential result."
    )
    print(
        "- Report Cohen's dz and the paired mean difference because n=5 gives low statistical power."
    )
    print(
        "- Holm-adjusted p-values control family-wise error across Hybrid comparisons."
    )

    print(f"\nJSON: {RESULTS_JSON}")
    print(f"CSV : {RESULTS_CSV}")


if __name__ == "__main__":
    main()
