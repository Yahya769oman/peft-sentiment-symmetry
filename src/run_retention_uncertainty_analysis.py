import os
import json
import csv
import math
import glob
from statistics import mean, stdev

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
OUTPUT_ROOT = os.path.join(PROJECT_DIR, "outputs")
OUT_DIR = os.path.join(OUTPUT_ROOT, "retention_uncertainty_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

PRINCIPAL_SEEDS = [42, 123, 2026, 3407, 7777]
SUBSET_SEEDS = [2026, 3407, 7777]

# Student-t two-sided 95% critical values.
T95 = {
    2: 12.706204736,  # df=1
    3: 4.302652730,   # df=2
    4: 3.182446305,   # df=3
    5: 2.776445105,   # df=4
}

SUMMARY_JSON = os.path.join(OUT_DIR, "retention_uncertainty_summary.json")
DETAIL_CSV = os.path.join(OUT_DIR, "retention_uncertainty_seed_details.csv")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_json_rows(results_path, worker_pattern):
    rows = []

    if os.path.exists(results_path):
        obj = load_json(results_path)
        if isinstance(obj, list):
            rows.extend(obj)
        elif isinstance(obj, dict):
            # Be permissive if a results file wraps rows in a key.
            for key in ("results", "rows", "runs"):
                if isinstance(obj.get(key), list):
                    rows.extend(obj[key])
                    break

    for path in glob.glob(worker_pattern):
        try:
            obj = load_json(path)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            pass

    # Deduplicate by a stable signature, keeping the last occurrence.
    dedup = {}
    for r in rows:
        dataset = str(r.get("dataset", "")).lower()
        method = str(r.get("method", "")).lower()
        seed = r.get("seed", r.get("train_seed"))
        subset_seed = r.get("train_subset_seed", r.get("subset_seed"))
        key = (dataset, method, seed, subset_seed)
        dedup[key] = r

    return list(dedup.values())


def accuracy_of(row):
    for key in ("eval_accuracy", "test_accuracy", "accuracy"):
        if row.get(key) is not None:
            return float(row[key])
    raise KeyError(f"No accuracy field in row: {row}")


def ci95(values):
    vals = [float(x) for x in values]
    n = len(vals)
    m = mean(vals)
    if n < 2:
        return {
            "n": n, "mean": m, "sd": None,
            "se": None, "ci95_low": None, "ci95_high": None
        }
    sd = stdev(vals)
    se = sd / math.sqrt(n)
    tcrit = T95.get(n)
    if tcrit is None:
        # Normal approximation only for unexpected n > 5.
        tcrit = 1.96
    half = tcrit * se
    return {
        "n": n,
        "mean": m,
        "sd": sd,
        "se": se,
        "ci95_low": m - half,
        "ci95_high": m + half,
    }


def find_principal_rows():
    full_dir = os.path.join(OUTPUT_ROOT, "full_ft_fresh_process")
    peft_dir = os.path.join(OUTPUT_ROOT, "exp2_peft_official")

    full = collect_json_rows(
        os.path.join(full_dir, "full_ft_fresh_process_results.json"),
        os.path.join(full_dir, "worker_*.json"),
    )
    peft = collect_json_rows(
        os.path.join(peft_dir, "exp2_peft_official_results.json"),
        os.path.join(peft_dir, "worker_*.json"),
    )

    def select(rows, dataset, method_aliases, seed):
        matches = []
        for r in rows:
            if str(r.get("status", "success")).lower() != "success":
                continue
            if str(r.get("dataset", "")).lower() != dataset:
                continue
            method = str(r.get("method", "")).lower()
            if method not in method_aliases:
                continue
            if int(r.get("seed")) != int(seed):
                continue
            matches.append(r)
        if not matches:
            return None
        return matches[-1]

    details = []
    for dataset in ("sst2", "imdb"):
        for seed in PRINCIPAL_SEEDS:
            f = select(full, dataset, {"full", "full_ft", "full-fine-tuning"}, seed)
            h = select(peft, dataset, {"hybrid"}, seed)
            if f is None or h is None:
                missing = []
                if f is None: missing.append("Full FT")
                if h is None: missing.append("Hybrid")
                raise FileNotFoundError(
                    f"Missing principal {'/'.join(missing)} result for {dataset}, seed={seed}. "
                    f"Expected results/worker JSON files under:\n{full_dir}\n{peft_dir}"
                )

            fa = accuracy_of(f)
            ha = accuracy_of(h)
            details.append({
                "analysis": "principal_training_seed",
                "dataset": dataset,
                "seed": int(seed),
                "full_accuracy": fa,
                "hybrid_accuracy": ha,
                "hybrid_minus_full_pp": 100.0 * (ha - fa),
                "retention_percent": 100.0 * ha / fa,
            })
    return details


def find_subset_rows():
    rob_dir = os.path.join(OUTPUT_ROOT, "data_subset_robustness")
    rows = collect_json_rows(
        os.path.join(rob_dir, "data_subset_robustness_results.json"),
        os.path.join(rob_dir, "worker_*.json"),
    )

    def subset_id(r):
        for key in ("train_subset_seed", "subset_seed", "data_seed"):
            if r.get(key) is not None:
                return int(r[key])
        return None

    def select(dataset, method_aliases, ss):
        matches = []
        for r in rows:
            if str(r.get("status", "success")).lower() != "success":
                continue
            if str(r.get("dataset", "")).lower() != dataset:
                continue
            if str(r.get("method", "")).lower() not in method_aliases:
                continue
            if subset_id(r) != int(ss):
                continue
            matches.append(r)
        return matches[-1] if matches else None

    details = []
    for dataset in ("sst2", "imdb"):
        for ss in SUBSET_SEEDS:
            f = select(dataset, {"full", "full_ft", "full-fine-tuning"}, ss)
            h = select(dataset, {"hybrid"}, ss)
            if f is None or h is None:
                raise FileNotFoundError(
                    f"Missing subset-robustness Full FT/Hybrid result for {dataset}, "
                    f"training-subset seed={ss}. Expected files under:\n{rob_dir}"
                )

            fa = accuracy_of(f)
            ha = accuracy_of(h)
            details.append({
                "analysis": "training_subset_seed",
                "dataset": dataset,
                "seed": int(ss),
                "full_accuracy": fa,
                "hybrid_accuracy": ha,
                "hybrid_minus_full_pp": 100.0 * (ha - fa),
                "retention_percent": 100.0 * ha / fa,
            })
    return details


def summarize_block(details, analysis_name):
    block = [r for r in details if r["analysis"] == analysis_name]
    out = {
        "analysis": analysis_name,
        "datasets": {},
    }

    for dataset in ("sst2", "imdb"):
        g = [r for r in block if r["dataset"] == dataset]
        out["datasets"][dataset] = {
            "seeds": [r["seed"] for r in g],
            "full_accuracy": ci95([100 * r["full_accuracy"] for r in g]),
            "hybrid_accuracy": ci95([100 * r["hybrid_accuracy"] for r in g]),
            "hybrid_minus_full_pp": ci95([r["hybrid_minus_full_pp"] for r in g]),
            "retention_percent": ci95([r["retention_percent"] for r in g]),
            "retention_values": [r["retention_percent"] for r in g],
        }

    # Cross-dataset retention contrast is paired by the same seed identifier.
    sst = {r["seed"]: r for r in block if r["dataset"] == "sst2"}
    imdb = {r["seed"]: r for r in block if r["dataset"] == "imdb"}
    common = sorted(set(sst) & set(imdb))
    contrast = [
        sst[s]["retention_percent"] - imdb[s]["retention_percent"]
        for s in common
    ]
    out["cross_dataset_retention_contrast_pp"] = {
        "definition": "SST-2 retention (%) minus IMDb retention (%)",
        "seeds": common,
        **ci95(contrast),
        "values": contrast,
    }

    return out


def save_csv(rows, path):
    fields = [
        "analysis", "dataset", "seed",
        "full_accuracy", "hybrid_accuracy",
        "hybrid_minus_full_pp", "retention_percent",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def fmt_ci(x):
    return f"{x['mean']:.2f}% (95% CI {x['ci95_low']:.2f} to {x['ci95_high']:.2f})"


def main():
    principal = find_principal_rows()
    subset = find_subset_rows()
    all_rows = principal + subset

    summary = {
        "method": (
            "Seed-level retention = 100 × Hybrid accuracy / Full-FT accuracy. "
            "95% CIs are Student-t intervals over seed-level values. "
            "The principal contrast uses five matched training seeds; "
            "training-subset sensitivity uses three matched subset seeds with training seed fixed at 42."
        ),
        "principal": summarize_block(all_rows, "principal_training_seed"),
        "training_subset_sensitivity": summarize_block(all_rows, "training_subset_seed"),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_csv(all_rows, DETAIL_CSV)

    print("=" * 88)
    print("RETENTION UNCERTAINTY ANALYSIS")
    print("=" * 88)

    p = summary["principal"]
    print("\nPRINCIPAL FIVE-SEED RETENTION")
    for ds in ("sst2", "imdb"):
        x = p["datasets"][ds]["retention_percent"]
        print(f"{ds.upper():5s}: {fmt_ci(x)}")
    c = p["cross_dataset_retention_contrast_pp"]
    print(
        f"Cross-dataset contrast (SST-2 - IMDb): "
        f"{c['mean']:.2f} pp (95% CI {c['ci95_low']:.2f} to {c['ci95_high']:.2f})"
    )

    s = summary["training_subset_sensitivity"]
    print("\nTRAINING-SUBSET SENSITIVITY")
    for ds in ("sst2", "imdb"):
        x = s["datasets"][ds]["retention_percent"]
        print(f"{ds.upper():5s}: {fmt_ci(x)}")
    c2 = s["cross_dataset_retention_contrast_pp"]
    print(
        f"Cross-dataset contrast across subset seeds: "
        f"{c2['mean']:.2f} pp (95% CI {c2['ci95_low']:.2f} to {c2['ci95_high']:.2f})"
    )

    print("\nSaved:")
    print(SUMMARY_JSON)
    print(DETAIL_CSV)


if __name__ == "__main__":
    main()
