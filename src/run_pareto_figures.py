import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
OUTPUT_ROOT = os.path.join(PROJECT_DIR, "outputs")
OUT_DIR = os.path.join(OUTPUT_ROOT, "pareto_figures")
os.makedirs(OUT_DIR, exist_ok=True)

FULL_FT_SUMMARY = os.path.join(
    OUTPUT_ROOT,
    "full_ft_fresh_process",
    "full_ft_fresh_process_summary.json",
)

PEFT_SUMMARY = os.path.join(
    OUTPUT_ROOT,
    "exp2_peft_official",
    "exp2_peft_official_summary.json",
)

IA3_SUMMARY = os.path.join(
    OUTPUT_ROOT,
    "ia3_official_5seed",
    "ia3_official_5seed_summary.json",
)

IA3_RESULTS = os.path.join(
    OUTPUT_ROOT,
    "ia3_official_5seed",
    "ia3_official_5seed_results.json",
)

OUT_CSV = os.path.join(
    OUT_DIR,
    "pareto_plot_data.csv",
)


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_method(method):
    value = str(method).strip().lower()

    aliases = {
        "full": "Full FT",
        "full_ft": "Full FT",
        "hybrid": "Hybrid",
        "lora": "LoRA",
        "adapter": "Adapter",
        "bitfit": "BitFit",
        "ia3": "IA3",
        "ia³": "IA3",
    }

    return aliases.get(value, str(method))


def get_value(row, keys):
    for key in keys:
        if key in row and row[key] is not None:
            return float(row[key])
    return None


def aggregate_ia3_resources(results):
    grouped = {}

    for dataset in ["sst2", "imdb"]:
        rows = [
            r for r in results
            if str(r.get("dataset", "")).lower() == dataset
            and str(r.get("status", "success")).lower() == "success"
        ]

        if not rows:
            continue

        gpu_vals = [
            float(r["peak_train_gpu_allocated_mb"])
            for r in rows
            if r.get("peak_train_gpu_allocated_mb") is not None
        ]

        time_vals = [
            float(r["train_time_seconds"])
            for r in rows
            if r.get("train_time_seconds") is not None
        ]

        grouped[dataset] = {
            "gpu_mean": float(np.mean(gpu_vals)) if gpu_vals else None,
            "time_mean": float(np.mean(time_vals)) if time_vals else None,
        }

    return grouped


def collect_data():
    full = load_json(FULL_FT_SUMMARY)
    peft = load_json(PEFT_SUMMARY)
    ia3_summary = load_json(IA3_SUMMARY)
    ia3_results = load_json(IA3_RESULTS)

    ia3_resources = aggregate_ia3_resources(ia3_results)

    rows = []

    for row in full:
        dataset = str(row["dataset"]).lower()

        rows.append({
            "dataset": dataset,
            "method": "Full FT",
            "accuracy_mean": get_value(
                row,
                ["eval_accuracy_mean"],
            ),
            "accuracy_std": get_value(
                row,
                ["eval_accuracy_std"],
            ),
            "peak_gpu_mb": get_value(
                row,
                ["peak_train_gpu_allocated_mb_mean"],
            ),
            "train_time_s": get_value(
                row,
                ["train_time_seconds_mean"],
            ),
        })

    for row in peft:
        dataset = str(row["dataset"]).lower()

        rows.append({
            "dataset": dataset,
            "method": normalize_method(row["method"]),
            "accuracy_mean": get_value(
                row,
                ["eval_accuracy_mean"],
            ),
            "accuracy_std": get_value(
                row,
                ["eval_accuracy_std"],
            ),
            "peak_gpu_mb": get_value(
                row,
                ["peak_train_gpu_allocated_mb_mean"],
            ),
            "train_time_s": get_value(
                row,
                ["train_time_seconds_mean"],
            ),
        })

    for row in ia3_summary:
        dataset = str(row["dataset"]).lower()
        resource = ia3_resources.get(dataset, {})

        rows.append({
            "dataset": dataset,
            "method": "IA3",
            "accuracy_mean": get_value(
                row,
                ["eval_accuracy_mean"],
            ),
            "accuracy_std": get_value(
                row,
                ["eval_accuracy_std"],
            ),
            "peak_gpu_mb": resource.get("gpu_mean"),
            "train_time_s": resource.get("time_mean"),
        })

    # Remove accidental duplicates, keeping the last row per dataset/method.
    dedup = {}
    for row in rows:
        dedup[(row["dataset"], row["method"])] = row

    return list(dedup.values())


def is_dominated(point, points):
    """
    Lower GPU memory is better.
    Higher accuracy is better.
    """
    for other in points:
        if other is point:
            continue

        at_least_as_good = (
            other["peak_gpu_mb"] <= point["peak_gpu_mb"]
            and other["accuracy_mean"] >= point["accuracy_mean"]
        )

        strictly_better = (
            other["peak_gpu_mb"] < point["peak_gpu_mb"]
            or other["accuracy_mean"] > point["accuracy_mean"]
        )

        if at_least_as_good and strictly_better:
            return True

    return False


def pareto_frontier(points):
    frontier = [
        p for p in points
        if not is_dominated(p, points)
    ]

    return sorted(
        frontier,
        key=lambda x: x["peak_gpu_mb"],
    )


def save_csv(rows):
    fields = [
        "dataset",
        "method",
        "accuracy_mean",
        "accuracy_std",
        "peak_gpu_mb",
        "train_time_s",
        "is_pareto",
    ]

    output_rows = []

    for dataset in ["sst2", "imdb"]:
        group = [
            r for r in rows
            if r["dataset"] == dataset
        ]

        frontier_methods = {
            r["method"]
            for r in pareto_frontier(group)
        }

        for row in group:
            item = dict(row)
            item["is_pareto"] = (
                row["method"] in frontier_methods
            )
            output_rows.append(item)

    with open(
        OUT_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(output_rows)


def make_plot(dataset, rows):
    points = [
        r for r in rows
        if r["dataset"] == dataset
        and r["accuracy_mean"] is not None
        and r["peak_gpu_mb"] is not None
    ]

    if not points:
        raise RuntimeError(
            f"No usable data for dataset={dataset}"
        )

    frontier = pareto_frontier(points)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = [p["peak_gpu_mb"] for p in points]
    y = [100.0 * p["accuracy_mean"] for p in points]
    yerr = [100.0 * p["accuracy_std"] for p in points]

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        capsize=4,
    )

    for p in points:
        ax.annotate(
            p["method"],
            (
                p["peak_gpu_mb"],
                100.0 * p["accuracy_mean"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    fx = [
        p["peak_gpu_mb"]
        for p in frontier
    ]

    fy = [
        100.0 * p["accuracy_mean"]
        for p in frontier
    ]

    if len(frontier) >= 2:
        ax.plot(
            fx,
            fy,
            linestyle="--",
            linewidth=1.5,
            label="Pareto frontier",
        )

    ax.set_xlabel(
        "Peak Training GPU Memory (MB)"
    )
    ax.set_ylabel(
        "Accuracy (%)"
    )

    pretty_dataset = (
        "SST-2"
        if dataset == "sst2"
        else "IMDb"
    )

    ax.set_title(
        f"{pretty_dataset}: Accuracy–Memory Trade-off"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    if len(frontier) >= 2:
        ax.legend()

    fig.tight_layout()

    png_path = os.path.join(
        OUT_DIR,
        f"pareto_accuracy_gpu_{dataset}.png",
    )

    pdf_path = os.path.join(
        OUT_DIR,
        f"pareto_accuracy_gpu_{dataset}.pdf",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    return png_path, pdf_path, frontier


def main():
    rows = collect_data()
    save_csv(rows)

    print("=" * 82)
    print("ACCURACY–EFFICIENCY PARETO FIGURES")
    print("=" * 82)

    for dataset in ["sst2", "imdb"]:
        png, pdf, frontier = make_plot(
            dataset,
            rows,
        )

        pretty_dataset = (
            "SST-2"
            if dataset == "sst2"
            else "IMDb"
        )

        print(
            f"\n{pretty_dataset} Pareto frontier:"
        )

        for p in frontier:
            print(
                f"  {p['method']:8s} | "
                f"acc={100*p['accuracy_mean']:.2f}% | "
                f"gpu={p['peak_gpu_mb']:.1f} MB"
            )

        print(f"PNG: {png}")
        print(f"PDF: {pdf}")

    print(f"\nPlot data CSV: {OUT_CSV}")
    print("\nDONE")


if __name__ == "__main__":
    main()
