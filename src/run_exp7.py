import os
import sys
import yaml
import json
import csv
import copy
import random
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from trainer_utils import train_and_evaluate
from data import load_and_prepare_dataset
from metrics import count_parameters
from peft_methods import build_model_by_method


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_subset(dataset_split, size: int, seed: int):
    size = min(int(size), len(dataset_split))
    return dataset_split.shuffle(seed=seed).select(range(size))


def apply_debug_settings(config: dict):
    """
    هذا فقط للتجريب السريع.
    لا تستخدمه للنتائج الرسمية إلا إذا debug_mode: true.
    """
    config = copy.deepcopy(config)

    debug_cfg = config.get("debug", {})

    config["training"]["num_train_epochs"] = int(
        debug_cfg.get("num_train_epochs", 1)
    )
    config["training"]["logging_steps"] = 10
    config["training"]["save_strategy"] = "no"
    config["training"]["load_best_model_at_end"] = False

    config["low_resource_settings"] = debug_cfg.get(
        "low_resource_settings",
        [100, 250],
    )
    config["eval_size"] = int(debug_cfg.get("eval_size", 100))

    return config


def get_train_sizes_from_percentages(dataset_train_len: int, percentages):
    train_sizes = []

    for pct in percentages:
        size = int(dataset_train_len * float(pct))
        size = max(size, 1)
        size = min(size, dataset_train_len)
        train_sizes.append(size)

    return train_sizes


def main():
    config_path = os.path.join(PROJECT_DIR, "configs", "exp7.yaml")
    config = load_config(config_path)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    debug_mode = bool(config.get("debug_mode", False))

    if debug_mode:
        config = apply_debug_settings(config)

    print("=" * 70)
    print("Loaded experiment config successfully")
    print(f"Experiment name: {config['experiment_name']}")
    print(f"DEBUG_MODE: {debug_mode}")
    print("=" * 70)

    model_name = config["model_name"]
    max_length = int(config["max_length"])
    method_name = "hybrid"

    all_results = []

    for dataset_name in config["datasets"]:
        print("\n" + "=" * 70)
        print(f"Running EXP7 dataset: {dataset_name}")
        print("=" * 70)

        dataset, tokenizer, num_labels = load_and_prepare_dataset(
            dataset_name=dataset_name,
            model_name=model_name,
            max_length=max_length,
        )

        eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]

        if "low_resource_settings" in config:
            train_sizes = [int(x) for x in config["low_resource_settings"]]
            train_size_labels = [str(x) for x in train_sizes]
        else:
            percentages = config.get("low_resource_percentages", [0.10, 0.25, 0.50])
            train_sizes = get_train_sizes_from_percentages(
                dataset_train_len=len(dataset["train"]),
                percentages=percentages,
            )
            train_size_labels = [f"{float(p):.2f}" for p in percentages]

        eval_size = int(config.get("eval_size", 500))
        eval_subset = make_subset(eval_source, eval_size, seed)

        print(f"Eval subset: {len(eval_subset)}")
        print(f"Train sizes: {train_sizes}")

        for train_size, train_size_label in zip(train_sizes, train_size_labels):
            print("\n" + "-" * 70)
            print(
                f"LOW RESOURCE run | "
                f"dataset={dataset_name} | train_size={train_size}"
            )
            print("-" * 70)

            train_subset = make_subset(
                dataset["train"],
                train_size,
                seed,
            )

            try:
                model = build_model_by_method(
                    method_name=method_name,
                    model_name=model_name,
                    num_labels=num_labels,
                    config=config,
                )

                param_info = count_parameters(model)

                run_output_dir = os.path.join(
                    PROJECT_DIR,
                    config["output_dir"],
                    dataset_name,
                    f"{method_name}_{train_size}",
                )

                _, results = train_and_evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=train_subset,
                    eval_dataset=eval_subset,
                    config=config,
                    run_output_dir=run_output_dir,
                    method_name=method_name,
                )

                row = {
                    "experiment": "exp7",
                    "dataset": dataset_name,
                    "train_size": int(train_size),
                    "train_percentage_or_label": train_size_label,
                    "method": method_name,
                    "status": "success",
                    "total_params": param_info["total_params"],
                    "trainable_params": param_info["trainable_params"],
                    "trainable_ratio": param_info["trainable_ratio"],
                    "eval_accuracy": results.get("eval_accuracy"),
                    "eval_f1_macro": results.get("eval_f1_macro"),
                    "eval_f1_weighted": results.get("eval_f1_weighted"),
                    "eval_loss": results.get("eval_loss"),
                    "train_time_seconds": results.get("train_time_seconds"),
                    "peak_gpu_memory_mb": results.get("peak_gpu_memory_mb"),
                }

                all_results.append(row)
                print(json.dumps(row, indent=2))

            except Exception as e:
                row = {
                    "experiment": "exp7",
                    "dataset": dataset_name,
                    "train_size": int(train_size),
                    "train_percentage_or_label": train_size_label,
                    "method": method_name,
                    "status": "failed",
                    "error": str(e),
                }

                all_results.append(row)
                print("Run failed:")
                print(str(e))

    results_dir = os.path.join(PROJECT_DIR, config["output_dir"])
    os.makedirs(results_dir, exist_ok=True)

    final_json_path = os.path.join(results_dir, "exp7_results.json")
    final_csv_path = os.path.join(results_dir, "exp7_results.csv")

    save_json({"results": all_results}, final_json_path)
    save_csv(all_results, final_csv_path)

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENT 7 RESULTS")
    print("=" * 70)
    print(json.dumps(all_results, indent=2))
    print(f"\nSaved JSON to: {final_json_path}")
    print(f"Saved CSV to : {final_csv_path}")


if __name__ == "__main__":
    main()