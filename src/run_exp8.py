import os
import sys
import json
import yaml
import csv
import random
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import count_parameters
from peft_methods import build_model_by_method
from trainer_utils import train_and_evaluate


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_subset(dataset_split, size, seed):
    size = min(int(size), len(dataset_split))
    return dataset_split.shuffle(seed=seed).select(range(size))


def main():
    config_path = os.path.join(PROJECT_DIR, "configs", "exp8.yaml")
    config = load_config(config_path)

    seed = int(config["seed"])
    set_seed(seed)

    print("=" * 70)
    print("Experiment 8: Transformer Architecture Scalability")
    print(f"Datasets: {config['datasets']}")
    print(f"Models  : {config['models']}")
    print("=" * 70)

    max_length = int(config["max_length"])
    method_name = config.get("method", "hybrid")

    debug_cfg = config.get("debug", {})
    train_size = int(debug_cfg.get("train_size", 500))
    eval_size = int(debug_cfg.get("eval_size", 200))

    all_results = []

    for dataset_name in config["datasets"]:
        print("\n" + "=" * 70)
        print(f"Running EXP8 dataset: {dataset_name}")
        print("=" * 70)

        for model_name in config["models"]:
            print("\n" + "-" * 70)
            print(f"Dataset={dataset_name} | Model={model_name}")
            print("-" * 70)

            try:
                dataset, tokenizer, num_labels = load_and_prepare_dataset(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    max_length=max_length,
                )

                eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]

                train_subset = make_subset(
                    dataset["train"],
                    train_size,
                    seed,
                )

                eval_subset = make_subset(
                    eval_source,
                    eval_size,
                    seed,
                )

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
                    model_name.replace("/", "_"),
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
                    "experiment": "exp8",
                    "dataset": dataset_name,
                    "model": model_name,
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
                    "experiment": "exp8",
                    "dataset": dataset_name,
                    "model": model_name,
                    "method": method_name,
                    "status": "failed",
                    "error": str(e),
                }

                all_results.append(row)
                print("Run failed:")
                print(str(e))

    results_dir = os.path.join(PROJECT_DIR, config["output_dir"])
    os.makedirs(results_dir, exist_ok=True)

    final_json_path = os.path.join(results_dir, "exp8_results.json")
    final_csv_path = os.path.join(results_dir, "exp8_results.csv")

    save_json({"results": all_results}, final_json_path)
    save_csv(all_results, final_csv_path)

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENT 8 RESULTS")
    print("=" * 70)
    print(json.dumps(all_results, indent=2))
    print(f"\nSaved JSON to: {final_json_path}")
    print(f"Saved CSV to : {final_csv_path}")


if __name__ == "__main__":
    main()