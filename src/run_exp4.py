import os
import sys
import json
import yaml
import copy
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


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(rows, path: str):
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


def apply_sensitivity_setting(config: dict, parameter_name: str, value):
    new_config = copy.deepcopy(config)

    if parameter_name == "prefix_virtual_tokens":
        new_config["hybrid"]["prefix_virtual_tokens"] = int(value)

    elif parameter_name == "prompt_virtual_tokens":
        new_config["hybrid"]["prompt_virtual_tokens"] = int(value)

    elif parameter_name == "learning_rate":
        new_config["training"]["learning_rate"] = float(value)

    else:
        raise ValueError(f"Unsupported sensitivity parameter: {parameter_name}")

    return new_config


def main():
    config_path = os.path.join(PROJECT_DIR, "configs", "exp4.yaml")
    config = load_config(config_path)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    print("=" * 70)
    print("Loaded experiment config successfully")
    print(f"Experiment name: {config['experiment_name']}")
    print(f"Datasets: {config['datasets']}")
    print("=" * 70)

    model_name = config["model_name"]
    max_length = int(config["max_length"])

    debug_cfg = config.get("debug", {})
    train_size = int(debug_cfg.get("train_size", 500))
    eval_size = int(debug_cfg.get("eval_size", 200))

    all_results = []
    sensitivity_settings = config["sensitivity"]

    for dataset_name in config["datasets"]:
        print("\n" + "=" * 70)
        print(f"Running EXP4 dataset: {dataset_name}")
        print("=" * 70)

        dataset, tokenizer, num_labels = load_and_prepare_dataset(
            dataset_name=dataset_name,
            model_name=model_name,
            max_length=max_length,
        )

        eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]

        train_subset = make_subset(dataset["train"], train_size, seed)
        eval_subset = make_subset(eval_source, eval_size, seed)

        print(f"Train subset: {len(train_subset)}")
        print(f"Eval subset : {len(eval_subset)}")

        for parameter_name, values in sensitivity_settings.items():
            print("\n" + "=" * 70)
            print(f"Testing sensitivity parameter: {parameter_name}")
            print("=" * 70)

            for value in values:
                print("\n" + "-" * 70)
                print(f"Dataset={dataset_name} | {parameter_name}={value}")
                print("-" * 70)

                run_config = apply_sensitivity_setting(
                    config=config,
                    parameter_name=parameter_name,
                    value=value,
                )

                method_name = "hybrid"

                try:
                    model = build_model_by_method(
                        method_name=method_name,
                        model_name=model_name,
                        num_labels=num_labels,
                        config=run_config,
                    )

                    param_info = count_parameters(model)

                    safe_value = str(value).replace(".", "_")
                    run_output_dir = os.path.join(
                        PROJECT_DIR,
                        run_config["output_dir"],
                        dataset_name,
                        f"{parameter_name}_{safe_value}",
                    )

                    _, results = train_and_evaluate(
                        model=model,
                        tokenizer=tokenizer,
                        train_dataset=train_subset,
                        eval_dataset=eval_subset,
                        config=run_config,
                        run_output_dir=run_output_dir,
                        method_name=method_name,
                    )

                    row = {
                        "experiment": "exp4",
                        "dataset": dataset_name,
                        "parameter": parameter_name,
                        "value": value,
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

                    print("\nFinished run:")
                    print(json.dumps(row, indent=2))

                except Exception as e:
                    row = {
                        "experiment": "exp4",
                        "dataset": dataset_name,
                        "parameter": parameter_name,
                        "value": value,
                        "method": method_name,
                        "status": "failed",
                        "error": str(e),
                    }

                    all_results.append(row)

                    print("\nRun failed:")
                    print(str(e))

    results_dir = os.path.join(PROJECT_DIR, config["output_dir"])
    os.makedirs(results_dir, exist_ok=True)

    final_results_path = os.path.join(results_dir, "exp4_results.json")
    final_csv_path = os.path.join(results_dir, "exp4_results.csv")

    save_json({"results": all_results}, final_results_path)
    save_csv(all_results, final_csv_path)

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENT 4 RESULTS")
    print("=" * 70)
    print(json.dumps(all_results, indent=2))
    print(f"\nSaved JSON to: {final_results_path}")
    print(f"Saved CSV to : {final_csv_path}")


if __name__ == "__main__":
    main()