import os
import sys
import yaml
import json
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
from trainer_utils import train_and_evaluate, make_small_subset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_train_eval_dataset(dataset, config):
    fast_cfg = config.get("fast_mode", {})
    fast_enabled = bool(fast_cfg.get("enabled", True))

    if fast_enabled:
        train_size = int(fast_cfg.get("train_size", 3000))
        eval_size = int(fast_cfg.get("eval_size", 500))

        train_dataset, eval_dataset = make_small_subset(
            dataset=dataset,
            train_size=train_size,
            eval_size=eval_size,
            seed=int(config.get("seed", 42)),
        )

        return train_dataset, eval_dataset

    train_dataset = dataset["train"]

    if "validation" in dataset:
        eval_dataset = dataset["validation"]
    elif "test" in dataset:
        eval_dataset = dataset["test"]
    else:
        raise ValueError("No validation or test split found.")

    return train_dataset, eval_dataset


def main():
    config_path = os.path.join(PROJECT_DIR, "configs", "exp1.yaml")
    config = load_config(config_path)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    fast_cfg = config.get("fast_mode", {})
    fast_enabled = bool(fast_cfg.get("enabled", True))

    print("=" * 70)
    print("Loaded experiment config successfully")
    print(f"Experiment name: {config['experiment_name']}")
    print(f"Datasets: {config['datasets']}")
    print(f"Methods : {config['methods']}")
    print(f"Model   : {config['model_name']}")
    print(f"Fast mode: {fast_enabled}")
    print("=" * 70)

    all_results = []

    for dataset_name in config["datasets"]:
        print("\n" + "=" * 70)
        print(f"Running dataset: {dataset_name}")
        print("=" * 70)

        dataset, tokenizer, num_labels = load_and_prepare_dataset(
            dataset_name=dataset_name,
            model_name=config["model_name"],
            max_length=int(config["max_length"]),
        )

        train_dataset, eval_dataset = get_train_eval_dataset(dataset, config)

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Eval dataset size    : {len(eval_dataset)}")

        for method_name in config["methods"]:
            print("\n" + "-" * 70)
            print(f"Experiment 1 | Dataset={dataset_name} | Method={method_name}")
            print("-" * 70)

            try:
                model = build_model_by_method(
                    method_name=method_name,
                    model_name=config["model_name"],
                    num_labels=num_labels,
                    config=config,
                )

                param_info = count_parameters(model)

                run_output_dir = os.path.join(
                    PROJECT_DIR,
                    config["output_dir"],
                    f"{dataset_name}_{method_name}",
                )

                _, results = train_and_evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    config=config,
                    run_output_dir=run_output_dir,
                    method_name=method_name,
                )

                row = {
                    "experiment": "exp1",
                    "dataset": dataset_name,
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
                    "experiment": "exp1",
                    "dataset": dataset_name,
                    "method": method_name,
                    "status": "failed",
                    "error": str(e),
                }

                all_results.append(row)

                print("\nRun failed:")
                print(str(e))

    results_path = os.path.join(
        PROJECT_DIR,
        config["output_dir"],
        "exp1_results.json",
    )

    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENT 1 RESULTS")
    print("=" * 70)
    print(json.dumps(all_results, indent=2))
    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()