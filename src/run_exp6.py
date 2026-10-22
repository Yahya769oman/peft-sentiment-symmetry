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
from metrics import count_parameters, compute_classification_metrics
from peft_methods import build_model_by_method
from trainer_utils import train_and_evaluate


def load_config(config_path: str):
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


def evaluate_on_target_dataset(
    trainer,
    target_eval_dataset,
):
    """
    يستخدم نفس trainer بعد التدريب على source domain،
    ثم يقيس الأداء على target domain.
    """
    eval_results = trainer.evaluate(eval_dataset=target_eval_dataset)

    clean_results = {}

    for key, value in eval_results.items():
        if isinstance(value, (int, float, np.floating)):
            clean_results[key] = float(value)
        else:
            clean_results[key] = value

    return clean_results


def main():
    config_path = os.path.join(PROJECT_DIR, "configs", "exp6.yaml")
    config = load_config(config_path)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    print("=" * 70)
    print("Loaded experiment config successfully")
    print(f"Experiment name: {config['experiment_name']}")
    print(f"Source datasets: {config['source_datasets']}")
    print(f"Target datasets: {config['target_datasets']}")
    print("=" * 70)

    model_name = config["model_name"]
    max_length = int(config["max_length"])
    method_name = config.get("method", "hybrid")

    debug_cfg = config.get("debug", {})
    train_size = int(debug_cfg.get("train_size", 500))
    source_eval_size = int(debug_cfg.get("source_eval_size", 200))
    target_eval_size = int(debug_cfg.get("target_eval_size", 300))

    all_results = []

    for source_dataset_name in config["source_datasets"]:
        print("\n" + "=" * 70)
        print(f"Loading source dataset: {source_dataset_name}")
        print("=" * 70)

        source_dataset, source_tokenizer, source_num_labels = load_and_prepare_dataset(
            dataset_name=source_dataset_name,
            model_name=model_name,
            max_length=max_length,
        )

        source_eval_source = (
            source_dataset["validation"]
            if "validation" in source_dataset
            else source_dataset["test"]
        )

        source_train_subset = make_subset(
            source_dataset["train"],
            train_size,
            seed,
        )

        source_eval_subset = make_subset(
            source_eval_source,
            source_eval_size,
            seed,
        )

        print(f"Source train subset: {len(source_train_subset)}")
        print(f"Source eval subset : {len(source_eval_subset)}")

        for target_dataset_name in config["target_datasets"]:
            print("\n" + "=" * 70)
            print(
                f"EXP6 Cross-Domain | "
                f"Train={source_dataset_name} -> Test={target_dataset_name}"
            )
            print("=" * 70)

            try:
                # نحمّل target بنفس tokenizer/model حتى تكون المدخلات متوافقة.
                target_dataset, _, target_num_labels = load_and_prepare_dataset(
                    dataset_name=target_dataset_name,
                    model_name=model_name,
                    max_length=max_length,
                )

                if source_num_labels != target_num_labels:
                    raise ValueError(
                        f"Label mismatch: source has {source_num_labels} labels, "
                        f"target has {target_num_labels} labels."
                    )

                target_eval_source = (
                    target_dataset["validation"]
                    if "validation" in target_dataset
                    else target_dataset["test"]
                )

                target_eval_subset = make_subset(
                    target_eval_source,
                    target_eval_size,
                    seed,
                )

                model = build_model_by_method(
                    method_name=method_name,
                    model_name=model_name,
                    num_labels=source_num_labels,
                    config=config,
                )

                param_info = count_parameters(model)

                run_output_dir = os.path.join(
                    PROJECT_DIR,
                    config["output_dir"],
                    f"{source_dataset_name}_to_{target_dataset_name}",
                    method_name,
                )

                trainer, source_results = train_and_evaluate(
                    model=model,
                    tokenizer=source_tokenizer,
                    train_dataset=source_train_subset,
                    eval_dataset=source_eval_subset,
                    config=config,
                    run_output_dir=run_output_dir,
                    method_name=method_name,
                )

                target_results = evaluate_on_target_dataset(
                    trainer=trainer,
                    target_eval_dataset=target_eval_subset,
                )

                row = {
                    "experiment": "exp6",
                    "source_dataset": source_dataset_name,
                    "target_dataset": target_dataset_name,
                    "method": method_name,
                    "status": "success",
                    "total_params": param_info["total_params"],
                    "trainable_params": param_info["trainable_params"],
                    "trainable_ratio": param_info["trainable_ratio"],

                    # داخل نفس الدومين source
                    "source_eval_accuracy": source_results.get("eval_accuracy"),
                    "source_eval_f1_macro": source_results.get("eval_f1_macro"),
                    "source_eval_f1_weighted": source_results.get("eval_f1_weighted"),
                    "source_eval_loss": source_results.get("eval_loss"),

                    # Cross-domain target
                    "target_eval_accuracy": target_results.get("eval_accuracy"),
                    "target_eval_f1_macro": target_results.get("eval_f1_macro"),
                    "target_eval_f1_weighted": target_results.get("eval_f1_weighted"),
                    "target_eval_loss": target_results.get("eval_loss"),

                    "domain_transfer_accuracy": target_results.get("eval_accuracy"),
                    "train_time_seconds": source_results.get("train_time_seconds"),
                    "peak_gpu_memory_mb": source_results.get("peak_gpu_memory_mb"),
                }

                all_results.append(row)

                print("\nFinished cross-domain run:")
                print(json.dumps(row, indent=2))

            except Exception as e:
                row = {
                    "experiment": "exp6",
                    "source_dataset": source_dataset_name,
                    "target_dataset": target_dataset_name,
                    "method": method_name,
                    "status": "failed",
                    "error": str(e),
                }

                all_results.append(row)

                print("\nRun failed:")
                print(str(e))

    results_dir = os.path.join(PROJECT_DIR, config["output_dir"])
    os.makedirs(results_dir, exist_ok=True)

    final_json_path = os.path.join(results_dir, "exp6_results.json")
    final_csv_path = os.path.join(results_dir, "exp6_results.csv")

    save_json({"results": all_results}, final_json_path)
    save_csv(all_results, final_csv_path)

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENT 6 RESULTS")
    print("=" * 70)
    print(json.dumps(all_results, indent=2))
    print(f"\nSaved JSON to: {final_json_path}")
    print(f"Saved CSV to : {final_csv_path}")


if __name__ == "__main__":
    main()