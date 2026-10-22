import os
import sys
import csv
import json
import time
import random
import platform
from pathlib import Path

import numpy as np
import torch
import yaml
import transformers
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import count_parameters
from models_symmetry import TrueHybridPrefixPromptBertClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility-oriented settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fixed_subset(split, n: int, data_seed: int):
    n = min(int(n), len(split))
    return split.shuffle(seed=int(data_seed)).select(range(n))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_class0": precision[0],
        "recall_class0": recall[0],
        "f1_class0": f1_per_class[0],
        "support_class0": support[0],
        "precision_class1": precision[1],
        "recall_class1": recall[1],
        "f1_class1": f1_per_class[1],
        "support_class1": support[1],
    }


def clean_number(v):
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return v


def build_model(method, model_name, num_labels, cfg):
    if method == "full":
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=int(num_labels),
        )

    if method == "hybrid":
        h = cfg["hybrid"]
        return TrueHybridPrefixPromptBertClassifier(
            model_name=model_name,
            num_labels=int(num_labels),
            prefix_length=int(h["prefix_length"]),
            prompt_length=int(h["prompt_length"]),
            dropout=float(h.get("dropout", 0.1)),
        )

    raise ValueError(f"Unknown method: {method}")


def run_key(dataset_name, method, seed):
    return f"{dataset_name}__{method}__seed{int(seed)}"


def load_existing_rows(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def save_rows(rows, json_path, csv_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if rows:
        # Union of all keys so failed/success rows can coexist.
        fieldnames = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def run_one(method, dataset_name, seed, cfg):
    set_seed(int(seed))

    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name,
        model_name=cfg["model_name"],
        max_length=int(cfg["max_length"]),
    )

    eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]
    data_seed = int(cfg.get("data_seed", 2026))
    train_ds = fixed_subset(dataset["train"], cfg["subset"]["train_size"], data_seed)
    eval_ds = fixed_subset(eval_source, cfg["subset"]["eval_size"], data_seed)

    # Set training seed again immediately before model initialization so each
    # method/seed combination is reproducible independent of dataset loading.
    set_seed(int(seed))
    model = build_model(method, cfg["model_name"], num_labels, cfg)
    params = count_parameters(model)

    train_cfg = cfg["training"]
    lr = float(cfg["method_overrides"][method]["learning_rate"])

    out_dir = os.path.join(
        PROJECT_DIR,
        cfg["output_dir"],
        f"{dataset_name}_{method}_seed{int(seed)}",
    )
    os.makedirs(out_dir, exist_ok=True)

    use_fp16 = bool(train_cfg.get("fp16", False)) and torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=int(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "linear")),
        optim=str(train_cfg.get("optimizer", "adamw_torch")),
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        fp16=use_fp16,
        seed=int(seed),
        data_seed=int(seed),
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    gpu_info = {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": None,
        "gpu_total_memory_mb": None,
        "peak_gpu_allocated_mb": 0.0,
        "peak_gpu_reserved_mb": 0.0,
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        props = torch.cuda.get_device_properties(0)
        gpu_info["gpu_name"] = props.name
        gpu_info["gpu_total_memory_mb"] = float(props.total_memory / (1024 ** 2))

    start = time.time()
    trainer.train()
    train_seconds = time.time() - start

    eval_results = trainer.evaluate()

    if torch.cuda.is_available():
        gpu_info["peak_gpu_allocated_mb"] = float(
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )
        gpu_info["peak_gpu_reserved_mb"] = float(
            torch.cuda.max_memory_reserved() / (1024 ** 2)
        )

    row = {
        "run_key": run_key(dataset_name, method, seed),
        "experiment": cfg["experiment_name"],
        "dataset": dataset_name,
        "method": method,
        "seed": int(seed),
        "data_seed": data_seed,
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "model_name": cfg["model_name"],
        "max_length": int(cfg["max_length"]),
        "epochs": int(train_cfg["num_train_epochs"]),
        "train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "eval_batch_size": int(train_cfg["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": lr,
        "optimizer": str(train_cfg.get("optimizer", "adamw_torch")),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "linear")),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.0)),
        "fp16": use_fp16,
        "prefix_length": int(cfg["hybrid"]["prefix_length"]) if method == "hybrid" else None,
        "prompt_length": int(cfg["hybrid"]["prompt_length"]) if method == "hybrid" else None,
        **params,
        "train_time_seconds": float(train_seconds),
        **gpu_info,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "status": "success",
    }

    for k, v in eval_results.items():
        row[k] = clean_number(v)

    return row


def make_summary(rows):
    success = [r for r in rows if r.get("status") == "success"]
    groups = {}

    metrics = [
        "eval_accuracy",
        "eval_f1_macro",
        "eval_f1_weighted",
        "eval_precision_class0",
        "eval_recall_class0",
        "eval_f1_class0",
        "eval_precision_class1",
        "eval_recall_class1",
        "eval_f1_class1",
        "train_time_seconds",
        "peak_gpu_allocated_mb",
        "peak_gpu_reserved_mb",
        "trainable_params",
        "trainable_ratio",
    ]

    for r in success:
        key = (r["dataset"], r["method"])
        groups.setdefault(key, []).append(r)

    summary = []
    for (dataset_name, method), rs in sorted(groups.items()):
        row = {
            "dataset": dataset_name,
            "method": method,
            "n_runs": len(rs),
            "seeds": ",".join(str(x["seed"]) for x in rs),
        }
        for metric in metrics:
            vals = [float(x[metric]) for x in rs if x.get(metric) is not None]
            if vals:
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(row)
    return summary


def save_summary(summary, json_path, csv_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if summary:
        fieldnames = []
        seen = set()
        for row in summary:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)


def main():
    config_path = os.path.join(PROJECT_DIR, "configs", "exp1_official.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results_dir = os.path.join(PROJECT_DIR, cfg["output_dir"])
    os.makedirs(results_dir, exist_ok=True)

    results_json = os.path.join(results_dir, "exp1_official_results.json")
    results_csv = os.path.join(results_dir, "exp1_official_results.csv")
    summary_json = os.path.join(results_dir, "exp1_official_summary.json")
    summary_csv = os.path.join(results_dir, "exp1_official_summary.csv")

    rows = load_existing_rows(results_json)
    completed = {
        r.get("run_key")
        for r in rows
        if r.get("status") == "success" and r.get("run_key")
    }

    for seed in cfg["seeds"]:
        for dataset_name in cfg["datasets"]:
            for method in cfg["methods"]:
                key = run_key(dataset_name, method, seed)

                if key in completed:
                    print(f"SKIP completed: {key}")
                    continue

                print("=" * 78)
                print(
                    f"OFFICIAL EXP1 | dataset={dataset_name} | "
                    f"method={method} | seed={seed}"
                )
                print("=" * 78)

                try:
                    row = run_one(method, dataset_name, int(seed), cfg)
                except Exception as e:
                    row = {
                        "run_key": key,
                        "experiment": cfg["experiment_name"],
                        "dataset": dataset_name,
                        "method": method,
                        "seed": int(seed),
                        "status": "failed",
                        "error": repr(e),
                    }

                # Replace any previous row with same run_key.
                rows = [r for r in rows if r.get("run_key") != key]
                rows.append(row)
                save_rows(rows, results_json, results_csv)

                print(json.dumps(row, indent=2))

                if row.get("status") == "success":
                    completed.add(key)

                # Release GPU memory between runs.
                try:
                    del row
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    summary = make_summary(rows)
    save_summary(summary, summary_json, summary_csv)

    print("\n" + "=" * 78)
    print("EXP1 COMPLETE / CURRENT SUMMARY")
    print("=" * 78)
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {results_json}")
    print(f"Summary: {summary_json}")


if __name__ == "__main__":
    main()
