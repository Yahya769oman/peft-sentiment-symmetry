import os
import sys
import json
import time
import random
import platform

import numpy as np
import torch
import yaml
import transformers
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import compute_classification_metrics, count_parameters
from models_symmetry import TrueHybridPrefixPromptBertClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset(split, n, seed):
    n = min(int(n), len(split))
    return split.shuffle(seed=seed).select(range(n))


def clean_number(v):
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    return v


def build_model(method, model_name, num_labels, cfg):
    if method == "full":
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=int(num_labels)
        )
    if method == "hybrid":
        h = cfg["hybrid"]
        return TrueHybridPrefixPromptBertClassifier(
            model_name=model_name,
            num_labels=num_labels,
            prefix_length=int(h["prefix_length"]),
            prompt_length=int(h["prompt_length"]),
            dropout=float(h.get("dropout", 0.1)),
        )
    raise ValueError(f"Unknown method: {method}")


def run_one(method, dataset_name, seed, cfg):
    set_seed(seed)
    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name,
        model_name=cfg["model_name"],
        max_length=int(cfg["max_length"]),
    )

    eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]
    train_ds = subset(dataset["train"], cfg["subset"]["train_size"], seed)
    eval_ds = subset(eval_source, cfg["subset"]["eval_size"], seed)

    model = build_model(method, cfg["model_name"], num_labels, cfg)
    params = count_parameters(model)

    train_cfg = cfg["training"]
    lr = float(cfg["method_overrides"][method]["learning_rate"])
    out_dir = os.path.join(
        PROJECT_DIR,
        cfg["output_dir"],
        f"{dataset_name}_{method}_seed{seed}",
    )
    os.makedirs(out_dir, exist_ok=True)

    use_fp16 = bool(train_cfg.get("fp16", False)) and torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=int(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        learning_rate=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        logging_steps=int(train_cfg.get("logging_steps", 20)),
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
        tokenizer=tokenizer,
        compute_metrics=compute_classification_metrics,
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
        gpu_info["peak_gpu_allocated_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        gpu_info["peak_gpu_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))

    row = {
        "experiment": "exp1_symmetry_smoke",
        "dataset": dataset_name,
        "method": method,
        "seed": int(seed),
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "model_name": cfg["model_name"],
        "max_length": int(cfg["max_length"]),
        "epochs": int(train_cfg["num_train_epochs"]),
        "train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "eval_batch_size": int(train_cfg["per_device_eval_batch_size"]),
        "learning_rate": lr,
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.0)),
        "fp16": use_fp16,
        **params,
        "train_time_seconds": float(train_seconds),
        **gpu_info,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    for k, v in eval_results.items():
        row[k] = clean_number(v)

    return row


def main():
    config_path = os.path.join(PROJECT_DIR, "configs", "exp1_smoke.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rows = []
    for seed in cfg["seeds"]:
        for dataset_name in cfg["datasets"]:
            for method in cfg["methods"]:
                print("=" * 70)
                print(f"SMOKE | dataset={dataset_name} | method={method} | seed={seed}")
                print("=" * 70)
                try:
                    row = run_one(method, dataset_name, int(seed), cfg)
                    row["status"] = "success"
                except Exception as e:
                    row = {
                        "experiment": "exp1_symmetry_smoke",
                        "dataset": dataset_name,
                        "method": method,
                        "seed": int(seed),
                        "status": "failed",
                        "error": repr(e),
                    }
                rows.append(row)
                print(json.dumps(row, indent=2))

    results_path = os.path.join(PROJECT_DIR, cfg["output_dir"], "exp1_smoke_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
