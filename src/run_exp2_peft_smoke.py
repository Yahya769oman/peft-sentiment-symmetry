import os
import sys
import csv
import gc
import json
import time
import random
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import count_parameters
from models_symmetry import TrueHybridPrefixPromptBertClassifier
from models_peft_symmetry import (
    LayerwiseAdapterBertClassifier,
    build_lora_bert,
    build_bitfit_bert,
)

EXPERIMENT = "exp2_peft_smoke"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)

DATASETS = ["sst2", "imdb"]
METHODS = ["hybrid", "lora", "adapter", "bitfit"]

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TRAIN_SIZE = 500
EVAL_SIZE = 200
DATA_SEED = 2026
TRAIN_SEED = 42
EPOCHS = 1
TRAIN_BATCH = 8
EVAL_BATCH = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Smoke-test LRs only. These are NOT the final publication hyperparameters.
SMOKE_LR = {
    "sst2": {
        "hybrid": 3e-4,
        "lora": 3e-4,
        "adapter": 3e-4,
        "bitfit": 3e-4,
    },
    "imdb": {
        "hybrid": 5e-4,
        "lora": 3e-4,
        "adapter": 3e-4,
        "bitfit": 3e-4,
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fixed_subset(split, n, seed):
    n = min(int(n), len(split))
    return split.shuffle(seed=int(seed)).select(range(n))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    p, r, f1c, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_class0": p[0],
        "recall_class0": r[0],
        "f1_class0": f1c[0],
        "support_class0": support[0],
        "precision_class1": p[1],
        "recall_class1": r[1],
        "f1_class1": f1c[1],
        "support_class1": support[1],
    }


def build_model(method, num_labels):
    if method == "hybrid":
        return TrueHybridPrefixPromptBertClassifier(
            model_name=MODEL_NAME,
            num_labels=int(num_labels),
            prefix_length=16,
            prompt_length=20,
            dropout=0.1,
        )

    if method == "lora":
        return build_lora_bert(
            model_name=MODEL_NAME,
            num_labels=int(num_labels),
            r=8,
            alpha=16,
            dropout=0.1,
        )

    if method == "adapter":
        return LayerwiseAdapterBertClassifier(
            model_name=MODEL_NAME,
            num_labels=int(num_labels),
            bottleneck_size=64,
            dropout=0.1,
        )

    if method == "bitfit":
        return build_bitfit_bert(
            model_name=MODEL_NAME,
            num_labels=int(num_labels),
        )

    raise ValueError(f"Unsupported method: {method}")


def worker_result_path(dataset_name, method):
    return os.path.join(OUTPUT_DIR, f"worker_{dataset_name}_{method}.json")


def run_worker(dataset_name, method):
    set_seed(TRAIN_SEED)

    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )
    eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]

    train_ds = fixed_subset(dataset["train"], TRAIN_SIZE, DATA_SEED)
    eval_ds = fixed_subset(eval_source, EVAL_SIZE, DATA_SEED)

    set_seed(TRAIN_SEED)
    model = build_model(method, num_labels)
    params = count_parameters(model)
    lr = float(SMOKE_LR[dataset_name][method])

    run_dir = os.path.join(OUTPUT_DIR, f"{dataset_name}_{method}")
    args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=lr,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        seed=TRAIN_SEED,
        data_seed=TRAIN_SEED,
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

    gpu_name = None
    gpu_total_mb = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_total_mb = float(props.total_memory / (1024 ** 2))

    started = time.time()
    train_output = trainer.train()
    train_time = time.time() - started

    # Capture training peak before evaluation.
    peak_train_allocated = (
        float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        if torch.cuda.is_available() else 0.0
    )
    peak_train_reserved = (
        float(torch.cuda.max_memory_reserved() / (1024 ** 2))
        if torch.cuda.is_available() else 0.0
    )

    metrics = trainer.evaluate()

    row = {
        "experiment": EXPERIMENT,
        "dataset": dataset_name,
        "method": method,
        "seed": TRAIN_SEED,
        "data_seed": DATA_SEED,
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "epochs": EPOCHS,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "learning_rate": lr,
        "train_batch_size": TRAIN_BATCH,
        "eval_batch_size": EVAL_BATCH,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "linear",
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "fp16": bool(torch.cuda.is_available()),
        **params,
        "train_time_seconds": float(train_time),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_mb,
        "peak_train_gpu_allocated_mb": peak_train_allocated,
        "peak_train_gpu_reserved_mb": peak_train_reserved,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "status": "success",
    }

    for k, v in metrics.items():
        row[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v

    if hasattr(train_output, "metrics") and isinstance(train_output.metrics, dict):
        if "train_loss" in train_output.metrics:
            row["train_loss"] = float(train_output.metrics["train_loss"])

    return row


def worker_main(dataset_name, method):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = worker_result_path(dataset_name, method)

    try:
        row = run_worker(dataset_name, method)
    except Exception as e:
        row = {
            "experiment": EXPERIMENT,
            "dataset": dataset_name,
            "method": method,
            "seed": TRAIN_SEED,
            "status": "failed",
            "error": repr(e),
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    print("\nWORKER RESULT")
    print(json.dumps(row, indent=2))

    # Exit non-zero on failure so the master visibly flags it.
    if row.get("status") != "success":
        raise SystemExit(1)


def save_aggregate(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_path = os.path.join(OUTPUT_DIR, "exp2_peft_smoke_results.json")
    csv_path = os.path.join(OUTPUT_DIR, "exp2_peft_smoke_results.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if rows:
        fields, seen = [], set()
        for row in rows:
            for k in row:
                if k not in seen:
                    fields.append(k)
                    seen.add(k)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    return json_path, csv_path


def master_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 78)
    print("EXP2 PEFT SMOKE TEST")
    print("Each method runs in a fresh Python process for clean GPU-memory measurement.")
    print("=" * 78)

    for dataset_name in DATASETS:
        for method in METHODS:
            result_path = worker_result_path(dataset_name, method)

            if os.path.exists(result_path):
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        old = json.load(f)
                    if old.get("status") == "success":
                        print(f"SKIP completed: {dataset_name} / {method}")
                        continue
                except Exception:
                    pass

            print("\n" + "=" * 78)
            print(f"SMOKE | dataset={dataset_name} | method={method}")
            print("=" * 78)

            cmd = [sys.executable, os.path.abspath(__file__), "--worker", dataset_name, method]
            completed = subprocess.run(cmd)

            if completed.returncode != 0:
                print(f"FAILED worker: dataset={dataset_name}, method={method}")
                print("The master will continue so you can see whether other methods work.")

    rows = []
    for dataset_name in DATASETS:
        for method in METHODS:
            path = worker_result_path(dataset_name, method)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    rows.append(json.load(f))

    json_path, csv_path = save_aggregate(rows)

    print("\n" + "=" * 78)
    print("EXP2 SMOKE SUMMARY")
    print("=" * 78)
    for r in rows:
        print(
            f"{r.get('dataset'):5s} | {r.get('method'):7s} | "
            f"status={r.get('status')} | "
            f"acc={r.get('eval_accuracy')} | "
            f"f1={r.get('eval_f1_macro')} | "
            f"trainable={r.get('trainable_params')} | "
            f"gpuMB={r.get('peak_train_gpu_allocated_mb')}"
        )

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], sys.argv[3])
    else:
        master_main()
