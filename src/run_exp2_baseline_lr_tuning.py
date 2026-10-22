import os
import sys
import json
import time
import random
import platform
import subprocess

import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import count_parameters
from models_peft_symmetry import (
    LayerwiseAdapterBertClassifier,
    build_lora_bert,
    build_bitfit_bert,
)

EXPERIMENT = "exp2_baseline_lr_tuning"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)

DATASETS = ["sst2", "imdb"]
METHODS = ["lora", "adapter", "bitfit"]
LEARNING_RATES = [1e-4, 3e-4, 5e-4]

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TRAIN_SIZE = 3000
VAL_SIZE = 500
DATA_SEED = 2026
TRAIN_SEED = 42
EPOCHS = 3
TRAIN_BATCH = 8
EVAL_BATCH = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

RESULTS_JSON = os.path.join(OUTPUT_DIR, "exp2_baseline_lr_tuning_results.json")
BEST_JSON = os.path.join(OUTPUT_DIR, "exp2_baseline_lr_tuning_best.json")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def build_model(method, num_labels):
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

    raise ValueError(method)


def key(dataset_name, method, lr):
    return f"{dataset_name}__{method}__lr{lr:.0e}"


def worker_path(dataset_name, method, lr):
    safe_lr = f"{lr:.0e}".replace("+", "")
    return os.path.join(OUTPUT_DIR, f"worker_{dataset_name}_{method}_{safe_lr}.json")


def run_worker(dataset_name, method, lr):
    set_seed(TRAIN_SEED)

    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )

    # Tuning is performed ONLY on the official training split.
    # The official validation/test split is untouched.
    shuffled = dataset["train"].shuffle(seed=DATA_SEED)
    required = TRAIN_SIZE + VAL_SIZE
    if len(shuffled) < required:
        raise ValueError(f"{dataset_name} train split has only {len(shuffled)} examples")

    train_ds = shuffled.select(range(TRAIN_SIZE))
    val_ds = shuffled.select(range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE))

    set_seed(TRAIN_SEED)
    model = build_model(method, num_labels)
    params = count_parameters(model)

    run_dir = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_{method}_{lr:.0e}".replace("+", ""),
    )

    args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=float(lr),
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        logging_steps=50,
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
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    trainer.train()
    train_seconds = time.time() - started
    metrics = trainer.evaluate()

    row = {
        "run_key": key(dataset_name, method, lr),
        "experiment": EXPERIMENT,
        "dataset": dataset_name,
        "method": method,
        "learning_rate": float(lr),
        "seed": TRAIN_SEED,
        "data_seed": DATA_SEED,
        "train_size": len(train_ds),
        "validation_size": len(val_ds),
        "official_eval_used": False,
        "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH,
        "eval_batch_size": EVAL_BATCH,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "linear",
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        **params,
        "train_time_seconds": float(train_seconds),
        "peak_train_gpu_allocated_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if torch.cuda.is_available() else 0.0
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "status": "success",
    }

    for k, v in metrics.items():
        row[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v

    return row


def worker_main(dataset_name, method, lr):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = worker_path(dataset_name, method, lr)

    try:
        row = run_worker(dataset_name, method, lr)
    except Exception as e:
        row = {
            "run_key": key(dataset_name, method, lr),
            "experiment": EXPERIMENT,
            "dataset": dataset_name,
            "method": method,
            "learning_rate": float(lr),
            "seed": TRAIN_SEED,
            "status": "failed",
            "error": repr(e),
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    print("\nWORKER RESULT")
    print(json.dumps(row, indent=2))

    if row.get("status") != "success":
        raise SystemExit(1)


def collect_rows():
    rows = []
    for dataset_name in DATASETS:
        for method in METHODS:
            for lr in LEARNING_RATES:
                path = worker_path(dataset_name, method, lr)
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        rows.append(json.load(f))
    return rows


def save_results(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    best = []
    for dataset_name in DATASETS:
        for method in METHODS:
            candidates = [
                r for r in rows
                if r.get("status") == "success"
                and r.get("dataset") == dataset_name
                and r.get("method") == method
            ]
            if candidates:
                winner = max(
                    candidates,
                    key=lambda r: (r.get("eval_f1_macro", -1), r.get("eval_accuracy", -1))
                )
                best.append(winner)

    with open(BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    return best


def master_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 78)
    print("EXP2 BASELINE LR TUNING")
    print("Official evaluation splits are NOT used.")
    print("Each run is a fresh Python process.")
    print("=" * 78)

    for dataset_name in DATASETS:
        for method in METHODS:
            for lr in LEARNING_RATES:
                path = worker_path(dataset_name, method, lr)

                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            old = json.load(f)
                        if old.get("status") == "success":
                            print(f"SKIP completed: {dataset_name} / {method} / lr={lr}")
                            continue
                    except Exception:
                        pass

                print("\n" + "=" * 78)
                print(f"TUNING | dataset={dataset_name} | method={method} | lr={lr}")
                print("=" * 78)

                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--worker",
                    dataset_name,
                    method,
                    str(lr),
                ]
                completed = subprocess.run(cmd)

                if completed.returncode != 0:
                    print(f"FAILED: {dataset_name} / {method} / lr={lr}")
                    print("Continuing to the next configuration.")

    rows = collect_rows()
    best = save_results(rows)

    print("\n" + "=" * 78)
    print("BEST LR PER DATASET / METHOD")
    print("=" * 78)
    for r in best:
        print(
            f"{r['dataset']:5s} | {r['method']:7s} | "
            f"lr={r['learning_rate']} | "
            f"acc={r['eval_accuracy']:.4f} | "
            f"macroF1={r['eval_f1_macro']:.4f}"
        )

    print(f"\nAll results: {RESULTS_JSON}")
    print(f"Best configs: {BEST_JSON}")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    else:
        master_main()
