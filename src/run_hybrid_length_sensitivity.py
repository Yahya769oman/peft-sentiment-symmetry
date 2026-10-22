import os
import sys
import json
import random
import platform
import subprocess
import time

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
from models_symmetry import TrueHybridPrefixPromptBertClassifier

EXPERIMENT = "hybrid_length_sensitivity"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)

DATASETS = ["sst2", "imdb"]
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

LEARNING_RATES = {"sst2": 3e-4, "imdb": 5e-4}
PROMPT_LENGTHS = [10, 20, 30]
PREFIX_LENGTHS = [8, 16, 32]
FIXED_PREFIX_FOR_PROMPT_SWEEP = 16

RESULTS_JSON = os.path.join(OUTPUT_DIR, "hybrid_length_sensitivity_results.json")
BEST_JSON = os.path.join(OUTPUT_DIR, "hybrid_length_sensitivity_best.json")

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

def config_key(dataset_name, prefix_length, prompt_length):
    return f"{dataset_name}__pfx{prefix_length}__prm{prompt_length}"

def worker_path(dataset_name, prefix_length, prompt_length):
    return os.path.join(
        OUTPUT_DIR,
        f"worker_{dataset_name}_pfx{prefix_length}_prm{prompt_length}.json",
    )

def run_worker(dataset_name, prefix_length, prompt_length):
    set_seed(TRAIN_SEED)
    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )

    shuffled = dataset["train"].shuffle(seed=DATA_SEED)
    required = TRAIN_SIZE + VAL_SIZE
    if len(shuffled) < required:
        raise ValueError(f"{dataset_name}: insufficient training examples")

    train_ds = shuffled.select(range(TRAIN_SIZE))
    val_ds = shuffled.select(range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE))

    set_seed(TRAIN_SEED)
    model = TrueHybridPrefixPromptBertClassifier(
        model_name=MODEL_NAME,
        num_labels=int(num_labels),
        prefix_length=int(prefix_length),
        prompt_length=int(prompt_length),
        dropout=0.1,
    )
    params = count_parameters(model)
    lr = float(LEARNING_RATES[dataset_name])

    run_dir = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_pfx{prefix_length}_prm{prompt_length}",
    )

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
    peak_allocated = (
        float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        if torch.cuda.is_available() else 0.0
    )
    metrics = trainer.evaluate()

    row = {
        "run_key": config_key(dataset_name, prefix_length, prompt_length),
        "experiment": EXPERIMENT,
        "dataset": dataset_name,
        "method": "hybrid",
        "purpose": "train_only_prefix_prompt_length_sensitivity",
        "seed": TRAIN_SEED,
        "data_seed": DATA_SEED,
        "train_size": len(train_ds),
        "validation_size": len(val_ds),
        "official_eval_used": False,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "learning_rate": lr,
        "prefix_length": int(prefix_length),
        "prompt_length": int(prompt_length),
        "train_batch_size": TRAIN_BATCH,
        "eval_batch_size": EVAL_BATCH,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        **params,
        "train_time_seconds": float(train_seconds),
        "peak_train_gpu_allocated_mb": peak_allocated,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "status": "success",
    }
    for k, v in metrics.items():
        row[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
    return row

def worker_main(dataset_name, prefix_length, prompt_length):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = worker_path(dataset_name, prefix_length, prompt_length)
    try:
        row = run_worker(dataset_name, int(prefix_length), int(prompt_length))
    except Exception as e:
        row = {
            "run_key": config_key(dataset_name, int(prefix_length), int(prompt_length)),
            "experiment": EXPERIMENT,
            "dataset": dataset_name,
            "method": "hybrid",
            "prefix_length": int(prefix_length),
            "prompt_length": int(prompt_length),
            "status": "failed",
            "error": repr(e),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    print(json.dumps(row, indent=2))
    if row.get("status") != "success":
        raise SystemExit(1)

def load_config(dataset_name, prefix_length, prompt_length):
    path = worker_path(dataset_name, prefix_length, prompt_length)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def run_config(dataset_name, prefix_length, prompt_length):
    old = load_config(dataset_name, prefix_length, prompt_length)
    if old and old.get("status") == "success":
        print(f"SKIP completed: {config_key(dataset_name, prefix_length, prompt_length)}")
        return

    print("\n" + "=" * 78)
    print(
        f"SENSITIVITY | dataset={dataset_name} | "
        f"prefix={prefix_length} | prompt={prompt_length} | "
        f"lr={LEARNING_RATES[dataset_name]}"
    )
    print("=" * 78)

    rc = subprocess.run([
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        dataset_name,
        str(prefix_length),
        str(prompt_length),
    ]).returncode
    if rc != 0:
        print("FAILED configuration; continuing.")

def all_rows():
    rows = []
    if not os.path.isdir(OUTPUT_DIR):
        return rows
    for name in os.listdir(OUTPUT_DIR):
        if name.startswith("worker_") and name.endswith(".json"):
            try:
                with open(os.path.join(OUTPUT_DIR, name), "r", encoding="utf-8") as f:
                    rows.append(json.load(f))
            except Exception:
                pass
    return rows

def choose_best(rows, dataset_name, allowed_configs=None):
    candidates = [
        r for r in rows
        if r.get("status") == "success" and r.get("dataset") == dataset_name
    ]
    if allowed_configs is not None:
        allowed = set(allowed_configs)
        candidates = [
            r for r in candidates
            if (int(r["prefix_length"]), int(r["prompt_length"])) in allowed
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["eval_f1_macro"], r["eval_accuracy"]))

def save_outputs():
    rows = all_rows()
    rows.sort(key=lambda r: (
        r.get("dataset", ""),
        int(r.get("prefix_length", 0)),
        int(r.get("prompt_length", 0)),
    ))
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    best = []
    for d in DATASETS:
        b = choose_best(rows, d)
        if b:
            best.append(b)

    with open(BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    return rows, best

def master_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 78)
    print("HYBRID PREFIX/PROMPT LENGTH SENSITIVITY")
    print("Train-only validation; official evaluation split is untouched.")
    print("=" * 78)

    for dataset_name in DATASETS:
        prompt_configs = [(FIXED_PREFIX_FOR_PROMPT_SWEEP, p) for p in PROMPT_LENGTHS]
        for prefix_length, prompt_length in prompt_configs:
            run_config(dataset_name, prefix_length, prompt_length)

        rows = all_rows()
        best_prompt_row = choose_best(rows, dataset_name, prompt_configs)
        if best_prompt_row is None:
            print(f"No successful prompt sweep for {dataset_name}; skipping prefix sweep.")
            continue

        best_prompt = int(best_prompt_row["prompt_length"])
        print(
            f"\nBest prompt for {dataset_name} at prefix=16: "
            f"{best_prompt} (macro-F1={best_prompt_row['eval_f1_macro']:.4f})"
        )

        for prefix_length in PREFIX_LENGTHS:
            run_config(dataset_name, prefix_length, best_prompt)

    rows, best = save_outputs()

    print("\n" + "=" * 78)
    print("FINAL SENSITIVITY BEST CONFIGS")
    print("=" * 78)
    for r in best:
        print(
            f"{r['dataset']:5s} | prefix={r['prefix_length']:2d} | "
            f"prompt={r['prompt_length']:2d} | "
            f"acc={r['eval_accuracy']:.4f} | "
            f"macroF1={r['eval_f1_macro']:.4f}"
        )

    print(f"\nResults: {RESULTS_JSON}")
    print(f"Best:    {BEST_JSON}")

if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        master_main()
