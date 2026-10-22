import os
import sys
import csv
import json
import time
import random
import platform
import subprocess

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
from models_ablation_symmetry import PrefixOnlyBertClassifier, PromptOnlyBertClassifier

EXPERIMENT = "exp3_ablation_independent_tuning"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)

DATASETS = ["sst2", "imdb"]
METHODS = ["prefix_only", "prompt_only"]
LEARNING_RATES = [1e-4, 3e-4, 5e-4]
PREFIX_LENGTHS = [8, 16, 32]
PROMPT_LENGTHS = [10, 20, 30]

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TRAIN_SIZE = 3000
VAL_SIZE = 500
EVAL_SIZE = 500
DATA_SEED = 2026
TUNING_SEED = 42
OFFICIAL_SEEDS = [42, 123, 2026, 3407, 7777]

EPOCHS = 3
TRAIN_BATCH = 8
EVAL_BATCH = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

TUNING_RESULTS_JSON = os.path.join(OUTPUT_DIR, "ablation_tuning_results.json")
TUNING_RESULTS_CSV = os.path.join(OUTPUT_DIR, "ablation_tuning_results.csv")
BEST_JSON = os.path.join(OUTPUT_DIR, "ablation_tuning_best.json")
OFFICIAL_RESULTS_JSON = os.path.join(OUTPUT_DIR, "ablation_tuned_official_results.json")
OFFICIAL_RESULTS_CSV = os.path.join(OUTPUT_DIR, "ablation_tuned_official_results.csv")
OFFICIAL_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "ablation_tuned_official_summary.json")
OFFICIAL_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "ablation_tuned_official_summary.csv")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fixed_subset(split, n, seed):
    return split.shuffle(seed=int(seed)).select(range(min(int(n), len(split))))


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


def lengths_for(method):
    return PREFIX_LENGTHS if method == "prefix_only" else PROMPT_LENGTHS


def build_model(method, num_labels, length):
    if method == "prefix_only":
        return PrefixOnlyBertClassifier(
            model_name=MODEL_NAME,
            num_labels=int(num_labels),
            prefix_length=int(length),
            dropout=0.1,
        )
    if method == "prompt_only":
        return PromptOnlyBertClassifier(
            model_name=MODEL_NAME,
            num_labels=int(num_labels),
            prompt_length=int(length),
            dropout=0.1,
        )
    raise ValueError(method)


def safe_lr(lr):
    return f"{float(lr):.0e}".replace("+", "")


def tuning_worker_path(dataset_name, method, lr, length):
    return os.path.join(
        OUTPUT_DIR,
        f"worker_tune_{dataset_name}_{method}_lr{safe_lr(lr)}_len{int(length)}.json",
    )


def official_worker_path(dataset_name, method, seed):
    return os.path.join(
        OUTPUT_DIR,
        f"worker_official_{dataset_name}_{method}_seed{int(seed)}.json",
    )


def make_args(output_dir, lr, seed):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=1,
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
        seed=int(seed),
        data_seed=int(seed),
        save_safetensors=False,
    )


def gpu_metadata():
    if not torch.cuda.is_available():
        return None, None
    props = torch.cuda.get_device_properties(0)
    return props.name, float(props.total_memory / (1024 ** 2))


def run_tuning_worker(dataset_name, method, lr, length):
    set_seed(TUNING_SEED)
    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )

    # Hyperparameter selection uses the training split only.
    shuffled = dataset["train"].shuffle(seed=DATA_SEED)
    required = TRAIN_SIZE + VAL_SIZE
    if len(shuffled) < required:
        raise ValueError(
            f"{dataset_name}: training split has {len(shuffled)} examples; "
            f"need at least {required}."
        )
    train_ds = shuffled.select(range(TRAIN_SIZE))
    val_ds = shuffled.select(range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE))

    set_seed(TUNING_SEED)
    model = build_model(method, num_labels, length)
    params = count_parameters(model)

    run_dir = os.path.join(
        OUTPUT_DIR,
        f"tune_{dataset_name}_{method}_lr{safe_lr(lr)}_len{int(length)}",
    )
    trainer = Trainer(
        model=model,
        args=make_args(run_dir, lr, TUNING_SEED),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    train_output = trainer.train()
    train_seconds = time.time() - started
    metrics = trainer.evaluate()
    gpu_name, gpu_total_mb = gpu_metadata()

    row = {
        "experiment": EXPERIMENT,
        "phase": "training_only_hyperparameter_tuning",
        "dataset": dataset_name,
        "method": method,
        "learning_rate": float(lr),
        "prefix_length": int(length) if method == "prefix_only" else 0,
        "prompt_length": int(length) if method == "prompt_only" else 0,
        "seed": TUNING_SEED,
        "data_seed": DATA_SEED,
        "train_size": len(train_ds),
        "validation_size": len(val_ds),
        "official_eval_used": False,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH,
        "eval_batch_size": EVAL_BATCH,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "linear",
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "fp16": bool(torch.cuda.is_available()),
        **params,
        "train_time_seconds": float(train_seconds),
        "peak_train_gpu_allocated_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if torch.cuda.is_available() else 0.0
        ),
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_mb,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "status": "success",
    }
    if hasattr(train_output, "metrics") and "train_loss" in train_output.metrics:
        row["train_loss"] = float(train_output.metrics["train_loss"])
    for k, v in metrics.items():
        row[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
    return row


def tuning_worker_main(dataset_name, method, lr, length):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        row = run_tuning_worker(dataset_name, method, float(lr), int(length))
    except Exception as e:
        row = {
            "experiment": EXPERIMENT,
            "phase": "training_only_hyperparameter_tuning",
            "dataset": dataset_name,
            "method": method,
            "learning_rate": float(lr),
            "length": int(length),
            "status": "failed",
            "error": repr(e),
        }
    path = tuning_worker_path(dataset_name, method, float(lr), int(length))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    print(json.dumps(row, indent=2))
    if row.get("status") != "success":
        raise SystemExit(1)


def collect_tuning_rows():
    rows = []
    for dataset_name in DATASETS:
        for method in METHODS:
            for lr in LEARNING_RATES:
                for length in lengths_for(method):
                    path = tuning_worker_path(dataset_name, method, lr, length)
                    if os.path.exists(path):
                        with open(path, "r", encoding="utf-8") as f:
                            rows.append(json.load(f))
    return rows


def choose_best(rows):
    best = []
    ok = [r for r in rows if r.get("status") == "success"]
    for dataset_name in DATASETS:
        for method in METHODS:
            candidates = [
                r for r in ok
                if r.get("dataset") == dataset_name and r.get("method") == method
            ]
            if not candidates:
                continue
            # Primary selection metric: Macro-F1; accuracy is the tie-breaker.
            # If predictive metrics tie, prefer fewer trainable parameters and then smaller LR.
            winner = max(
                candidates,
                key=lambda r: (
                    r.get("eval_f1_macro", -1.0),
                    r.get("eval_accuracy", -1.0),
                    -r.get("trainable_params", 10**18),
                    -r.get("learning_rate", 1.0),
                ),
            ).copy()
            winner["selection_metric"] = "validation Macro-F1 (accuracy tie-breaker)"
            best.append(winner)
    return best


def best_lookup(best_rows, dataset_name, method):
    for r in best_rows:
        if r.get("dataset") == dataset_name and r.get("method") == method:
            return r
    raise KeyError(f"No tuned configuration for {dataset_name}/{method}")


def run_official_worker(dataset_name, method, seed, best_cfg):
    set_seed(seed)
    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )
    eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]
    train_ds = fixed_subset(dataset["train"], TRAIN_SIZE, DATA_SEED)
    eval_ds = fixed_subset(eval_source, EVAL_SIZE, DATA_SEED)

    lr = float(best_cfg["learning_rate"])
    length = int(best_cfg["prefix_length"] if method == "prefix_only" else best_cfg["prompt_length"])

    set_seed(seed)
    model = build_model(method, num_labels, length)
    params = count_parameters(model)
    run_dir = os.path.join(OUTPUT_DIR, f"official_{dataset_name}_{method}_seed{int(seed)}")
    trainer = Trainer(
        model=model,
        args=make_args(run_dir, lr, seed),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    train_output = trainer.train()
    train_seconds = time.time() - started
    peak_alloc = (
        float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        if torch.cuda.is_available() else 0.0
    )
    peak_reserved = (
        float(torch.cuda.max_memory_reserved() / (1024 ** 2))
        if torch.cuda.is_available() else 0.0
    )
    metrics = trainer.evaluate()
    gpu_name, gpu_total_mb = gpu_metadata()

    row = {
        "experiment": EXPERIMENT,
        "phase": "tuned_heldout_evaluation",
        "dataset": dataset_name,
        "method": method,
        "seed": int(seed),
        "data_seed": DATA_SEED,
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "learning_rate": lr,
        "prefix_length": length if method == "prefix_only" else 0,
        "prompt_length": length if method == "prompt_only" else 0,
        "train_batch_size": TRAIN_BATCH,
        "eval_batch_size": EVAL_BATCH,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "linear",
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "fp16": bool(torch.cuda.is_available()),
        "selection_metric": best_cfg.get("selection_metric"),
        **params,
        "train_time_seconds": float(train_seconds),
        "peak_train_gpu_allocated_mb": peak_alloc,
        "peak_train_gpu_reserved_mb": peak_reserved,
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_mb,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "status": "success",
    }
    if hasattr(train_output, "metrics") and "train_loss" in train_output.metrics:
        row["train_loss"] = float(train_output.metrics["train_loss"])
    for k, v in metrics.items():
        row[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
    return row


def official_worker_main(dataset_name, method, seed):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(BEST_JSON, "r", encoding="utf-8") as f:
        best_rows = json.load(f)
    best_cfg = best_lookup(best_rows, dataset_name, method)
    try:
        row = run_official_worker(dataset_name, method, int(seed), best_cfg)
    except Exception as e:
        row = {
            "experiment": EXPERIMENT,
            "phase": "tuned_heldout_evaluation",
            "dataset": dataset_name,
            "method": method,
            "seed": int(seed),
            "status": "failed",
            "error": repr(e),
        }
    path = official_worker_path(dataset_name, method, int(seed))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    print(json.dumps(row, indent=2))
    if row.get("status") != "success":
        raise SystemExit(1)


def collect_official_rows():
    rows = []
    for dataset_name in DATASETS:
        for method in METHODS:
            for seed in OFFICIAL_SEEDS:
                path = official_worker_path(dataset_name, method, seed)
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        rows.append(json.load(f))
    return rows


def save_table(rows, json_path, csv_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_official(rows):
    ok = [r for r in rows if r.get("status") == "success"]
    metrics = [
        "eval_accuracy", "eval_f1_macro", "eval_f1_weighted",
        "eval_precision_class0", "eval_recall_class0", "eval_f1_class0",
        "eval_precision_class1", "eval_recall_class1", "eval_f1_class1",
        "train_time_seconds", "peak_train_gpu_allocated_mb",
        "peak_train_gpu_reserved_mb", "trainable_params", "trainable_ratio",
    ]
    summary = []
    for dataset_name in DATASETS:
        for method in METHODS:
            group = [
                r for r in ok
                if r.get("dataset") == dataset_name and r.get("method") == method
            ]
            if not group:
                continue
            item = {
                "dataset": dataset_name,
                "method": method,
                "protocol": "independently_tuned_then_heldout_five_seed",
                "learning_rate": group[0].get("learning_rate"),
                "prefix_length": group[0].get("prefix_length"),
                "prompt_length": group[0].get("prompt_length"),
                "n_runs": len(group),
                "seeds": ",".join(str(r["seed"]) for r in group),
            }
            for metric in metrics:
                vals = [float(r[metric]) for r in group if r.get(metric) is not None]
                if vals:
                    item[metric + "_mean"] = float(np.mean(vals))
                    item[metric + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary.append(item)
    return summary


def config_matches(row, best_cfg, method):
    if row.get("status") != "success":
        return False
    if float(row.get("learning_rate", -1)) != float(best_cfg["learning_rate"]):
        return False
    expected_length = int(best_cfg["prefix_length"] if method == "prefix_only" else best_cfg["prompt_length"])
    actual_length = int(row.get("prefix_length", 0) if method == "prefix_only" else row.get("prompt_length", 0))
    return actual_length == expected_length


def master_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 88)
    print("EXP3 INDEPENDENT ABLATION TUNING")
    print("Tuning uses only a validation slice from the training split; held-out evaluation remains untouched.")
    print("Full grid: LR x component length for Prefix-only and Prompt-only, followed by five-seed held-out evaluation.")
    print("=" * 88)

    # Phase 1: full independent hyperparameter sweep.
    for dataset_name in DATASETS:
        for method in METHODS:
            for lr in LEARNING_RATES:
                for length in lengths_for(method):
                    path = tuning_worker_path(dataset_name, method, lr, length)
                    if os.path.exists(path):
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                old = json.load(f)
                            if old.get("status") == "success":
                                print(f"SKIP tuned: {dataset_name}/{method}/lr={lr}/len={length}")
                                continue
                        except Exception:
                            pass
                    print("\n" + "=" * 88)
                    print(f"TUNING | {dataset_name} | {method} | lr={lr} | length={length}")
                    print("=" * 88)
                    rc = subprocess.run([
                        sys.executable, os.path.abspath(__file__), "--tune-worker",
                        dataset_name, method, str(lr), str(length),
                    ]).returncode
                    if rc != 0:
                        print(f"FAILED tuning: {dataset_name}/{method}/lr={lr}/len={length}; continuing.")

    tuning_rows = collect_tuning_rows()
    save_table(tuning_rows, TUNING_RESULTS_JSON, TUNING_RESULTS_CSV)
    best = choose_best(tuning_rows)
    with open(BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\n" + "=" * 88)
    print("SELECTED CONFIGURATIONS (training-only validation)")
    print("=" * 88)
    for b in best:
        length = b.get("prefix_length") if b["method"] == "prefix_only" else b.get("prompt_length")
        print(
            f"{b['dataset']:5s} | {b['method']:11s} | lr={b['learning_rate']:.0e} | "
            f"length={length} | val Macro-F1={b.get('eval_f1_macro', float('nan')):.4f} | "
            f"val Acc={b.get('eval_accuracy', float('nan')):.4f}"
        )

    # Phase 2: evaluate each independently tuned ablation on the untouched held-out split.
    for dataset_name in DATASETS:
        for method in METHODS:
            best_cfg = best_lookup(best, dataset_name, method)
            for seed in OFFICIAL_SEEDS:
                path = official_worker_path(dataset_name, method, seed)
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            old = json.load(f)
                        if config_matches(old, best_cfg, method):
                            print(f"SKIP official: {dataset_name}/{method}/seed={seed}")
                            continue
                    except Exception:
                        pass
                print("\n" + "=" * 88)
                print(f"OFFICIAL | {dataset_name} | {method} | seed={seed}")
                print("=" * 88)
                rc = subprocess.run([
                    sys.executable, os.path.abspath(__file__), "--official-worker",
                    dataset_name, method, str(seed),
                ]).returncode
                if rc != 0:
                    print(f"FAILED official: {dataset_name}/{method}/seed={seed}; continuing.")

    official_rows = collect_official_rows()
    save_table(official_rows, OFFICIAL_RESULTS_JSON, OFFICIAL_RESULTS_CSV)
    summary = summarize_official(official_rows)
    save_table(summary, OFFICIAL_SUMMARY_JSON, OFFICIAL_SUMMARY_CSV)

    print("\n" + "=" * 88)
    print("TUNED ABLATION HELD-OUT SUMMARY")
    print("=" * 88)
    for s in summary:
        length = s.get("prefix_length") if s["method"] == "prefix_only" else s.get("prompt_length")
        print(
            f"{s['dataset']:5s} | {s['method']:11s} | lr={s['learning_rate']:.0e} | length={length} | "
            f"n={s['n_runs']} | Acc={s.get('eval_accuracy_mean', float('nan')):.4f} +/- "
            f"{s.get('eval_accuracy_std', float('nan')):.4f} | Macro-F1={s.get('eval_f1_macro_mean', float('nan')):.4f} +/- "
            f"{s.get('eval_f1_macro_std', float('nan')):.4f}"
        )
    print(f"\nBest configs: {BEST_JSON}")
    print(f"Official summary: {OFFICIAL_SUMMARY_JSON}")


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--tune-worker":
        tuning_worker_main(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]))
    elif len(sys.argv) == 5 and sys.argv[1] == "--official-worker":
        official_worker_main(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        master_main()
