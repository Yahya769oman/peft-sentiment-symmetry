import os
import sys
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import count_parameters
from models_symmetry import TrueHybridPrefixPromptBertClassifier


MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TRAIN_SIZE = 3000
VAL_SIZE = 500
DATA_SEED = 2026
TRAIN_SEED = 42

PREFIX_LENGTH = 16
PROMPT_LENGTH = 20

LEARNING_RATES = [1e-4, 3e-4, 5e-4]

EPOCHS = 3
TRAIN_BATCH = 8
EVAL_BATCH = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "imdb_hybrid_lr_tuning")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "imdb_hybrid_lr_tuning_results.json")


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


def load_existing():
    if not os.path.exists(RESULTS_PATH):
        return []
    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            x = json.load(f)
        return x if isinstance(x, list) else []
    except Exception:
        return []


def save(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # IMPORTANT: tuning uses ONLY the IMDb training split.
    # The official IMDb test split is never touched here.
    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name="imdb",
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )

    shuffled_train = dataset["train"].shuffle(seed=DATA_SEED)
    required = TRAIN_SIZE + VAL_SIZE
    if len(shuffled_train) < required:
        raise ValueError(f"IMDb train split is too small: {len(shuffled_train)}")

    train_ds = shuffled_train.select(range(TRAIN_SIZE))
    val_ds = shuffled_train.select(range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE))

    rows = load_existing()
    completed = {float(r["learning_rate"]) for r in rows if r.get("status") == "success"}

    for lr in LEARNING_RATES:
        if float(lr) in completed:
            print(f"SKIP completed learning_rate={lr}")
            continue

        print("=" * 78)
        print(f"IMDb HYBRID LR TUNING | lr={lr} | seed={TRAIN_SEED}")
        print("Validation is carved from IMDb TRAIN split; official TEST is untouched.")
        print("=" * 78)

        set_seed(TRAIN_SEED)

        model = TrueHybridPrefixPromptBertClassifier(
            model_name=MODEL_NAME,
            num_labels=int(num_labels),
            prefix_length=PREFIX_LENGTH,
            prompt_length=PROMPT_LENGTH,
            dropout=0.1,
        )
        params = count_parameters(model)

        run_dir = os.path.join(OUTPUT_DIR, f"lr_{lr:.0e}".replace("+", ""))
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
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
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
        try:
            trainer.train()
            metrics = trainer.evaluate()
            elapsed = time.time() - started

            row = {
                "dataset": "imdb",
                "method": "hybrid",
                "purpose": "learning_rate_tuning_on_train_validation_only",
                "learning_rate": float(lr),
                "seed": TRAIN_SEED,
                "data_seed": DATA_SEED,
                "train_size": len(train_ds),
                "validation_size": len(val_ds),
                "official_test_used": False,
                "epochs": EPOCHS,
                "prefix_length": PREFIX_LENGTH,
                "prompt_length": PROMPT_LENGTH,
                "train_batch_size": TRAIN_BATCH,
                "eval_batch_size": EVAL_BATCH,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                **params,
                "train_time_seconds": float(elapsed),
                "peak_gpu_allocated_mb": (
                    float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                    if torch.cuda.is_available() else 0.0
                ),
                "eval_loss": float(metrics["eval_loss"]),
                "eval_accuracy": float(metrics["eval_accuracy"]),
                "eval_f1_macro": float(metrics["eval_f1_macro"]),
                "eval_f1_weighted": float(metrics["eval_f1_weighted"]),
                "status": "success",
            }
        except Exception as e:
            row = {
                "dataset": "imdb",
                "method": "hybrid",
                "learning_rate": float(lr),
                "seed": TRAIN_SEED,
                "status": "failed",
                "error": repr(e),
            }

        rows = [r for r in rows if float(r.get("learning_rate", -1)) != float(lr)]
        rows.append(row)
        save(rows)
        print(json.dumps(row, indent=2))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    successful = [r for r in rows if r.get("status") == "success"]
    if successful:
        best = max(successful, key=lambda r: (r["eval_f1_macro"], r["eval_accuracy"]))
        print("\n" + "=" * 78)
        print("BEST VALIDATION CONFIG")
        print("=" * 78)
        print(json.dumps(best, indent=2))

    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
