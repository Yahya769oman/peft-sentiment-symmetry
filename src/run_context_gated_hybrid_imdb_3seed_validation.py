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
from models_gated_symmetry import ContextGatedHybridBertClassifier

EXPERIMENT = "context_gated_hybrid_imdb_3seed_validation"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TRAIN_SIZE = 3000
VAL_SIZE = 500
DATA_SEED = 2026
SEEDS = [42, 123, 2026]

PREFIX_LENGTH = 16
PROMPT_LENGTH = 20
LEARNING_RATE = 5e-4
EPOCHS = 3
TRAIN_BATCH = 8
EVAL_BATCH = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

RESULTS_JSON = os.path.join(OUTPUT_DIR, "context_gated_hybrid_imdb_3seed_results.json")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "context_gated_hybrid_imdb_3seed_summary.json")


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


def worker_path(seed):
    return os.path.join(OUTPUT_DIR, f"worker_seed{seed}.json")


@torch.no_grad()
def gate_stats(model, trainer):
    model.eval()
    prompt_values = []
    layer_values = [[] for _ in model.prefix_attention_modules()]

    for batch in trainer.get_eval_dataloader():
        batch = trainer._prepare_inputs(batch)
        model(**batch)

        if model.last_prompt_gate_scale is not None:
            prompt_values.extend(
                model.last_prompt_gate_scale.view(-1).float().cpu().tolist()
            )

        for i, module in enumerate(model.prefix_attention_modules()):
            if module.last_gate_scale is not None:
                layer_values[i].extend(
                    module.last_gate_scale.view(-1).float().cpu().tolist()
                )

    out = {}
    if prompt_values:
        a = np.asarray(prompt_values, dtype=float)
        out["prompt_gate_mean"] = float(a.mean())
        out["prompt_gate_std"] = float(a.std(ddof=0))

    layer_means = []
    for vals in layer_values:
        if vals:
            a = np.asarray(vals, dtype=float)
            layer_means.append(float(a.mean()))
        else:
            layer_means.append(None)

    out["prefix_gate_layer_means"] = layer_means

    flat = [x for vals in layer_values for x in vals]
    if flat:
        a = np.asarray(flat, dtype=float)
        out["prefix_gate_overall_mean"] = float(a.mean())
        out["prefix_gate_overall_std"] = float(a.std(ddof=0))
    return out


def run_one(seed):
    set_seed(seed)

    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name="imdb",
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )

    # Same train-only split for every seed; official test untouched.
    shuffled = dataset["train"].shuffle(seed=DATA_SEED)
    train_ds = shuffled.select(range(TRAIN_SIZE))
    val_ds = shuffled.select(range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE))

    set_seed(seed)
    model = ContextGatedHybridBertClassifier(
        model_name=MODEL_NAME,
        num_labels=int(num_labels),
        prefix_length=PREFIX_LENGTH,
        prompt_length=PROMPT_LENGTH,
        dropout=0.1,
    )
    params = count_parameters(model)

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"seed{seed}"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LEARNING_RATE,
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
    elapsed = time.time() - started
    metrics = trainer.evaluate()
    gates = gate_stats(model, trainer)

    row = {
        "experiment": EXPERIMENT,
        "dataset": "imdb",
        "method": "context_gated_hybrid",
        "purpose": "three_seed_train_only_validation_before_official_test",
        "seed": int(seed),
        "data_seed": DATA_SEED,
        "train_size": len(train_ds),
        "validation_size": len(val_ds),
        "official_test_used": False,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "prefix_length": PREFIX_LENGTH,
        "prompt_length": PROMPT_LENGTH,
        "train_batch_size": TRAIN_BATCH,
        "eval_batch_size": EVAL_BATCH,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        **params,
        "train_time_seconds": float(elapsed),
        "peak_train_gpu_allocated_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if torch.cuda.is_available() else 0.0
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "status": "success",
        **gates,
    }

    for k, v in metrics.items():
        row[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v

    return row


def worker_main(seed):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        row = run_one(int(seed))
    except Exception as e:
        row = {
            "experiment": EXPERIMENT,
            "dataset": "imdb",
            "method": "context_gated_hybrid",
            "seed": int(seed),
            "status": "failed",
            "error": repr(e),
        }

    with open(worker_path(int(seed)), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    print(json.dumps(row, indent=2))
    if row.get("status") != "success":
        raise SystemExit(1)


def collect():
    rows = []
    for seed in SEEDS:
        p = worker_path(seed)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
    return rows


def summarize(rows):
    ok = [r for r in rows if r.get("status") == "success"]

    def mean_std(key):
        vals = [float(r[key]) for r in ok if r.get(key) is not None]
        if not vals:
            return None, None
        return (
            float(np.mean(vals)),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        )

    summary = {
        "dataset": "imdb",
        "method": "context_gated_hybrid",
        "n_runs": len(ok),
        "seeds": [r["seed"] for r in ok],
        "official_test_used": False,
        "learning_rate": LEARNING_RATE,
        "prefix_length": PREFIX_LENGTH,
        "prompt_length": PROMPT_LENGTH,
    }

    for key in [
        "eval_accuracy",
        "eval_f1_macro",
        "eval_f1_weighted",
        "train_time_seconds",
        "peak_train_gpu_allocated_mb",
        "trainable_params",
        "trainable_ratio",
        "prompt_gate_mean",
        "prefix_gate_overall_mean",
    ]:
        mean, std = mean_std(key)
        if mean is not None:
            summary[key + "_mean"] = mean
            summary[key + "_std"] = std

    return summary


def master_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 78)
    print("CONTEXT-GATED HYBRID | IMDb | 3-SEED TRAIN-ONLY VALIDATION")
    print("=" * 78)

    for seed in SEEDS:
        p = worker_path(seed)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    old = json.load(f)
                if old.get("status") == "success":
                    print(f"SKIP completed seed={seed}")
                    continue
            except Exception:
                pass

        rc = subprocess.run([
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            str(seed),
        ]).returncode

        if rc != 0:
            print(f"FAILED seed={seed}; continuing.")

    rows = collect()
    summary = summarize(rows)

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("3-SEED VALIDATION SUMMARY")
    print("=" * 78)
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {RESULTS_JSON}")
    print(f"Summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker_main(int(sys.argv[2]))
    else:
        master_main()
