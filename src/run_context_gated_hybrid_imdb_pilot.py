import os
import sys
import json
import time
import random
import platform

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
from models_gated_symmetry import ContextGatedHybridBertClassifier

EXPERIMENT = "context_gated_hybrid_imdb_pilot"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)
RESULTS_JSON = os.path.join(OUTPUT_DIR, "context_gated_hybrid_imdb_pilot.json")

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TRAIN_SIZE = 3000
VAL_SIZE = 500
DATA_SEED = 2026
TRAIN_SEED = 42

PREFIX_LENGTH = 16
PROMPT_LENGTH = 20
LEARNING_RATE = 5e-4
EPOCHS = 3
TRAIN_BATCH = 8
EVAL_BATCH = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1


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


@torch.no_grad()
def collect_gate_statistics(model, trainer, max_batches=None):
    model.eval()

    prompt_values = []
    prefix_by_layer = [[] for _ in model.prefix_attention_modules()]

    dataloader = trainer.get_eval_dataloader()

    for batch_i, batch in enumerate(dataloader):
        if max_batches is not None and batch_i >= max_batches:
            break

        batch = trainer._prepare_inputs(batch)
        model(**batch)

        if model.last_prompt_gate_scale is not None:
            prompt_values.extend(
                model.last_prompt_gate_scale.view(-1).float().cpu().tolist()
            )

        for i, module in enumerate(model.prefix_attention_modules()):
            if module.last_gate_scale is not None:
                prefix_by_layer[i].extend(
                    module.last_gate_scale.view(-1).float().cpu().tolist()
                )

    stats = {}

    if prompt_values:
        a = np.asarray(prompt_values, dtype=float)
        stats["prompt_gate_mean"] = float(a.mean())
        stats["prompt_gate_std"] = float(a.std(ddof=0))
        stats["prompt_gate_min"] = float(a.min())
        stats["prompt_gate_max"] = float(a.max())

    layer_means = []
    layer_stds = []
    for values in prefix_by_layer:
        if values:
            a = np.asarray(values, dtype=float)
            layer_means.append(float(a.mean()))
            layer_stds.append(float(a.std(ddof=0)))
        else:
            layer_means.append(None)
            layer_stds.append(None)

    stats["prefix_gate_layer_means"] = layer_means
    stats["prefix_gate_layer_stds"] = layer_stds

    valid = [x for values in prefix_by_layer for x in values]
    if valid:
        a = np.asarray(valid, dtype=float)
        stats["prefix_gate_overall_mean"] = float(a.mean())
        stats["prefix_gate_overall_std"] = float(a.std(ddof=0))
        stats["prefix_gate_overall_min"] = float(a.min())
        stats["prefix_gate_overall_max"] = float(a.max())

    return stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 78)
    print("CONTEXT-GATED HYBRID | IMDb PILOT")
    print("Train-only validation; official IMDb test is untouched.")
    print("=" * 78)

    set_seed(TRAIN_SEED)

    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name="imdb",
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )

    shuffled = dataset["train"].shuffle(seed=DATA_SEED)
    required = TRAIN_SIZE + VAL_SIZE
    if len(shuffled) < required:
        raise ValueError(f"IMDb train split has only {len(shuffled)} examples")

    train_ds = shuffled.select(range(TRAIN_SIZE))
    val_ds = shuffled.select(range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE))

    set_seed(TRAIN_SEED)
    model = ContextGatedHybridBertClassifier(
        model_name=MODEL_NAME,
        num_labels=int(num_labels),
        prefix_length=PREFIX_LENGTH,
        prompt_length=PROMPT_LENGTH,
        dropout=0.1,
    )
    params = count_parameters(model)

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "run"),
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
    train_seconds = time.time() - started

    peak_allocated = (
        float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        if torch.cuda.is_available() else 0.0
    )
    peak_reserved = (
        float(torch.cuda.max_memory_reserved() / (1024 ** 2))
        if torch.cuda.is_available() else 0.0
    )

    metrics = trainer.evaluate()
    gate_stats = collect_gate_statistics(model, trainer)

    row = {
        "experiment": EXPERIMENT,
        "dataset": "imdb",
        "method": "context_gated_hybrid",
        "purpose": "single_seed_train_only_pilot_before_any_official_run",
        "seed": TRAIN_SEED,
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
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "linear",
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "fp16": bool(torch.cuda.is_available()),
        **params,
        "train_time_seconds": float(train_seconds),
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_mb,
        "peak_train_gpu_allocated_mb": peak_allocated,
        "peak_train_gpu_reserved_mb": peak_reserved,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "status": "success",
        **gate_stats,
    }

    if hasattr(train_output, "metrics") and "train_loss" in train_output.metrics:
        row["train_loss"] = float(train_output.metrics["train_loss"])

    for k, v in metrics.items():
        row[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    print("\n" + "=" * 78)
    print("PILOT COMPLETE")
    print("=" * 78)
    print(json.dumps(row, indent=2))
    print(f"\nSaved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
