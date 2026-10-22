import os
import sys
import json
import csv
import time
import random
import platform
import subprocess

import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import count_parameters
from models_symmetry import TrueHybridPrefixPromptBertClassifier

EXPERIMENT = "imdb_sequence_length_sensitivity"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)

DATASET = "imdb"
METHODS = ["full_ft", "hybrid"]
MAX_LENGTHS = [128, 256, 512]
SEEDS = [42, 123, 2026]
DATA_SEED = 2026

MODEL_NAME = "bert-base-uncased"
TRAIN_SIZE = 3000
EVAL_SIZE = 500
EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Preserve the paper's official learning rates.
LEARNING_RATES = {
    "full_ft": 2e-5,
    "hybrid": 5e-4,
}

HYBRID_PREFIX_LENGTH = 16
HYBRID_PROMPT_LENGTH = 20

# Conservative settings for the 4 GB Quadro P520.
# Effective training batch remains 8 through gradient accumulation.
BATCH_CONFIG = {
    128: {
        "full_ft": {"train": 8, "eval": 16, "accum": 1},
        "hybrid":  {"train": 8, "eval": 16, "accum": 1},
    },
    256: {
        "full_ft": {"train": 2, "eval": 8, "accum": 4},
        "hybrid":  {"train": 4, "eval": 8, "accum": 2},
    },
    512: {
        "full_ft": {"train": 1, "eval": 4, "accum": 8},
        "hybrid":  {"train": 1, "eval": 4, "accum": 8},
    },
}

RESULTS_JSON = os.path.join(OUTPUT_DIR, "imdb_sequence_length_results.json")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "imdb_sequence_length_results.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "imdb_sequence_length_summary.json")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "imdb_sequence_length_summary.csv")
LENGTH_DISTRIBUTION_JSON = os.path.join(OUTPUT_DIR, "imdb_heldout_token_length_distribution.json")


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


def heldout_raw_lengths():
    """Original token lengths for the exact fixed 500-example IMDb held-out subset."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(LENGTH_DISTRIBUTION_JSON):
        with open(LENGTH_DISTRIBUTION_JSON, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj["token_lengths"]

    raw = load_dataset("imdb")
    held = fixed_subset(raw["test"], EVAL_SIZE, DATA_SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    lengths = []
    for batch_start in range(0, len(held), 64):
        batch = held[batch_start: batch_start + 64]["text"]
        enc = tok(batch, add_special_tokens=True, truncation=False, padding=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])

    arr = np.array(lengths, dtype=np.int64)
    dist = {
        "dataset": DATASET,
        "eval_size": int(len(arr)),
        "data_seed": DATA_SEED,
        "tokenizer": MODEL_NAME,
        "token_lengths": [int(x) for x in arr],
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "n_gt_128": int(np.sum(arr > 128)),
        "pct_gt_128": float(100 * np.mean(arr > 128)),
        "n_gt_256": int(np.sum(arr > 256)),
        "pct_gt_256": float(100 * np.mean(arr > 256)),
        "n_gt_492": int(np.sum(arr > 492)),
        "pct_gt_492": float(100 * np.mean(arr > 492)),
        "n_gt_512": int(np.sum(arr > 512)),
        "pct_gt_512": float(100 * np.mean(arr > 512)),
    }
    with open(LENGTH_DISTRIBUTION_JSON, "w", encoding="utf-8") as f:
        json.dump(dist, f, indent=2)
    return dist["token_lengths"]


def bucket_metrics(labels, preds, raw_lengths):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    lens = np.asarray(raw_lengths)

    # Buckets defined by original, untruncated tokenizer length.
    specs = [
        ("le128", lens <= 128),
        ("129_256", (lens > 128) & (lens <= 256)),
        ("gt256", lens > 256),
    ]
    out = {}
    for name, mask in specs:
        n = int(mask.sum())
        out[f"bucket_{name}_n"] = n
        if n:
            out[f"bucket_{name}_accuracy"] = float(accuracy_score(labels[mask], preds[mask]))
            out[f"bucket_{name}_f1_macro"] = float(f1_score(labels[mask], preds[mask], average="macro"))
        else:
            out[f"bucket_{name}_accuracy"] = None
            out[f"bucket_{name}_f1_macro"] = None
    return out


def worker_path(method, max_length, seed):
    return os.path.join(
        OUTPUT_DIR, f"worker_{METHOD_SAFE(method)}_len{max_length}_seed{seed}.json"
    )


def METHOD_SAFE(method):
    return method.replace("-", "_")


def build_model(method, num_labels, max_length):
    if method == "full_ft":
        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=int(num_labels)
        )
        if max_length >= 256:
            model.gradient_checkpointing_enable()
        return model

    if method == "hybrid":
        model = TrueHybridPrefixPromptBertClassifier(
            model_name=MODEL_NAME,
            num_labels=int(num_labels),
            prefix_length=HYBRID_PREFIX_LENGTH,
            prompt_length=HYBRID_PROMPT_LENGTH,
            dropout=0.1,
        )
        if max_length >= 256:
            # Enable checkpointing on the underlying BERT encoder.
            model.bert.gradient_checkpointing_enable()
        return model

    raise ValueError(method)


def run_worker(method, max_length, seed):
    set_seed(seed)
    cfg = BATCH_CONFIG[int(max_length)][method]

    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=DATASET,
        model_name=MODEL_NAME,
        max_length=int(max_length),
    )
    eval_source = dataset["test"]
    train_ds = fixed_subset(dataset["train"], TRAIN_SIZE, DATA_SEED)
    eval_ds = fixed_subset(eval_source, EVAL_SIZE, DATA_SEED)

    set_seed(seed)
    model = build_model(method, num_labels, int(max_length))
    params = count_parameters(model)

    # At nominal 512, Hybrid must reserve 20 BERT positions for soft prompts.
    # Thus its maximum number of real text-token positions is 492.
    if method == "hybrid":
        effective_real_token_cap = min(
            int(max_length),
            int(model.bert.config.max_position_embeddings) - HYBRID_PROMPT_LENGTH,
        )
    else:
        effective_real_token_cap = int(max_length)

    run_dir = os.path.join(
        OUTPUT_DIR, f"{METHOD_SAFE(method)}_len{max_length}_seed{seed}"
    )

    args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=cfg["train"],
        per_device_eval_batch_size=cfg["eval"],
        gradient_accumulation_steps=cfg["accum"],
        learning_rate=LEARNING_RATES[method],
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
        dataloader_pin_memory=False,
        eval_accumulation_steps=1,
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
    gpu_total = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_total = float(props.total_memory / (1024 ** 2))

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

    pred_output = trainer.predict(eval_ds)
    logits = pred_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = np.asarray(pred_output.label_ids)
    preds = np.argmax(logits, axis=-1)

    raw_lengths = heldout_raw_lengths()
    buckets = bucket_metrics(labels, preds, raw_lengths)

    row = {
        "experiment": EXPERIMENT,
        "dataset": DATASET,
        "method": method,
        "max_length_setting": int(max_length),
        "effective_real_token_cap": int(effective_real_token_cap),
        "seed": int(seed),
        "data_seed": DATA_SEED,
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "model_name": MODEL_NAME,
        "epochs": EPOCHS,
        "train_batch_size": cfg["train"],
        "eval_batch_size": cfg["eval"],
        "gradient_accumulation_steps": cfg["accum"],
        "effective_train_batch_size": cfg["train"] * cfg["accum"],
        "learning_rate": LEARNING_RATES[method],
        "prefix_length": HYBRID_PREFIX_LENGTH if method == "hybrid" else 0,
        "prompt_length": HYBRID_PROMPT_LENGTH if method == "hybrid" else 0,
        "gradient_checkpointing": bool(int(max_length) >= 256),
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "linear",
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "fp16": bool(torch.cuda.is_available()),
        **params,
        "train_time_seconds": float(train_seconds),
        "peak_train_gpu_allocated_mb": peak_alloc,
        "peak_train_gpu_reserved_mb": peak_reserved,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "status": "success",
        **buckets,
    }

    if hasattr(train_output, "metrics") and "train_loss" in train_output.metrics:
        row["train_loss"] = float(train_output.metrics["train_loss"])

    for k, v in pred_output.metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            row[k] = float(v)
        else:
            row[k] = v

    return row


def worker_main(method, max_length, seed):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = worker_path(method, int(max_length), int(seed))
    try:
        row = run_worker(method, int(max_length), int(seed))
    except Exception as e:
        row = {
            "experiment": EXPERIMENT,
            "dataset": DATASET,
            "method": method,
            "max_length_setting": int(max_length),
            "seed": int(seed),
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
    for length in MAX_LENGTHS:
        for method in METHODS:
            for seed in SEEDS:
                p = worker_path(method, length, seed)
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        rows.append(json.load(f))
    return rows


def summarize(rows):
    ok = [r for r in rows if r.get("status") == "success"]
    out = []

    metrics = [
        "test_accuracy",
        "test_f1_macro",
        "test_f1_weighted",
        "train_time_seconds",
        "peak_train_gpu_allocated_mb",
        "bucket_le128_accuracy",
        "bucket_129_256_accuracy",
        "bucket_gt256_accuracy",
    ]

    for length in MAX_LENGTHS:
        for method in METHODS:
            g = [
                r for r in ok
                if r["method"] == method
                and int(r["max_length_setting"]) == int(length)
            ]
            if not g:
                continue

            s = {
                "dataset": DATASET,
                "method": method,
                "max_length_setting": int(length),
                "effective_real_token_cap": int(g[0]["effective_real_token_cap"]),
                "n_runs": len(g),
                "seeds": ",".join(str(r["seed"]) for r in g),
                "learning_rate": float(g[0]["learning_rate"]),
                "effective_train_batch_size": int(g[0]["effective_train_batch_size"]),
                "gradient_checkpointing": bool(g[0]["gradient_checkpointing"]),
                "trainable_params": int(g[0]["trainable_params"]),
            }
            for m in metrics:
                vals = [
                    float(r[m]) for r in g
                    if r.get(m) is not None
                ]
                if vals:
                    s[m + "_mean"] = float(np.mean(vals))
                    s[m + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

            # Report held-out length bucket counts once.
            for key in ["bucket_le128_n", "bucket_129_256_n", "bucket_gt256_n"]:
                s[key] = int(g[0][key])

            out.append(s)

    return out


def save_table(rows, path):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def parent_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create token-length distribution first.
    lengths = heldout_raw_lengths()
    with open(LENGTH_DISTRIBUTION_JSON, "r", encoding="utf-8") as f:
        dist = json.load(f)

    print("=" * 92)
    print("IMDb SEQUENCE-LENGTH SENSITIVITY | FULL FT vs HYBRID")
    print("Diagnostic evaluation at max_length 128 / 256 / 512 over 3 seeds.")
    print("Effective training batch size is kept at 8; gradient checkpointing is enabled at 256/512.")
    print("For Hybrid at nominal 512, the 20 soft prompts leave a 492-real-token cap.")
    print("=" * 92)
    print(
        f"Held-out original token lengths: mean={dist['mean']:.1f}, median={dist['median']:.1f}, "
        f">128={dist['pct_gt_128']:.1f}%, >256={dist['pct_gt_256']:.1f}%, "
        f">492={dist['pct_gt_492']:.1f}%, >512={dist['pct_gt_512']:.1f}%"
    )

    for length in MAX_LENGTHS:
        for method in METHODS:
            for seed in SEEDS:
                p = worker_path(method, length, seed)
                if os.path.exists(p):
                    try:
                        old = json.load(open(p, "r", encoding="utf-8"))
                        if old.get("status") == "success":
                            print(f"SKIP completed: {method} | len={length} | seed={seed}")
                            continue
                    except Exception:
                        pass

                print("\n" + "=" * 92)
                print(f"RUN | IMDb | {method} | max_length={length} | seed={seed}")
                print("=" * 92)

                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--worker",
                    method,
                    str(length),
                    str(seed),
                ]
                completed = subprocess.run(cmd)

                if completed.returncode != 0:
                    print(
                        f"WARNING: run failed for {method}/len={length}/seed={seed}. "
                        f"The failure was saved; continuing to the next run."
                    )

    rows = collect_rows()
    summary = summarize(rows)

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_table(rows, RESULTS_CSV)
    save_table(summary, SUMMARY_CSV)

    print("\n" + "=" * 92)
    print("FINAL SUMMARY")
    print("=" * 92)
    for s in summary:
        acc = s.get("test_accuracy_mean")
        sd = s.get("test_accuracy_std")
        print(
            f"{s['method']:7s} | len={s['max_length_setting']:3d} "
            f"(real cap={s['effective_real_token_cap']:3d}) | n={s['n_runs']} | "
            f"acc={acc:.4f} +/- {sd:.4f}"
            if acc is not None else
            f"{s['method']:7s} | len={s['max_length_setting']:3d} | n={s['n_runs']}"
        )

    print("\nResults:", RESULTS_JSON)
    print("Summary:", SUMMARY_JSON)
    print("Length distribution:", LENGTH_DISTRIBUTION_JSON)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        parent_main()
