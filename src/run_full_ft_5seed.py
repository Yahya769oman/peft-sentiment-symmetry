import os, sys, csv, json, time, random, platform, subprocess
import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import count_parameters

EXPERIMENT = "full_ft_fresh_process"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)

DATASETS = ["sst2", "imdb"]
SEEDS = [42, 123, 2026, 3407, 7777]
DATA_SEED = 2026

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TRAIN_SIZE = 3000
EVAL_SIZE = 500
LEARNING_RATE = 2e-5
EPOCHS = 3
TRAIN_BATCH = 8
EVAL_BATCH = 16
GRAD_ACCUM = 1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

RESULTS_JSON = os.path.join(OUTPUT_DIR, "full_ft_fresh_process_results.json")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "full_ft_fresh_process_results.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "full_ft_fresh_process_summary.json")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "full_ft_fresh_process_summary.csv")

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
        labels, preds, labels=[0,1], average=None, zero_division=0
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

def worker_path(dataset_name, seed):
    return os.path.join(OUTPUT_DIR, f"worker_{dataset_name}_seed{seed}.json")

def run_one(dataset_name, seed):
    set_seed(seed)

    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )

    eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]
    train_ds = fixed_subset(dataset["train"], TRAIN_SIZE, DATA_SEED)
    eval_ds = fixed_subset(eval_source, EVAL_SIZE, DATA_SEED)

    set_seed(seed)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=int(num_labels),
        attn_implementation="eager",
    )
    params = count_parameters(model)

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"{dataset_name}_seed{seed}"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
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

    gpu_name = None
    gpu_total_mb = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_total_mb = float(props.total_memory / (1024**2))

    started = time.time()
    train_output = trainer.train()
    train_seconds = time.time() - started

    peak_alloc = float(torch.cuda.max_memory_allocated()/(1024**2)) if torch.cuda.is_available() else 0.0
    peak_reserved = float(torch.cuda.max_memory_reserved()/(1024**2)) if torch.cuda.is_available() else 0.0

    metrics = trainer.evaluate()

    row = {
        "experiment": EXPERIMENT,
        "dataset": dataset_name,
        "method": "full",
        "resource_protocol": "fresh_python_process_per_run",
        "seed": int(seed),
        "data_seed": DATA_SEED,
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_batch_size": TRAIN_BATCH,
        "eval_batch_size": EVAL_BATCH,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "linear",
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "fp16": bool(torch.cuda.is_available()),
        **params,
        "train_time_seconds": float(train_seconds),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_mb,
        "peak_train_gpu_allocated_mb": peak_alloc,
        "peak_train_gpu_reserved_mb": peak_reserved,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "status": "success",
    }

    if hasattr(train_output, "metrics") and "train_loss" in train_output.metrics:
        row["train_loss"] = float(train_output.metrics["train_loss"])

    for k, v in metrics.items():
        row[k] = float(v) if isinstance(v, (int,float,np.integer,np.floating)) else v

    return row

def worker_main(dataset_name, seed):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        row = run_one(dataset_name, int(seed))
    except Exception as e:
        row = {
            "experiment": EXPERIMENT,
            "dataset": dataset_name,
            "method": "full",
            "seed": int(seed),
            "status": "failed",
            "error": repr(e),
        }
    with open(worker_path(dataset_name, int(seed)), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    print(json.dumps(row, indent=2))
    if row.get("status") != "success":
        raise SystemExit(1)

def collect_rows():
    rows = []
    for dataset_name in DATASETS:
        for seed in SEEDS:
            path = worker_path(dataset_name, seed)
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
                fields.append(key); seen.add(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def summarize(rows):
    ok = [r for r in rows if r.get("status") == "success"]
    metrics = [
        "eval_accuracy","eval_f1_macro","eval_f1_weighted",
        "eval_precision_class0","eval_recall_class0","eval_f1_class0",
        "eval_precision_class1","eval_recall_class1","eval_f1_class1",
        "train_time_seconds","peak_train_gpu_allocated_mb",
        "peak_train_gpu_reserved_mb","trainable_params","trainable_ratio",
    ]
    summary = []
    for dataset_name in DATASETS:
        group = [r for r in ok if r["dataset"] == dataset_name]
        if not group:
            continue
        item = {
            "dataset": dataset_name,
            "method": "full",
            "resource_protocol": "fresh_python_process_per_run",
            "learning_rate": LEARNING_RATE,
            "n_runs": len(group),
            "seeds": ",".join(str(r["seed"]) for r in group),
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in group if r.get(metric) is not None]
            if vals:
                item[metric+"_mean"] = float(np.mean(vals))
                item[metric+"_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(item)
    return summary

def master_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*78)
    print("FULL FINE-TUNING | FRESH PROCESS RESOURCE BENCHMARK")
    print("="*78)

    for dataset_name in DATASETS:
        for seed in SEEDS:
            path = worker_path(dataset_name, seed)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        old = json.load(f)
                    if old.get("status") == "success":
                        print(f"SKIP completed: {dataset_name}/seed={seed}")
                        continue
                except Exception:
                    pass

            print("\n"+"="*78)
            print(f"FULL FT | dataset={dataset_name} | seed={seed} | lr={LEARNING_RATE}")
            print("="*78)

            rc = subprocess.run([
                sys.executable, os.path.abspath(__file__),
                "--worker", dataset_name, str(seed)
            ]).returncode

            if rc != 0:
                print(f"FAILED: {dataset_name}/seed={seed}; continuing.")

    rows = collect_rows()
    save_table(rows, RESULTS_JSON, RESULTS_CSV)

    summary = summarize(rows)
    save_table(summary, SUMMARY_JSON, SUMMARY_CSV)

    print("\n"+"="*78)
    print("FULL FT SUMMARY")
    print("="*78)
    for s in summary:
        print(
            f"{s['dataset']:5s} | n={s['n_runs']} | "
            f"acc={s.get('eval_accuracy_mean', float('nan')):.4f} "
            f"+/- {s.get('eval_accuracy_std', float('nan')):.4f} | "
            f"f1={s.get('eval_f1_macro_mean', float('nan')):.4f} "
            f"+/- {s.get('eval_f1_macro_std', float('nan')):.4f} | "
            f"time={s.get('train_time_seconds_mean', float('nan')):.2f}s | "
            f"gpu={s.get('peak_train_gpu_allocated_mb_mean', float('nan')):.2f}MB"
        )

    print(f"\nResults: {RESULTS_JSON}")
    print(f"Summary: {SUMMARY_JSON}")

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], int(sys.argv[3]))
    else:
        master_main()
