import os, sys, csv, json, time, random, platform, subprocess
import numpy as np
import torch, transformers, peft
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from peft import IA3Config, TaskType, get_peft_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import count_parameters
from models_symmetry import TrueHybridPrefixPromptBertClassifier
from models_peft_symmetry import build_lora_bert

EXPERIMENT = "data_subset_robustness"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)

DATASETS = ["sst2", "imdb"]
METHODS = ["full_ft", "hybrid", "lora", "ia3"]
TRAIN_SUBSET_SEEDS = [2026, 3407, 7777]

TRAIN_SEED = 42
EVAL_SUBSET_SEED = 2026

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TRAIN_SIZE = 3000
EVAL_SIZE = 500
EPOCHS = 3
TRAIN_BATCH = 8
EVAL_BATCH = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

LEARNING_RATES = {
    "sst2": {"full_ft": 2e-5, "hybrid": 3e-4, "lora": 5e-4, "ia3": 5e-4},
    "imdb": {"full_ft": 2e-5, "hybrid": 5e-4, "lora": 3e-4, "ia3": 5e-4},
}

RESULTS_JSON = os.path.join(OUTPUT_DIR, "data_subset_robustness_results.json")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "data_subset_robustness_results.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "data_subset_robustness_summary.json")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "data_subset_robustness_summary.csv")


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
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def build_ia3(num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=int(num_labels)
    )
    cfg = IA3Config(
        task_type=TaskType.SEQ_CLS,
        target_modules=["key", "value", "output.dense"],
        feedforward_modules=["output.dense"],
        modules_to_save=["classifier"],
    )
    return get_peft_model(model, cfg)


def build_model(method, num_labels):
    if method == "full_ft":
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=int(num_labels)
        )
    if method == "hybrid":
        return TrueHybridPrefixPromptBertClassifier(
            MODEL_NAME, int(num_labels), prefix_length=16, prompt_length=20, dropout=0.1
        )
    if method == "lora":
        return build_lora_bert(
            MODEL_NAME, int(num_labels), r=8, alpha=16, dropout=0.1
        )
    if method == "ia3":
        return build_ia3(num_labels)
    raise ValueError(method)


def worker_path(dataset_name, method, subset_seed):
    return os.path.join(
        OUTPUT_DIR, f"worker_{dataset_name}_{method}_subsetseed{subset_seed}.json"
    )


def run_worker(dataset_name, method, subset_seed):
    subset_seed = int(subset_seed)
    set_seed(TRAIN_SEED)

    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )

    eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]

    # Only the training sample changes across runs.
    train_ds = fixed_subset(dataset["train"], TRAIN_SIZE, subset_seed)

    # Evaluation examples stay identical across all runs.
    eval_ds = fixed_subset(eval_source, EVAL_SIZE, EVAL_SUBSET_SEED)

    # Same initialization/training randomness for every subset.
    set_seed(TRAIN_SEED)
    model = build_model(method, num_labels)
    params = count_parameters(model)
    lr = float(LEARNING_RATES[dataset_name][method])

    args = TrainingArguments(
        output_dir=os.path.join(
            OUTPUT_DIR, f"{dataset_name}_{method}_subsetseed{subset_seed}"
        ),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=1,
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
    train_seconds = time.time() - started

    peak_alloc = (
        float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        if torch.cuda.is_available() else 0.0
    )
    peak_res = (
        float(torch.cuda.max_memory_reserved() / (1024 ** 2))
        if torch.cuda.is_available() else 0.0
    )

    metrics = trainer.evaluate()

    row = {
        "experiment": EXPERIMENT,
        "purpose": "training_subset_robustness",
        "dataset": dataset_name,
        "method": method,
        "train_subset_seed": subset_seed,
        "fixed_train_seed": TRAIN_SEED,
        "fixed_eval_subset_seed": EVAL_SUBSET_SEED,
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH,
        "eval_batch_size": EVAL_BATCH,
        "learning_rate": lr,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "linear",
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        **params,
        "train_time_seconds": float(train_seconds),
        "peak_train_gpu_allocated_mb": peak_alloc,
        "peak_train_gpu_reserved_mb": peak_res,
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_mb,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "status": "success",
    }

    if hasattr(train_output, "metrics") and "train_loss" in train_output.metrics:
        row["train_loss"] = float(train_output.metrics["train_loss"])

    for k, v in metrics.items():
        row[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v

    return row


def worker_main(dataset_name, method, subset_seed):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subset_seed = int(subset_seed)
    try:
        row = run_worker(dataset_name, method, subset_seed)
    except Exception as e:
        row = {
            "experiment": EXPERIMENT,
            "dataset": dataset_name,
            "method": method,
            "train_subset_seed": subset_seed,
            "fixed_train_seed": TRAIN_SEED,
            "fixed_eval_subset_seed": EVAL_SUBSET_SEED,
            "status": "failed",
            "error": repr(e),
        }

    with open(worker_path(dataset_name, method, subset_seed), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    print("\nWORKER RESULT")
    print(json.dumps(row, indent=2))

    if row.get("status") != "success":
        raise SystemExit(1)


def collect_rows():
    rows = []
    for d in DATASETS:
        for m in METHODS:
            for s in TRAIN_SUBSET_SEEDS:
                p = worker_path(d, m, s)
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        rows.append(json.load(f))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_summary(rows):
    ok = [r for r in rows if r.get("status") == "success"]
    summary = []

    for d in DATASETS:
        for m in METHODS:
            g = [r for r in ok if r["dataset"] == d and r["method"] == m]
            if not g:
                continue

            item = {
                "dataset": d,
                "method": m,
                "learning_rate": float(LEARNING_RATES[d][m]),
                "n_subset_seeds": len(g),
                "train_subset_seeds": ",".join(str(r["train_subset_seed"]) for r in g),
                "fixed_train_seed": TRAIN_SEED,
                "fixed_eval_subset_seed": EVAL_SUBSET_SEED,
            }

            for metric in ["eval_accuracy", "eval_f1_macro", "eval_f1_weighted"]:
                vals = [float(r[metric]) for r in g if r.get(metric) is not None]
                if vals:
                    item[metric + "_mean"] = float(np.mean(vals))
                    item[metric + "_std"] = (
                        float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    )
                    item[metric + "_min"] = float(np.min(vals))
                    item[metric + "_max"] = float(np.max(vals))

            summary.append(item)

    return summary


def master_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 78)
    print("DATA-SUBSET ROBUSTNESS ANALYSIS")
    print("Only the 3000-example TRAINING subset changes.")
    print("Training seed and evaluation subset remain fixed.")
    print("=" * 78)

    for d in DATASETS:
        for m in METHODS:
            for s in TRAIN_SUBSET_SEEDS:
                p = worker_path(d, m, s)

                if os.path.exists(p):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            old = json.load(f)
                        if old.get("status") == "success":
                            print(f"SKIP completed: {d} / {m} / subset_seed={s}")
                            continue
                    except Exception:
                        pass

                print("\n" + "=" * 78)
                print(
                    f"ROBUSTNESS | dataset={d} | method={m} | "
                    f"subset_seed={s} | train_seed={TRAIN_SEED}"
                )
                print("=" * 78)

                rc = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "--worker", d, m, str(s)]
                ).returncode

                if rc != 0:
                    print(f"FAILED: {d} / {m} / subset_seed={s}; continuing.")

    rows = collect_rows()

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    write_csv(RESULTS_CSV, rows)

    summary = make_summary(rows)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(SUMMARY_CSV, summary)

    print("\n" + "=" * 78)
    print("DATA-SUBSET ROBUSTNESS SUMMARY")
    print("=" * 78)

    for x in summary:
        print(
            f"{x['dataset']:5s} | {x['method']:8s} | n={x['n_subset_seeds']} | "
            f"acc={x.get('eval_accuracy_mean', float('nan')):.4f} "
            f"+/- {x.get('eval_accuracy_std', float('nan')):.4f} | "
            f"range=[{x.get('eval_accuracy_min', float('nan')):.4f}, "
            f"{x.get('eval_accuracy_max', float('nan')):.4f}] | "
            f"f1={x.get('eval_f1_macro_mean', float('nan')):.4f} "
            f"+/- {x.get('eval_f1_macro_std', float('nan')):.4f}"
        )

    print(f"\nResults JSON: {RESULTS_JSON}")
    print(f"Results CSV : {RESULTS_CSV}")
    print(f"Summary JSON: {SUMMARY_JSON}")
    print(f"Summary CSV : {SUMMARY_CSV}")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        master_main()
