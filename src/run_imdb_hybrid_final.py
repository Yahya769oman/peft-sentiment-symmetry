import os, sys, csv, json, time, random, platform
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

EXPERIMENT = "imdb_hybrid_final_lr5e4"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TRAIN_SIZE = 3000
EVAL_SIZE = 500
DATA_SEED = 2026
SEEDS = [42, 123, 2026]
PREFIX_LENGTH = 16
PROMPT_LENGTH = 20
DROPOUT = 0.1
LEARNING_RATE = 5e-4
EPOCHS = 3
TRAIN_BATCH = 8
EVAL_BATCH = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
RESULTS_JSON = os.path.join(OUTPUT_DIR, "imdb_hybrid_final_results.json")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "imdb_hybrid_final_summary.json")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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
        labels, preds, labels=[0,1], average=None, zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_class0": p[0], "recall_class0": r[0], "f1_class0": f1c[0],
        "support_class0": support[0],
        "precision_class1": p[1], "recall_class1": r[1], "f1_class1": f1c[1],
        "support_class1": support[1],
    }

def load_rows():
    if not os.path.exists(RESULTS_JSON):
        return []
    try:
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            x = json.load(f)
        return x if isinstance(x, list) else []
    except Exception:
        return []

def save_rows(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

def summarize(rows):
    ok = [r for r in rows if r.get("status") == "success"]
    metrics = ["eval_accuracy","eval_f1_macro","eval_f1_weighted",
               "train_time_seconds","peak_gpu_allocated_mb","peak_gpu_reserved_mb",
               "trainable_params","trainable_ratio"]
    s = {
        "dataset":"imdb","method":"hybrid","learning_rate":LEARNING_RATE,
        "n_runs":len(ok),"seeds":[r["seed"] for r in ok]
    }
    for m in metrics:
        vals = [float(r[m]) for r in ok if r.get(m) is not None]
        if vals:
            s[m+"_mean"] = float(np.mean(vals))
            s[m+"_std"] = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    return s

def run_one(seed):
    set_seed(seed)
    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name="imdb", model_name=MODEL_NAME, max_length=MAX_LENGTH
    )
    train_ds = fixed_subset(dataset["train"], TRAIN_SIZE, DATA_SEED)
    eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]
    eval_ds = fixed_subset(eval_source, EVAL_SIZE, DATA_SEED)

    set_seed(seed)
    model = TrueHybridPrefixPromptBertClassifier(
        model_name=MODEL_NAME, num_labels=int(num_labels),
        prefix_length=PREFIX_LENGTH, prompt_length=PROMPT_LENGTH, dropout=DROPOUT
    )
    params = count_parameters(model)

    run_dir = os.path.join(OUTPUT_DIR, f"seed{seed}")
    args = TrainingArguments(
        output_dir=run_dir, num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO, lr_scheduler_type="linear",
        optim="adamw_torch", logging_steps=50,
        eval_strategy="epoch", save_strategy="no", report_to="none",
        remove_unused_columns=False, fp16=torch.cuda.is_available(),
        seed=seed, data_seed=seed, save_safetensors=False
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer, compute_metrics=compute_metrics
    )

    gpu_name = None
    gpu_total = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_total = float(props.total_memory/(1024**2))

    started = time.time()
    trainer.train()
    elapsed = time.time() - started
    metrics = trainer.evaluate()

    row = {
        "run_key":f"imdb__hybrid__lr5e-4__seed{seed}",
        "experiment":EXPERIMENT,"dataset":"imdb","method":"hybrid",
        "seed":seed,"data_seed":DATA_SEED,"train_size":len(train_ds),"eval_size":len(eval_ds),
        "model_name":MODEL_NAME,"max_length":MAX_LENGTH,"epochs":EPOCHS,
        "train_batch_size":TRAIN_BATCH,"eval_batch_size":EVAL_BATCH,
        "learning_rate":LEARNING_RATE,"optimizer":"adamw_torch",
        "lr_scheduler_type":"linear","weight_decay":WEIGHT_DECAY,"warmup_ratio":WARMUP_RATIO,
        "fp16":bool(torch.cuda.is_available()),"prefix_length":PREFIX_LENGTH,"prompt_length":PROMPT_LENGTH,
        **params,
        "train_time_seconds":float(elapsed),"cuda_available":bool(torch.cuda.is_available()),
        "gpu_name":gpu_name,"gpu_total_memory_mb":gpu_total,
        "peak_gpu_allocated_mb":float(torch.cuda.max_memory_allocated()/(1024**2)) if torch.cuda.is_available() else 0.0,
        "peak_gpu_reserved_mb":float(torch.cuda.max_memory_reserved()/(1024**2)) if torch.cuda.is_available() else 0.0,
        "python_version":platform.python_version(),"torch_version":torch.__version__,
        "transformers_version":transformers.__version__,"status":"success"
    }
    for k,v in metrics.items():
        row[k] = float(v) if isinstance(v,(int,float,np.integer,np.floating)) else v
    return row

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = load_rows()
    completed = {int(r["seed"]) for r in rows if r.get("status")=="success"}
    for seed in SEEDS:
        if seed in completed:
            print(f"SKIP completed seed={seed}")
            continue
        print("="*78)
        print(f"FINAL IMDb HYBRID | lr={LEARNING_RATE} | seed={seed}")
        print("="*78)
        try:
            row = run_one(seed)
        except Exception as e:
            row = {"run_key":f"imdb__hybrid__lr5e-4__seed{seed}",
                   "experiment":EXPERIMENT,"dataset":"imdb","method":"hybrid",
                   "seed":seed,"learning_rate":LEARNING_RATE,
                   "status":"failed","error":repr(e)}
        rows = [r for r in rows if r.get("seed") != seed]
        rows.append(row)
        save_rows(rows)
        print(json.dumps(row, indent=2))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize(rows)
    print("\n"+"="*78)
    print("FINAL IMDb HYBRID SUMMARY")
    print("="*78)
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {RESULTS_JSON}")
    print(f"Summary: {SUMMARY_JSON}")

if __name__ == "__main__":
    main()
