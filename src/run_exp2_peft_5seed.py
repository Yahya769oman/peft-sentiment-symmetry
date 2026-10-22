import os, sys, csv, json, time, random, platform, subprocess
import numpy as np
import torch, transformers
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import count_parameters
from models_symmetry import TrueHybridPrefixPromptBertClassifier
from models_peft_symmetry import LayerwiseAdapterBertClassifier, build_lora_bert, build_bitfit_bert

EXPERIMENT = "exp2_peft_official"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)
DATASETS = ["sst2", "imdb"]
METHODS = ["hybrid", "lora", "adapter", "bitfit"]
SEEDS = [42, 123, 2026, 3407, 7777]
DATA_SEED = 2026
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
    "sst2": {"hybrid": 3e-4, "lora": 5e-4, "adapter": 3e-4, "bitfit": 3e-4},
    "imdb": {"hybrid": 5e-4, "lora": 3e-4, "adapter": 1e-4, "bitfit": 3e-4},
}

RESULTS_JSON = os.path.join(OUTPUT_DIR, "exp2_peft_official_results.json")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "exp2_peft_official_results.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "exp2_peft_official_summary.json")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "exp2_peft_official_summary.csv")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fixed_subset(split, n, seed):
    return split.shuffle(seed=int(seed)).select(range(min(int(n), len(split))))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple): logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    p, r, f1c, support = precision_recall_fscore_support(labels, preds, labels=[0,1], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_class0": p[0], "recall_class0": r[0], "f1_class0": f1c[0], "support_class0": support[0],
        "precision_class1": p[1], "recall_class1": r[1], "f1_class1": f1c[1], "support_class1": support[1],
    }

def build_model(method, num_labels):
    if method == "hybrid":
        return TrueHybridPrefixPromptBertClassifier(MODEL_NAME, int(num_labels), prefix_length=16, prompt_length=20, dropout=0.1)
    if method == "lora":
        return build_lora_bert(MODEL_NAME, int(num_labels), r=8, alpha=16, dropout=0.1)
    if method == "adapter":
        return LayerwiseAdapterBertClassifier(MODEL_NAME, int(num_labels), bottleneck_size=64, dropout=0.1)
    if method == "bitfit":
        return build_bitfit_bert(MODEL_NAME, int(num_labels))
    raise ValueError(method)

def run_key(dataset_name, method, seed):
    return f"{dataset_name}__{method}__seed{seed}"

def worker_path(dataset_name, method, seed):
    return os.path.join(OUTPUT_DIR, f"worker_{dataset_name}_{method}_seed{seed}.json")

def run_worker(dataset_name, method, seed):
    set_seed(seed)
    dataset, tokenizer, num_labels = load_and_prepare_dataset(dataset_name=dataset_name, model_name=MODEL_NAME, max_length=MAX_LENGTH)
    eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]
    train_ds = fixed_subset(dataset["train"], TRAIN_SIZE, DATA_SEED)
    eval_ds = fixed_subset(eval_source, EVAL_SIZE, DATA_SEED)
    set_seed(seed)
    model = build_model(method, num_labels)
    params = count_parameters(model)
    lr = float(LEARNING_RATES[dataset_name][method])

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"{dataset_name}_{method}_seed{seed}"),
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
        seed=seed,
        data_seed=seed,
        save_safetensors=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      processing_class=tokenizer, compute_metrics=compute_metrics)

    gpu_name = None; gpu_total_mb = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name; gpu_total_mb = float(props.total_memory/(1024**2))

    started = time.time()
    train_output = trainer.train()
    train_seconds = time.time() - started
    peak_alloc = float(torch.cuda.max_memory_allocated()/(1024**2)) if torch.cuda.is_available() else 0.0
    peak_res = float(torch.cuda.max_memory_reserved()/(1024**2)) if torch.cuda.is_available() else 0.0
    metrics = trainer.evaluate()

    row = {
        "run_key": run_key(dataset_name, method, seed),
        "experiment": EXPERIMENT, "dataset": dataset_name, "method": method, "seed": seed, "data_seed": DATA_SEED,
        "train_size": len(train_ds), "eval_size": len(eval_ds), "model_name": MODEL_NAME, "max_length": MAX_LENGTH,
        "epochs": EPOCHS, "train_batch_size": TRAIN_BATCH, "eval_batch_size": EVAL_BATCH,
        "gradient_accumulation_steps": 1, "learning_rate": lr, "optimizer": "adamw_torch",
        "lr_scheduler_type": "linear", "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
        "fp16": bool(torch.cuda.is_available()),
        "prefix_length": 16 if method=="hybrid" else None,
        "prompt_length": 20 if method=="hybrid" else None,
        "lora_r": 8 if method=="lora" else None,
        "lora_alpha": 16 if method=="lora" else None,
        "adapter_bottleneck": 64 if method=="adapter" else None,
        **params,
        "train_time_seconds": float(train_seconds),
        "cuda_available": bool(torch.cuda.is_available()), "gpu_name": gpu_name, "gpu_total_memory_mb": gpu_total_mb,
        "peak_train_gpu_allocated_mb": peak_alloc, "peak_train_gpu_reserved_mb": peak_res,
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "status": "success",
    }
    if hasattr(train_output, "metrics") and "train_loss" in train_output.metrics:
        row["train_loss"] = float(train_output.metrics["train_loss"])
    for k,v in metrics.items():
        row[k] = float(v) if isinstance(v,(int,float,np.integer,np.floating)) else v
    return row

def worker_main(dataset_name, method, seed):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        row = run_worker(dataset_name, method, int(seed))
    except Exception as e:
        row = {"run_key": run_key(dataset_name, method, int(seed)), "experiment": EXPERIMENT,
               "dataset": dataset_name, "method": method, "seed": int(seed),
               "learning_rate": float(LEARNING_RATES[dataset_name][method]),
               "status": "failed", "error": repr(e)}
    path = worker_path(dataset_name, method, int(seed))
    with open(path, "w", encoding="utf-8") as f: json.dump(row, f, indent=2)
    print(json.dumps(row, indent=2))
    if row.get("status") != "success": raise SystemExit(1)

def collect_rows():
    rows=[]
    for d in DATASETS:
        for m in METHODS:
            for s in SEEDS:
                p = worker_path(d,m,s)
                if os.path.exists(p):
                    with open(p,"r",encoding="utf-8") as f: rows.append(json.load(f))
    return rows

def save_results(rows):
    with open(RESULTS_JSON,"w",encoding="utf-8") as f: json.dump(rows,f,indent=2)
    if rows:
        fields=[]; seen=set()
        for r in rows:
            for k in r:
                if k not in seen: fields.append(k); seen.add(k)
        with open(RESULTS_CSV,"w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def make_summary(rows):
    ok=[r for r in rows if r.get("status")=="success"]
    metrics=["eval_accuracy","eval_f1_macro","eval_f1_weighted","eval_precision_class0","eval_recall_class0","eval_f1_class0",
             "eval_precision_class1","eval_recall_class1","eval_f1_class1","train_time_seconds",
             "peak_train_gpu_allocated_mb","peak_train_gpu_reserved_mb","trainable_params","trainable_ratio"]
    summary=[]
    for d in DATASETS:
        for m in METHODS:
            g=[r for r in ok if r["dataset"]==d and r["method"]==m]
            if not g: continue
            s={"dataset":d,"method":m,"learning_rate":float(LEARNING_RATES[d][m]),"n_runs":len(g),
               "seeds":",".join(str(r["seed"]) for r in g)}
            for metric in metrics:
                vals=[float(r[metric]) for r in g if r.get(metric) is not None]
                if vals:
                    s[metric+"_mean"]=float(np.mean(vals))
                    s[metric+"_std"]=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0
            summary.append(s)
    with open(SUMMARY_JSON,"w",encoding="utf-8") as f: json.dump(summary,f,indent=2)
    if summary:
        fields=[]; seen=set()
        for r in summary:
            for k in r:
                if k not in seen: fields.append(k); seen.add(k)
        with open(SUMMARY_CSV,"w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)
    return summary

def master_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*78)
    print("OFFICIAL EXP2 | HYBRID vs LoRA vs ADAPTER vs BitFit")
    print("="*78)
    for d in DATASETS:
        for m in METHODS:
            for s in SEEDS:
                p=worker_path(d,m,s)
                if os.path.exists(p):
                    try:
                        with open(p,"r",encoding="utf-8") as f: old=json.load(f)
                        if old.get("status")=="success":
                            print(f"SKIP completed: {d} / {m} / seed={s}")
                            continue
                    except Exception:
                        pass
                print("\n"+"="*78)
                print(f"OFFICIAL EXP2 | dataset={d} | method={m} | seed={s} | lr={LEARNING_RATES[d][m]}")
                print("="*78)
                rc=subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", d, m, str(s)]).returncode
                if rc != 0:
                    print(f"FAILED: {d} / {m} / seed={s}; continuing.")
    rows=collect_rows(); save_results(rows); summary=make_summary(rows)
    print("\n"+"="*78); print("EXP2 COMPLETE / CURRENT SUMMARY"); print("="*78)
    for s in summary:
        print(f"{s['dataset']:5s} | {s['method']:7s} | n={s['n_runs']} | "
              f"acc={s.get('eval_accuracy_mean',float('nan')):.4f} +/- {s.get('eval_accuracy_std',float('nan')):.4f} | "
              f"f1={s.get('eval_f1_macro_mean',float('nan')):.4f} +/- {s.get('eval_f1_macro_std',float('nan')):.4f} | "
              f"trainable={s.get('trainable_params_mean',float('nan')):.0f} | "
              f"gpuMB={s.get('peak_train_gpu_allocated_mb_mean',float('nan')):.1f}")
    print(f"\nResults: {RESULTS_JSON}")
    print(f"Summary: {SUMMARY_JSON}")

if __name__=="__main__":
    if len(sys.argv)==5 and sys.argv[1]=="--worker":
        worker_main(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        master_main()
