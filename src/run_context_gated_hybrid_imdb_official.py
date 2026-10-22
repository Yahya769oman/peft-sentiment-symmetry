
import os, sys, json, time, random, platform, subprocess
import numpy as np, torch, transformers
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

HERE=os.path.dirname(os.path.abspath(__file__))
PROJ=os.path.dirname(HERE)
if HERE not in sys.path: sys.path.append(HERE)

from data import load_and_prepare_dataset
from metrics import count_parameters
from models_gated_symmetry import ContextGatedHybridBertClassifier

OUT=os.path.join(PROJ,"outputs","context_gated_hybrid_imdb_official")
SEEDS=[42,123,2026]
DATA_SEED=2026

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def metrics_fn(ep):
    logits, labels = ep
    if isinstance(logits, tuple): logits=logits[0]
    pred=np.argmax(logits,axis=-1)
    return {
        "accuracy": accuracy_score(labels,pred),
        "f1_macro": f1_score(labels,pred,average="macro"),
        "f1_weighted": f1_score(labels,pred,average="weighted")
    }

def worker_file(s): return os.path.join(OUT,f"worker_seed{s}.json")

def run_one(s):
    seed_all(s)
    ds,tok,nlab=load_and_prepare_dataset(
        dataset_name="imdb", model_name="bert-base-uncased", max_length=128
    )
    train=ds["train"].shuffle(seed=DATA_SEED).select(range(3000))
    test=ds["test"].shuffle(seed=DATA_SEED).select(range(500))

    seed_all(s)
    model=ContextGatedHybridBertClassifier(
        model_name="bert-base-uncased", num_labels=int(nlab),
        prefix_length=16, prompt_length=20, dropout=0.1
    )
    params=count_parameters(model)

    args=TrainingArguments(
        output_dir=os.path.join(OUT,f"seed{s}"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        seed=s, data_seed=s,
        save_safetensors=False
    )

    tr=Trainer(
        model=model,args=args,train_dataset=train,eval_dataset=test,
        processing_class=tok,compute_metrics=metrics_fn
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    t0=time.time(); tr.train(); elapsed=time.time()-t0
    peak=float(torch.cuda.max_memory_allocated()/(1024**2)) if torch.cuda.is_available() else 0.0
    ev=tr.evaluate()

    row={
        "experiment":"context_gated_hybrid_imdb_official",
        "dataset":"imdb","method":"context_gated_hybrid",
        "purpose":"official_fixed_evaluation_after_train_only_validation",
        "seed":s,"data_seed":DATA_SEED,
        "train_size":len(train),"eval_size":len(test),
        "official_test_used":True,
        "model_name":"bert-base-uncased","max_length":128,
        "epochs":3,"learning_rate":5e-4,
        "prefix_length":16,"prompt_length":20,
        "train_batch_size":8,"eval_batch_size":16,
        "weight_decay":0.01,"warmup_ratio":0.1,
        **params,
        "train_time_seconds":elapsed,
        "peak_train_gpu_allocated_mb":peak,
        "python_version":platform.python_version(),
        "torch_version":torch.__version__,
        "transformers_version":transformers.__version__,
        "status":"success"
    }
    for k,v in ev.items():
        row[k]=float(v) if isinstance(v,(int,float,np.integer,np.floating)) else v
    return row

def worker(s):
    os.makedirs(OUT,exist_ok=True)
    try: row=run_one(s)
    except Exception as e:
        row={"dataset":"imdb","method":"context_gated_hybrid","seed":s,"status":"failed","error":repr(e)}
    with open(worker_file(s),"w",encoding="utf-8") as f: json.dump(row,f,indent=2)
    print(json.dumps(row,indent=2))
    if row["status"]!="success": raise SystemExit(1)

def collect():
    rows=[]
    for s in SEEDS:
        p=worker_file(s)
        if os.path.exists(p):
            with open(p,encoding="utf-8") as f: rows.append(json.load(f))
    return rows

def main():
    os.makedirs(OUT,exist_ok=True)
    for s in SEEDS:
        p=worker_file(s)
        if os.path.exists(p):
            try:
                with open(p,encoding="utf-8") as f:
                    if json.load(f).get("status")=="success":
                        print(f"SKIP seed={s}"); continue
            except: pass
        subprocess.run([sys.executable,os.path.abspath(__file__),"--worker",str(s)])

    rows=collect()
    ok=[r for r in rows if r.get("status")=="success"]
    summary={
        "dataset":"imdb","method":"context_gated_hybrid",
        "n_runs":len(ok),"seeds":[r["seed"] for r in ok],
        "official_test_used":True,"learning_rate":5e-4,
        "prefix_length":16,"prompt_length":20
    }
    for key in ["eval_accuracy","eval_f1_macro","eval_f1_weighted","train_time_seconds",
                "peak_train_gpu_allocated_mb","trainable_params","trainable_ratio"]:
        vals=[float(r[key]) for r in ok if r.get(key) is not None]
        if vals:
            summary[key+"_mean"]=float(np.mean(vals))
            summary[key+"_std"]=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0

    with open(os.path.join(OUT,"context_gated_hybrid_imdb_official_results.json"),"w",encoding="utf-8") as f:
        json.dump(rows,f,indent=2)
    with open(os.path.join(OUT,"context_gated_hybrid_imdb_official_summary.json"),"w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--worker": worker(int(sys.argv[2]))
    else: main()
