import os
import sys
import gc
import csv
import json
import time
import random
import platform
import subprocess
from contextlib import nullcontext

import numpy as np
import torch
import transformers
import peft
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import IA3Config, TaskType, get_peft_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import count_parameters
from models_symmetry import TrueHybridPrefixPromptBertClassifier
from models_peft_symmetry import (
    LayerwiseAdapterBertClassifier,
    build_lora_bert,
    build_bitfit_bert,
)

EXPERIMENT = "trained_checkpoint_inference_benchmark"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

DATASETS = ["sst2", "imdb"]
METHODS = ["full_ft", "hybrid", "lora", "adapter", "bitfit", "ia3"]

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TRAIN_SIZE = 3000
EVAL_SIZE = 500
DATA_SEED = 2026
TRAIN_SEED = 42

EPOCHS = 3
TRAIN_BATCH = 8
EVAL_BATCH = 16
GRAD_ACCUM = 1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

INFERENCE_BATCH = 16
WARMUP_BATCHES = 5
TIMED_REPEATS = 5
USE_FP16 = True

LEARNING_RATES = {
    "sst2": {
        "full_ft": 2e-5,
        "hybrid": 3e-4,
        "lora": 5e-4,
        "adapter": 3e-4,
        "bitfit": 3e-4,
        "ia3": 5e-4,
    },
    "imdb": {
        "full_ft": 2e-5,
        "hybrid": 5e-4,
        "lora": 3e-4,
        "adapter": 1e-4,
        "bitfit": 3e-4,
        "ia3": 5e-4,
    },
}

RESULTS_JSON = os.path.join(OUTPUT_DIR, "trained_checkpoint_inference_results.json")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "trained_checkpoint_inference_results.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "trained_checkpoint_inference_summary.json")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "trained_checkpoint_inference_summary.csv")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fixed_subset(split, n, seed):
    n = min(int(n), len(split))
    return split.shuffle(seed=int(seed)).select(range(n))


def build_ia3_bert(num_labels):
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
            model_name=MODEL_NAME,
            num_labels=int(num_labels),
            prefix_length=16,
            prompt_length=20,
            dropout=0.1,
        )
    if method == "lora":
        return build_lora_bert(
            model_name=MODEL_NAME,
            num_labels=int(num_labels),
            r=8,
            alpha=16,
            dropout=0.1,
        )
    if method == "adapter":
        return LayerwiseAdapterBertClassifier(
            model_name=MODEL_NAME,
            num_labels=int(num_labels),
            bottleneck_size=64,
            dropout=0.1,
        )
    if method == "bitfit":
        return build_bitfit_bert(
            model_name=MODEL_NAME,
            num_labels=int(num_labels),
        )
    if method == "ia3":
        return build_ia3_bert(num_labels)
    raise ValueError(method)


def checkpoint_path(dataset_name, method):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(
        CHECKPOINT_DIR,
        f"{dataset_name}_{method}_seed{TRAIN_SEED}.pt"
    )


def train_record_path(dataset_name, method):
    return os.path.join(
        OUTPUT_DIR,
        f"train_{dataset_name}_{method}_seed{TRAIN_SEED}.json"
    )


def benchmark_record_path(dataset_name, method):
    return os.path.join(
        OUTPUT_DIR,
        f"benchmark_{dataset_name}_{method}_seed{TRAIN_SEED}.json"
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def save_trained_weights(model, path, method):
    # Full FT requires all weights. For PEFT/custom frozen-backbone methods,
    # save only trainable tensors and reconstruct the unchanged pretrained base
    # before loading them in a fresh process.
    if method == "full_ft":
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        mode = "full_state_dict"
    else:
        trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
        full_state = model.state_dict()
        state = {
            k: v.detach().cpu()
            for k, v in full_state.items()
            if k in trainable_names
        }
        mode = "trainable_state_dict_on_pretrained_base"

    torch.save(
        {
            "method": method,
            "save_mode": mode,
            "state_dict": state,
        },
        path,
    )
    return mode, len(state)


def train_one(dataset_name, method):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(TRAIN_SEED)

    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )
    eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]
    train_ds = fixed_subset(dataset["train"], TRAIN_SIZE, DATA_SEED)
    eval_ds = fixed_subset(eval_source, EVAL_SIZE, DATA_SEED)

    set_seed(TRAIN_SEED)
    model = build_model(method, num_labels)
    params = count_parameters(model)
    lr = LEARNING_RATES[dataset_name][method]

    run_dir = os.path.join(
        OUTPUT_DIR, f"trainer_{dataset_name}_{method}_seed{TRAIN_SEED}"
    )

    args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
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
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    started = time.time()
    trainer.train()
    train_seconds = time.time() - started

    eval_metrics = trainer.evaluate()
    trained_accuracy = float(eval_metrics["eval_accuracy"])

    ckpt = checkpoint_path(dataset_name, method)
    save_mode, saved_tensor_count = save_trained_weights(model, ckpt, method)

    row = {
        "experiment": EXPERIMENT,
        "stage": "training_and_checkpoint_save",
        "dataset": dataset_name,
        "method": method,
        "seed": TRAIN_SEED,
        "data_seed": DATA_SEED,
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "learning_rate": lr,
        "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH,
        "eval_batch_size": EVAL_BATCH,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "trained_eval_accuracy": trained_accuracy,
        "train_time_seconds": train_seconds,
        "checkpoint_path": ckpt,
        "checkpoint_save_mode": save_mode,
        "checkpoint_saved_tensor_count": saved_tensor_count,
        **params,
        "status": "success",
    }

    with open(train_record_path(dataset_name, method), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    del trainer, model, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


def prepare_inputs(batch, device):
    keep = {"input_ids", "attention_mask", "token_type_ids"}
    return {
        k: v.to(device, non_blocking=True)
        for k, v in batch.items()
        if k in keep
    }


def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def autocast_context(device):
    if device.type == "cuda" and USE_FP16 and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def evaluate_loaded_accuracy(model, loader, device):
    preds, labels = [], []
    model.eval()

    with torch.inference_mode():
        for batch in loader:
            y = batch["labels"].cpu().numpy()
            inputs = prepare_inputs(batch, device)
            with autocast_context(device):
                out = model(**inputs)
            logits = out.logits if hasattr(out, "logits") else out[0]
            p = torch.argmax(logits, dim=-1).cpu().numpy()
            labels.extend(y.tolist())
            preds.extend(p.tolist())

    return float(accuracy_score(labels, preds))


def warmup(model, loader, device):
    model.eval()
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if i >= WARMUP_BATCHES:
                break
            inputs = prepare_inputs(batch, device)
            with autocast_context(device):
                _ = model(**inputs)
    sync_if_cuda()


def benchmark(model, loader, device, n_examples):
    warmup(model, loader, device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    repeat_seconds = []

    for _ in range(TIMED_REPEATS):
        sync_if_cuda()
        t0 = time.perf_counter()
        seen = 0

        with torch.inference_mode():
            for batch in loader:
                inputs = prepare_inputs(batch, device)
                with autocast_context(device):
                    _ = model(**inputs)
                seen += int(inputs["input_ids"].shape[0])

        sync_if_cuda()
        elapsed = time.perf_counter() - t0

        if seen != int(n_examples):
            raise RuntimeError(f"Expected {n_examples} examples, saw {seen}")

        repeat_seconds.append(float(elapsed))

    mean_seconds = float(np.mean(repeat_seconds))
    std_seconds = (
        float(np.std(repeat_seconds, ddof=1))
        if len(repeat_seconds) > 1 else 0.0
    )

    return {
        "repeat_seconds": repeat_seconds,
        "mean_pass_seconds": mean_seconds,
        "std_pass_seconds": std_seconds,
        "examples_per_second": float(n_examples / mean_seconds),
        "latency_ms_per_sample": float(mean_seconds * 1000.0 / n_examples),
        "peak_inference_gpu_allocated_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if torch.cuda.is_available() else 0.0
        ),
        "peak_inference_gpu_reserved_mb": (
            float(torch.cuda.max_memory_reserved() / (1024 ** 2))
            if torch.cuda.is_available() else 0.0
        ),
    }


def benchmark_worker(dataset_name, method):
    set_seed(TRAIN_SEED)

    train_info = json.load(
        open(train_record_path(dataset_name, method), "r", encoding="utf-8")
    )

    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )
    eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]
    eval_ds = fixed_subset(eval_source, EVAL_SIZE, DATA_SEED)

    loader = DataLoader(
        eval_ds,
        batch_size=INFERENCE_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=default_data_collator,
    )

    # Fresh model construction in a fresh process.
    set_seed(TRAIN_SEED)
    model = build_model(method, num_labels)

    ckpt = torch.load(
        checkpoint_path(dataset_name, method),
        map_location="cpu",
        weights_only=False,
    )
    state = ckpt["state_dict"]

    if method == "full_ft":
        load_result = model.load_state_dict(state, strict=True)
    else:
        load_result = model.load_state_dict(state, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    reloaded_accuracy = evaluate_loaded_accuracy(model, loader, device)
    trained_accuracy = float(train_info["trained_eval_accuracy"])
    accuracy_match = abs(reloaded_accuracy - trained_accuracy) <= 1e-9

    if not accuracy_match:
        raise RuntimeError(
            f"Reloaded accuracy mismatch for {dataset_name}/{method}: "
            f"trained={trained_accuracy}, reloaded={reloaded_accuracy}. "
            f"This benchmark is rejected rather than reported."
        )

    if torch.cuda.is_available():
        sync_if_cuda()
        model_gpu_mb = float(torch.cuda.memory_allocated() / (1024 ** 2))
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_total_mb = float(props.total_memory / (1024 ** 2))
    else:
        model_gpu_mb = 0.0
        gpu_name = None
        gpu_total_mb = None

    perf = benchmark(model, loader, device, len(eval_ds))

    row = {
        "experiment": EXPERIMENT,
        "benchmark_type": "fresh_process_trained_checkpoint_loaded_inference",
        "dataset": dataset_name,
        "method": method,
        "training_seed": TRAIN_SEED,
        "data_seed": DATA_SEED,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "eval_size": len(eval_ds),
        "batch_size": INFERENCE_BATCH,
        "warmup_batches": WARMUP_BATCHES,
        "timed_repeats": TIMED_REPEATS,
        "fp16": bool(USE_FP16 and torch.cuda.is_available()),
        "checkpoint_save_mode": ckpt["save_mode"],
        "checkpoint_loaded": True,
        "missing_keys_count": len(load_result.missing_keys),
        "unexpected_keys_count": len(load_result.unexpected_keys),
        "trained_eval_accuracy": trained_accuracy,
        "reloaded_eval_accuracy": reloaded_accuracy,
        "accuracy_match_after_reload": accuracy_match,
        "model_gpu_allocated_mb_before_benchmark": model_gpu_mb,
        **perf,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_mb,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "status": "success",
    }

    with open(benchmark_record_path(dataset_name, method), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    return row


def benchmark_worker_main(dataset_name, method):
    try:
        row = benchmark_worker(dataset_name, method)
        print("\nBENCHMARK WORKER RESULT")
        print(json.dumps(row, indent=2))
    except Exception as e:
        row = {
            "experiment": EXPERIMENT,
            "dataset": dataset_name,
            "method": method,
            "status": "failed",
            "error": repr(e),
        }
        with open(benchmark_record_path(dataset_name, method), "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        print(json.dumps(row, indent=2))
        raise SystemExit(1)


def collect_results():
    rows = []
    for ds in DATASETS:
        for method in METHODS:
            p = benchmark_record_path(ds, method)
            if os.path.exists(p):
                try:
                    rows.append(json.load(open(p, "r", encoding="utf-8")))
                except Exception:
                    pass
    return rows


def save_table(rows, path):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_summary(rows):
    ok = [r for r in rows if r.get("status") == "success"]
    summary = []
    for r in ok:
        summary.append({
            "dataset": r["dataset"],
            "method": r["method"],
            "training_seed": r["training_seed"],
            "trained_eval_accuracy": r["trained_eval_accuracy"],
            "reloaded_eval_accuracy": r["reloaded_eval_accuracy"],
            "accuracy_match_after_reload": r["accuracy_match_after_reload"],
            "examples_per_second": r["examples_per_second"],
            "latency_ms_per_sample": r["latency_ms_per_sample"],
            "peak_inference_gpu_allocated_mb": r["peak_inference_gpu_allocated_mb"],
            "mean_pass_seconds": r["mean_pass_seconds"],
            "std_pass_seconds": r["std_pass_seconds"],
            "benchmark_type": r["benchmark_type"],
        })

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_table(summary, SUMMARY_CSV)


def master_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 92)
    print("TRAINED-CHECKPOINT INFERENCE BENCHMARK")
    print("Each model is trained, saved, then reloaded in a fresh Python process before timing.")
    print("=" * 92)

    for ds in DATASETS:
        for method in METHODS:
            bench_path = benchmark_record_path(ds, method)

            if os.path.exists(bench_path):
                try:
                    old = json.load(open(bench_path, "r", encoding="utf-8"))
                    if old.get("status") == "success":
                        print(f"SKIP completed benchmark: {ds}/{method}")
                        continue
                except Exception:
                    pass

            train_path = train_record_path(ds, method)
            ckpt_path = checkpoint_path(ds, method)

            if not (os.path.exists(train_path) and os.path.exists(ckpt_path)):
                print("\n" + "=" * 92)
                print(f"TRAIN + SAVE | {ds} | {method} | seed={TRAIN_SEED}")
                print("=" * 92)
                train_one(ds, method)
            else:
                print(f"Reuse saved trained checkpoint: {ds}/{method}")

            print("\n" + "-" * 92)
            print(f"FRESH-PROCESS RELOAD + BENCHMARK | {ds} | {method}")
            print("-" * 92)

            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--benchmark-worker",
                ds,
                method,
            ]
            completed = subprocess.run(cmd)
            if completed.returncode != 0:
                print(f"WARNING: benchmark failed for {ds}/{method}; continuing.")

    rows = collect_results()
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    save_table(rows, RESULTS_CSV)
    save_summary(rows)

    print("\n" + "=" * 92)
    print("FINAL TRAINED-CHECKPOINT INFERENCE SUMMARY")
    print("=" * 92)
    for r in rows:
        if r.get("status") == "success":
            print(
                f"{r['dataset']:5s} | {r['method']:8s} | "
                f"acc={r['reloaded_eval_accuracy']:.4f} | "
                f"{r['examples_per_second']:.2f} ex/s | "
                f"{r['latency_ms_per_sample']:.2f} ms/sample | "
                f"GPU={r['peak_inference_gpu_allocated_mb']:.1f} MB"
            )
        else:
            print(
                f"{r.get('dataset')} | {r.get('method')} | FAILED | {r.get('error')}"
            )

    print("\nResults:", RESULTS_JSON)
    print("Summary:", SUMMARY_JSON)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--benchmark-worker":
        benchmark_worker_main(sys.argv[2], sys.argv[3])
    else:
        master_main()
