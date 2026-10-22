import os, sys, csv, gc, json, time, random, platform, subprocess
from contextlib import nullcontext

import numpy as np
import torch
import transformers
import peft
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from peft import IA3Config, TaskType, get_peft_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data import load_and_prepare_dataset
from metrics import count_parameters
from models_symmetry import TrueHybridPrefixPromptBertClassifier
from models_peft_symmetry import LayerwiseAdapterBertClassifier, build_lora_bert, build_bitfit_bert

EXPERIMENT = "inference_efficiency_benchmark"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", EXPERIMENT)

DATASETS = ["sst2", "imdb"]
METHODS = ["full_ft", "hybrid", "lora", "adapter", "bitfit", "ia3"]

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
EVAL_SIZE = 500
DATA_SEED = 2026

BATCH_SIZE = 16
WARMUP_BATCHES = 5
TIMED_REPEATS = 5
USE_FP16 = True

RESULTS_JSON = os.path.join(OUTPUT_DIR, "inference_efficiency_results.json")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "inference_efficiency_results.csv")


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
        MODEL_NAME,
        num_labels=int(num_labels),
    )
    config = IA3Config(
        task_type=TaskType.SEQ_CLS,
        target_modules=["key", "value", "output.dense"],
        feedforward_modules=["output.dense"],
        modules_to_save=["classifier"],
    )
    return get_peft_model(model, config)


def build_model(method, num_labels):
    if method == "full_ft":
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=int(num_labels),
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

    raise ValueError(f"Unsupported method: {method}")


def worker_path(dataset_name, method):
    return os.path.join(OUTPUT_DIR, f"worker_{dataset_name}_{method}.json")


def prepare_inputs(batch, device):
    keep = {"input_ids", "attention_mask", "token_type_ids"}
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in keep
    }


def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def autocast_context(device):
    if device.type == "cuda" and USE_FP16 and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def warmup(model, loader, device):
    model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= WARMUP_BATCHES:
                break
            inputs = prepare_inputs(batch, device)
            with autocast_context(device):
                _ = model(**inputs)
    sync_if_cuda()


def benchmark(model, loader, device, num_examples):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    warmup(model, loader, device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    repeat_seconds = []

    for _ in range(TIMED_REPEATS):
        sync_if_cuda()
        started = time.perf_counter()
        seen = 0

        with torch.inference_mode():
            for batch in loader:
                inputs = prepare_inputs(batch, device)
                with autocast_context(device):
                    _ = model(**inputs)
                seen += int(inputs["input_ids"].size(0))

        sync_if_cuda()
        elapsed = time.perf_counter() - started

        if seen != int(num_examples):
            raise RuntimeError(f"Expected {num_examples} examples, saw {seen}")

        repeat_seconds.append(float(elapsed))

    mean_seconds = float(np.mean(repeat_seconds))
    std_seconds = float(np.std(repeat_seconds, ddof=1)) if len(repeat_seconds) > 1 else 0.0

    return {
        "repeat_seconds": repeat_seconds,
        "mean_pass_seconds": mean_seconds,
        "std_pass_seconds": std_seconds,
        "examples_per_second": float(num_examples / mean_seconds),
        "latency_ms_per_sample": float(mean_seconds * 1000.0 / num_examples),
        "peak_inference_gpu_allocated_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if torch.cuda.is_available() else 0.0
        ),
        "peak_inference_gpu_reserved_mb": (
            float(torch.cuda.max_memory_reserved() / (1024 ** 2))
            if torch.cuda.is_available() else 0.0
        ),
    }


def run_worker(dataset_name, method):
    set_seed(DATA_SEED)

    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )

    eval_source = dataset["validation"] if "validation" in dataset else dataset["test"]
    eval_ds = fixed_subset(eval_source, EVAL_SIZE, DATA_SEED)

    loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(method, num_labels)
    params = count_parameters(model)
    model.to(device)
    model.eval()

    gpu_name = None
    gpu_total_mb = None
    model_allocated_mb = 0.0

    if torch.cuda.is_available():
        sync_if_cuda()
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_total_mb = float(props.total_memory / (1024 ** 2))
        model_allocated_mb = float(torch.cuda.memory_allocated() / (1024 ** 2))

    metrics = benchmark(
        model=model,
        loader=loader,
        device=device,
        num_examples=len(eval_ds),
    )

    row = {
        "experiment": EXPERIMENT,
        "benchmark_type": "architecture_level_inference_no_accuracy",
        "dataset": dataset_name,
        "method": method,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "eval_size": len(eval_ds),
        "data_seed": DATA_SEED,
        "batch_size": BATCH_SIZE,
        "warmup_batches": WARMUP_BATCHES,
        "timed_repeats": TIMED_REPEATS,
        "fp16": bool(USE_FP16 and torch.cuda.is_available()),
        **params,
        "model_gpu_allocated_mb_before_benchmark": model_allocated_mb,
        **metrics,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_mb,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "status": "success",
    }

    del model, loader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


def worker_main(dataset_name, method):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = worker_path(dataset_name, method)

    try:
        row = run_worker(dataset_name, method)
    except Exception as exc:
        row = {
            "experiment": EXPERIMENT,
            "dataset": dataset_name,
            "method": method,
            "status": "failed",
            "error": repr(exc),
        }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(row, file, indent=2)

    print("\nWORKER RESULT")
    print(json.dumps(row, indent=2))

    if row.get("status") != "success":
        raise SystemExit(1)


def collect_rows():
    rows = []
    for dataset_name in DATASETS:
        for method in METHODS:
            path = worker_path(dataset_name, method)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as file:
                    rows.append(json.load(file))
    return rows


def save_results(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(RESULTS_JSON, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    if not rows:
        return

    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def master_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 78)
    print("INFERENCE EFFICIENCY BENCHMARK")
    print("Architecture-level timing; no accuracy is computed.")
    print("=" * 78)

    for dataset_name in DATASETS:
        for method in METHODS:
            path = worker_path(dataset_name, method)

            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        old = json.load(file)
                    if old.get("status") == "success":
                        print(f"SKIP completed: {dataset_name} / {method}")
                        continue
                except Exception:
                    pass

            print("\n" + "=" * 78)
            print(f"BENCHMARK | dataset={dataset_name} | method={method}")
            print("=" * 78)

            command = [
                sys.executable,
                os.path.abspath(__file__),
                "--worker",
                dataset_name,
                method,
            ]
            return_code = subprocess.run(command).returncode

            if return_code != 0:
                print(f"FAILED: {dataset_name} / {method}; continuing.")

    rows = collect_rows()
    save_results(rows)

    print("\n" + "=" * 78)
    print("INFERENCE BENCHMARK SUMMARY")
    print("=" * 78)

    successful = [row for row in rows if row.get("status") == "success"]

    for dataset_name in DATASETS:
        print(f"\n{dataset_name.upper()}")
        for row in successful:
            if row["dataset"] != dataset_name:
                continue

            print(
                f"{row['method']:8s} | "
                f"{row['examples_per_second']:.2f} ex/s | "
                f"{row['latency_ms_per_sample']:.2f} ms/sample | "
                f"GPU={row['peak_inference_gpu_allocated_mb']:.1f} MB"
            )

    print(f"\nResults JSON: {RESULTS_JSON}")
    print(f"Results CSV : {RESULTS_CSV}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], sys.argv[3])
    else:
        master_main()
