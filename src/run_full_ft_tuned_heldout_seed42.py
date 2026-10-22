"""Independent held-out confirmation of Full FT LRs selected on training-only validation.

Run from project's src directory (peft_sentiment env):
    python run_full_ft_tuned_heldout_seed42.py

Reads previous LR winners from outputs/full_ft_lr_tuning_independent/full_ft_lr_tuning_best.json.
Trains one fresh model per dataset (seed 42), evaluates 500 examples from the
OFFICIAL validation split (SST-2) or OFFICIAL test split (IMDb), and records results
in a NEW directory. Never uses official held-out examples for model selection.
"""
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from data import load_and_prepare_dataset

TUNING_DIR = os.path.join(ROOT, 'outputs', 'full_ft_lr_tuning_independent')
BEST_PATH = os.path.join(TUNING_DIR, 'full_ft_lr_tuning_best.json')
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'full_ft_tuned_heldout_seed42')
DATASETS = ('sst2', 'imdb')
TRAIN_SEED = 42
DATA_SEED = 2026
TRAIN_SIZE = 3000
HELDOUT_SIZE = 500
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128
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


def metrics_fn(prediction):
    logits, labels = prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_weighted': f1_score(labels, preds, average='weighted'),
    }


def selected_lrs():
    if not os.path.isfile(BEST_PATH):
        raise FileNotFoundError('Missing tuning winners: ' + BEST_PATH)
    with open(BEST_PATH, encoding='utf-8') as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError('Tuning winners JSON must contain a list')
    selected = {}
    for dataset in DATASETS:
        matches = [r for r in rows if r.get('dataset') == dataset and r.get('method') == 'full'
                   and r.get('status') == 'success']
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one successful Full FT winner for {dataset}; got {len(matches)}')
        row = matches[0]
        if row.get('official_eval_used') is not False:
            raise ValueError(f'Tuning protocol must exclude official evaluation: {dataset}')
        if (int(row.get('train_size', -1)) != TRAIN_SIZE
                or int(row.get('validation_size', -1)) != 500
                or int(row.get('seed', -1)) != TRAIN_SEED
                or int(row.get('data_seed', -1)) != DATA_SEED):
            raise ValueError(f'Tuning protocol mismatch: {dataset}')
        lr = float(row['learning_rate'])
        if lr not in (1e-5, 2e-5, 3e-5, 5e-5):
            raise ValueError(f'Unexpected selected LR for {dataset}: {lr}')
        selected[dataset] = lr
    return selected


def worker_result_path(dataset):
    return os.path.join(OUTPUT_DIR, f'worker_{dataset}_seed{TRAIN_SEED}.json')


def run_one(dataset_name, lr):
    set_seed(TRAIN_SEED)
    dataset, tokenizer, num_labels = load_and_prepare_dataset(
        dataset_name=dataset_name, model_name=MODEL_NAME, max_length=MAX_LENGTH
    )
    train_split = dataset['train'].shuffle(seed=DATA_SEED)
    if len(train_split) < TRAIN_SIZE + 500:
        raise ValueError('Official training split is too short to reproduce tuning split')
    train_ds = train_split.select(range(TRAIN_SIZE))
    eval_key = 'validation' if dataset_name == 'sst2' else 'test'
    if eval_key not in dataset:
        raise ValueError(f'Missing official {eval_key} split in {dataset_name}')
    if len(dataset[eval_key]) < HELDOUT_SIZE:
        raise ValueError('Official held-out split too short')
    heldout_ds = dataset[eval_key].shuffle(seed=DATA_SEED).select(range(HELDOUT_SIZE))

    set_seed(TRAIN_SEED)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=int(num_labels), attn_implementation='eager'
    )
    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f'{dataset_name}_seed{TRAIN_SEED}'),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=lr,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type='linear',
        optim='adamw_torch',
        logging_steps=50,
        eval_strategy='no',
        save_strategy='no',
        report_to='none',
        fp16=torch.cuda.is_available(),
        seed=TRAIN_SEED,
        data_seed=TRAIN_SEED,
        save_safetensors=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=heldout_ds, processing_class=tokenizer,
                      compute_metrics=metrics_fn)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    train_output = trainer.train()
    seconds = time.time() - started
    heldout_metrics = trainer.evaluate()
    result = {
        'status': 'success',
        'experiment': 'full_ft_tuned_heldout_seed42',
        'dataset': dataset_name,
        'method': 'full',
        'selected_learning_rate': lr,
        'selection_source': os.path.relpath(BEST_PATH, ROOT),
        'selection_criterion': 'validation_macro_f1',
        'selection_seed': TRAIN_SEED,
        'training_seed': TRAIN_SEED,
        'data_seed': DATA_SEED,
        'official_eval_used_for_tuning': False,
        'heldout_source': f'official_{eval_key}',
        'train_size': len(train_ds),
        'heldout_size': len(heldout_ds),
        'epochs': EPOCHS,
        'model': MODEL_NAME,
        'max_length': MAX_LENGTH,
        'train_batch_size': TRAIN_BATCH,
        'eval_batch_size': EVAL_BATCH,
        'train_time_seconds': seconds,
        'train_loss': float(train_output.training_loss),
        'peak_train_gpu_allocated_mb': (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            if torch.cuda.is_available() else None
        ),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'transformers_version': transformers.__version__,
    }
    result.update({k: float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v
                   for k, v in heldout_metrics.items()})
    return result


def worker(dataset_name, lr):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        result = run_one(dataset_name, lr)
    except Exception as e:
        result = {'status': 'failed', 'dataset': dataset_name,
                  'selected_learning_rate': lr, 'error': repr(e)}
    with open(worker_result_path(dataset_name), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print('\nWORKER RESULT\n' + json.dumps(result, indent=2), flush=True)
    if result['status'] != 'success':
        raise SystemExit(1)


def main():
    winners = selected_lrs()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('FULL FT | TUNED LEARNING-RATE HELD-OUT CONFIRMATION (SEED 42)', flush=True)
    print('LRs loaded from validation-only tuning. Official splits used ONLY for final evaluation.', flush=True)
    for dataset, lr in winners.items():
        print(f'Selected {dataset}: {lr}', flush=True)
    for dataset in DATASETS:
        lr = winners[dataset]
        path = worker_result_path(dataset)
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                old = json.load(f)
            if old.get('status') == 'success' and float(old.get('selected_learning_rate', -1)) == lr:
                print(f'SKIP already completed {dataset} (lr={lr})', flush=True)
                continue
        print(f'RUN {dataset} | seed=42 | selected LR={lr}', flush=True)
        completed = subprocess.run([sys.executable, os.path.abspath(__file__), '--worker', dataset, str(lr)])
        if completed.returncode:
            print(f'ERROR: {dataset} failed; see {path}', flush=True)
            break
    rows = []
    for dataset in DATASETS:
        path = worker_result_path(dataset)
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                rows.append(json.load(f))
    out = os.path.join(OUTPUT_DIR, 'full_ft_tuned_heldout_seed42_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print('\nHELD-OUT RESULTS:', flush=True)
    for r in rows:
        if r.get('status') == 'success':
            print(f"{r['dataset']}: LR={r['selected_learning_rate']:.0e}, "
                  f"accuracy={r['eval_accuracy']:.4f}, macroF1={r['eval_f1_macro']:.4f}", flush=True)
    print('Results file:', out, flush=True)
    if len(rows) != len(DATASETS) or any(r.get('status') != 'success' for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        worker(sys.argv[2], float(sys.argv[3]))
    else:
        main()
