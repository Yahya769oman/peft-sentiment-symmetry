"""Paired held-out analysis of saved seed-42 Full FT and Hybrid checkpoints.
No training; no checkpoint changes. Run from the project's src directory.
"""
import os
import csv
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from transformers import default_data_collator

import run_trained_checkpoint_inference_benchmark as base

OUT = Path(base.OUTPUT_DIR) / 'paired_statistics_seed42'
N_BOOT = 10000
BOOT_SEED = 20261021


def get_predictions(ds_name, method, eval_ds, num_labels):
    path = base.checkpoint_path(ds_name, method)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing saved weights: {path}')
    base.set_seed(base.TRAIN_SEED)
    model = base.build_model(method, num_labels)
    # Checkpoints were produced by the local project, not untrusted downloads.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('method') != method:
        raise ValueError(f'Checkpoint method mismatch: {path}')
    state = checkpoint['state_dict']
    loaded = model.load_state_dict(state, strict=(method == 'full_ft'))
    if loaded.unexpected_keys:
        raise ValueError(f'Unexpected keys in {path}: {loaded.unexpected_keys[:10]}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    loader = DataLoader(eval_ds, batch_size=base.INFERENCE_BATCH, shuffle=False,
                        num_workers=0, collate_fn=default_data_collator)
    labels, preds = [], []
    with torch.inference_mode():
        for batch in loader:
            labels.extend(batch['labels'].cpu().tolist())
            inputs = base.prepare_inputs(batch, device)
            with base.autocast_context(device):
                result = model(**inputs)
            logits = result.logits if hasattr(result, 'logits') else result[0]
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(labels), np.asarray(preds), len(loaded.missing_keys)


def run_one(ds_name):
    dataset, _, num_labels = base.load_and_prepare_dataset(
        ds_name, base.MODEL_NAME, base.MAX_LENGTH)
    source = dataset['validation'] if 'validation' in dataset else dataset['test']
    eval_ds = base.fixed_subset(source, base.EVAL_SIZE, base.DATA_SEED)
    y, full, missing_full = get_predictions(ds_name, 'full_ft', eval_ds, num_labels)
    y2, hybrid, missing_hybrid = get_predictions(ds_name, 'hybrid', eval_ds, num_labels)
    if not np.array_equal(y, y2):
        raise RuntimeError('Evaluation label order mismatch')
    n = len(y)
    full_ok, hybrid_ok = full == y, hybrid == y
    full_acc = float(full_ok.mean())
    hybrid_acc = float(hybrid_ok.mean())
    # Check the reloaded checkpoint accuracy against the original saved record.
    for method, accuracy in [('full_ft', full_acc), ('hybrid', hybrid_acc)]:
        record_path = base.train_record_path(ds_name, method)
        if not os.path.isfile(record_path):
            raise FileNotFoundError(f'Missing training record: {record_path}')
        with open(record_path, encoding='utf-8') as f:
            expected = float(json.load(f)['trained_eval_accuracy'])
        if abs(accuracy - expected) > 1e-9:
            raise RuntimeError(f'{ds_name}/{method}: accuracy {accuracy} != saved {expected}; abort')
    full_only = int(np.sum(full_ok & ~hybrid_ok))
    hybrid_only = int(np.sum(~full_ok & hybrid_ok))
    discordant = full_only + hybrid_only
    pvalue = float(binomtest(min(full_only, hybrid_only), discordant, p=0.5).pvalue) if discordant else 1.0
    rng = np.random.default_rng(BOOT_SEED)
    diff = full_ok.astype(float) - hybrid_ok.astype(float)
    # Paired bootstrap: resample example indices, retaining both model results together.
    bootstrap = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        ix = rng.integers(0, n, size=n)
        bootstrap[i] = diff[ix].mean()
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f'{ds_name}_paired_predictions.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eval_subset_index', 'true_label', 'full_ft_prediction', 'hybrid_prediction',
                         'full_ft_correct', 'hybrid_correct'])
        for i in range(n):
            writer.writerow([i, int(y[i]), int(full[i]), int(hybrid[i]), int(full_ok[i]), int(hybrid_ok[i])])
    result = {
        'dataset': ds_name, 'training_seed': base.TRAIN_SEED,
        'data_seed': base.DATA_SEED, 'n_examples': n, 'max_length': base.MAX_LENGTH,
        'full_ft_accuracy': full_acc, 'hybrid_accuracy': hybrid_acc,
        'accuracy_difference_full_minus_hybrid': full_acc - hybrid_acc,
        'paired_bootstrap_95pct_CI_full_minus_hybrid': [float(low), float(high)],
        'bootstrap_resamples': N_BOOT, 'bootstrap_rng_seed': BOOT_SEED,
        'mcnemar_exact_two_sided_p': pvalue,
        'full_correct_hybrid_wrong': full_only,
        'full_wrong_hybrid_correct': hybrid_only,
        'discordant_total': discordant,
        'missing_keys_full_ft': missing_full,
        'missing_keys_hybrid_expected_frozen_base': missing_hybrid,
        'checkpoint_accuracy_matches_training_record': True,
        'scope': 'single-seed 42, 500-example held-out subset; not five-seed inference',
    }
    with open(OUT / f'{ds_name}_paired_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"{ds_name}: Full FT={full_acc:.4f}, Hybrid={hybrid_acc:.4f}, "
          f"difference={full_acc-hybrid_acc:+.4f}, 95% CI=[{low:+.4f}, {high:+.4f}], "
          f"McNemar exact p={pvalue:.6g}", flush=True)
    return result


if __name__ == '__main__':
    print('SAVED CHECKPOINT PAIRED STATISTICS — INFERENCE ONLY', flush=True)
    print('Output:', OUT, flush=True)
    if base.TRAIN_SEED != 42 or base.DATA_SEED != 2026 or base.EVAL_SIZE != 500:
        raise RuntimeError('Original benchmark constants changed; refusing unverified comparison')
    results = [run_one(name) for name in ('sst2', 'imdb')]
    with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('SUCCESS. Results saved in:', OUT, flush=True)
