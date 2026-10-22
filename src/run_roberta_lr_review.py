"""Reviewer-requested RoBERTa Hybrid learning-rate check.
Place in src/ beside run_roberta_backbone_sensitivity.py and data.py.
Run: python run_roberta_lr_review.py
Never uses official held-out validation/test split for LR selection.
"""
import os, json, random, time, gc
from pathlib import Path
import numpy as np
import torch
from transformers import Trainer, TrainingArguments
from data import load_and_prepare_dataset
from run_roberta_backbone_sensitivity import (
    HybridPrefixPromptRobertaClassifier, compute_metrics,
    MODEL_NAME, MAX_LENGTH, TRAIN_SIZE, EVAL_SIZE,
    DATA_SEED, PREFIX_LENGTH, PROMPT_LENGTH, TRAIN_BATCH,
    EVAL_BATCH, GRAD_ACCUM, EPOCHS, WEIGHT_DECAY, WARMUP_RATIO,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'roberta_lr_review'
OUT.mkdir(parents=True, exist_ok=True)
RATES = (1e-4, 3e-4, 5e-4, 1e-3)
SEED = 42
VAL_SIZE = 500

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def fit(dataset, tokenizer, num_labels, train, evaluate, lr, tag):
    seed_all(SEED)
    model = HybridPrefixPromptRobertaClassifier(MODEL_NAME, num_labels,
                 prefix_length=PREFIX_LENGTH, prompt_length=PROMPT_LENGTH)
    args = TrainingArguments(
        output_dir=str(OUT / tag), num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=float(lr), weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO, lr_scheduler_type='linear',
        optim='adamw_torch', logging_steps=50, eval_strategy='epoch',
        save_strategy='no', report_to='none', remove_unused_columns=False,
        fp16=torch.cuda.is_available(), seed=SEED, data_seed=DATA_SEED,
        dataloader_pin_memory=False, eval_accumulation_steps=1,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train,
                      eval_dataset=evaluate, processing_class=tokenizer,
                      compute_metrics=compute_metrics)
    start = time.time()
    trainer.train()
    metrics = trainer.evaluate()
    result = {'learning_rate':lr, 'accuracy':float(metrics['eval_accuracy']),
              'macro_f1':float(metrics['eval_f1_macro']),
              'train_seconds':round(time.time()-start,2)}
    del trainer, model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def main():
    for name in ('sst2','imdb'):
        output = OUT / f'{name}_review_results.json'
        if output.exists():
            record = json.loads(output.read_text(encoding='utf-8'))
        else:
            record = {'dataset':name, 'seed':SEED, 'train_size':TRAIN_SIZE,
                      'validation_size':VAL_SIZE, 'heldout_size':EVAL_SIZE,
                      'selection_metric':'validation_macro_f1', 'trials':[]}
        dataset, tokenizer, labels = load_and_prepare_dataset(name, MODEL_NAME, MAX_LENGTH)
        shuffled = dataset['train'].shuffle(seed=DATA_SEED)
        assert len(shuffled) >= TRAIN_SIZE + VAL_SIZE
        train = shuffled.select(range(TRAIN_SIZE))
        validation = shuffled.select(range(TRAIN_SIZE, TRAIN_SIZE+VAL_SIZE))
        # Original held-out selection, never used to choose learning rate.
        heldout_source = dataset['validation'] if name == 'sst2' else dataset['test']
        heldout = heldout_source.shuffle(seed=DATA_SEED).select(range(EVAL_SIZE))
        for lr in RATES:
            if any(abs(t['learning_rate']-lr)<1e-12 for t in record['trials']):
                print(f'SKIP {name} LR={lr}', flush=True); continue
            print(f'TUNING {name} LR={lr}', flush=True)
            result = fit(dataset, tokenizer, labels, train, validation, lr,
                         f'{name}_tune_{lr:.0e}')
            record['trials'].append(result)
            save_json(output, record)
        best = sorted(record['trials'], key=lambda x:(-x['macro_f1'], -x['accuracy'], x['learning_rate']))[0]
        record['selected_lr'] = best['learning_rate']
        if 'heldout_evaluation' not in record:
            print(f'FINAL {name} selected LR={best["learning_rate"]}', flush=True)
            record['heldout_evaluation'] = fit(dataset, tokenizer, labels, train, heldout,
                                              best['learning_rate'], f'{name}_final')
        save_json(output, record)
        print(json.dumps(record, indent=2), flush=True)
    print('DONE. Upload both *_review_results.json files from', OUT)

if __name__ == '__main__':
    main()
