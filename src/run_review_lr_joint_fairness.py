"""Symmetry 4631037: Reviewer 3 wider LR and fair Hybrid tuning.

Compatibility target: user's existing bert-base-uncased project, Transformers 4.57.6.
Only the phases 'screen', 'full-joint', and 'heldout' train models.
'plan', 'select', 'report' work without torch/transformers imports.

Protocol:
- SST-2 and IMDb official training split: shuffled with data seed 2026.
- 3,000 fit samples, disjoint 500 validation samples for configuration selection.
- Official validation (SST-2) / test (IMDb) sampled, 500 per dataset for final evaluation ONLY.
- Selection by final-epoch validation Macro-F1, accuracy tie-break.
- Matched search: six LRs x three length configurations for EACH of three methods,
  giving 18 selection trials per method per dataset.
- Optional full-joint Hybrid: all 3x3 length combinations x six LRs. This has a
  larger search budget and MUST be reported separately, not as a fair RQ2 comparator.
- Five held-out training seeds per selected method/dataset.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent.parent
OUT = ROOT / 'outputs' / 'reviewer3_wide_lr_matched_hybrid'
BACKBONE = 'bert-base-uncased'
DATA_SEED = 2026
TRAIN_SIZE = 3000
VAL_SIZE = 500
TEST_SIZE = 500
SELECTION_SEED = 42
FINAL_SEEDS = (42, 123, 2026, 3407, 7777)
LR_GRID = (1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 1e-2)
PFX = (8, 16, 32)
PRM = (10, 20, 30)
PAIRED_LENGTHS = ((8, 10), (16, 20), (32, 30))
DATASETS = ('sst2', 'imdb')
METHODS = ('prefix_only', 'prompt_only', 'hybrid')


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def get_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def lr_label(lr):
    return f'{lr:.0e}' .replace('+', '')


def config_id(cfg):
    return f"{cfg['dataset']}_{cfg['method']}_pfx{cfg['prefix_length']}_prm{cfg['prompt_length']}_lr{lr_label(cfg['learning_rate'])}"


def configurations(dataset, scope):
    if scope == 'matched':
        for method in METHODS:
            if method == 'prefix_only':
                combos = [(p, 0) for p in PFX]
            elif method == 'prompt_only':
                combos = [(0, m) for m in PRM]
            else:
                combos = list(PAIRED_LENGTHS)
            for p, m in combos:
                for lr in LR_GRID:
                    yield dict(dataset=dataset, method=method, prefix_length=p, prompt_length=m, learning_rate=lr)
    elif scope == 'joint':
        for p in PFX:
            for m in PRM:
                if (p, m) in PAIRED_LENGTHS:
                    continue  # These 18 candidates per dataset are already in the matched phase.
                for lr in LR_GRID:
                    yield dict(dataset=dataset, method='hybrid', prefix_length=p, prompt_length=m, learning_rate=lr)
    elif scope == 'joint_all':
        for p in PFX:
            for m in PRM:
                for lr in LR_GRID:
                    yield dict(dataset=dataset, method='hybrid', prefix_length=p, prompt_length=m, learning_rate=lr)
    else:
        raise ValueError(scope)


def path_for(cfg, mode, seed, selection='matched'):
    if mode == 'selection':
        return OUT / 'selection' / (config_id(cfg) + '_seed42.json')
    return OUT / 'heldout' / selection / (config_id(cfg) + f'_seed{seed}.json')


def select_winner(dataset, method, selection='matched', strict=True, allow_failed=False):
    if selection == 'matched':
        cand = [x for x in configurations(dataset, 'matched') if x['method'] == method]
    elif selection == 'joint' and method == 'hybrid':
        cand = list(configurations(dataset, 'joint_all'))
    else:
        raise ValueError('joint selection applies only to Hybrid. Use matched for prefix/prompt.')
    good, missing, failed = [], [], []
    for cfg in cand:
        row = get_json(path_for(cfg, 'selection', SELECTION_SEED))
        if row is None:
            missing.append(config_id(cfg))
        elif row.get('status') != 'success':
            failed.append(config_id(cfg))
        elif not math.isfinite(float(row.get('eval_f1_macro', float('nan')))):
            failed.append(config_id(cfg))
        else:
            good.append(row)
    if strict and (missing or (failed and not allow_failed)):
        raise RuntimeError(f'{dataset}/{method}/{selection}: incomplete grid. Missing={len(missing)} failed={len(failed)}; no selection made. Consult grid_status.json. An explicit --allow-failed-grid permits attempted-but-diverged candidates, NOT missing trials.')
    if not good:
        raise RuntimeError(f'{dataset}/{method}/{selection}: zero valid candidates')
    best = max(good, key=lambda r: (float(r['eval_f1_macro']), float(r['eval_accuracy']), -float(r['learning_rate'])))
    return {'dataset':dataset, 'method':method, 'selection':selection,
            'expected_candidates':len(cand), 'successful_candidates':len(good),
            'missing_candidates':missing,'failed_candidates':failed,
            'selected_config':{k:best[k] for k in ('dataset','method','prefix_length','prompt_length','learning_rate')},
            'validation_macro_f1':best['eval_f1_macro'], 'validation_accuracy':best['eval_accuracy'],
            'selection_seed':SELECTION_SEED,'data_seed':DATA_SEED,
            'complete':not (missing or failed), 'documented_failed_candidate_override':bool(failed and allow_failed)}


def choose(dataset_filter, selection, allow_failed=False):
    records = []
    for ds in dataset_filter:
        for method in METHODS:
            mode = selection if method == 'hybrid' else 'matched'
            records.append(select_winner(ds, method, selection=mode, strict=True, allow_failed=allow_failed))
    save_json(OUT / f'selected_{selection}.json', records)
    return records


def plan(dataset_filter):
    m = [x for d in dataset_filter for x in configurations(d,'matched')]
    j = [x for d in dataset_filter for x in configurations(d,'joint')]
    p = {'datasets':dataset_filter,'backbone':BACKBONE,'fit':TRAIN_SIZE,'train_only_validation':VAL_SIZE,
         'heldout_evaluation':TEST_SIZE, 'data_seed':DATA_SEED,'screen_seed':SELECTION_SEED,
         'training_epochs':3,'candidate_learning_rates':LR_GRID,
         'matched_trial_count':len(m), 'optional_joint_additional_trials':len(j),
         'heldout_matched_trial_count':len(dataset_filter)*len(METHODS)*len(FINAL_SEEDS),
         'optional_joint_heldout_count':len(dataset_filter)*len(FINAL_SEEDS),
         'matched_hybrid_prefix_prompt_pairs':PAIRED_LENGTHS,
         'fairness_note':'matched tuning compares 18 configurations for each method; optional full joint Hybrid 54 configurations uses a larger budget and is exploratory',
         'all_screen_configs':[config_id(x) for x in m],
         'all_optional_joint_configs':[config_id(x) for x in j]}
    save_json(OUT/'protocol_and_plan.json',p)
    print(json.dumps({k:v for k,v in p.items() if not k.startswith('all_')}, indent=2))
    return p


def run_training(cfg, mode, seed, selection):
    import random
    import platform
    import numpy as np
    import torch
    import transformers
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import Trainer, TrainingArguments
    from data import load_and_prepare_dataset
    from metrics import count_parameters
    from models_ablation_symmetry import PrefixOnlyBertClassifier, PromptOnlyBertClassifier
    from models_symmetry import TrueHybridPrefixPromptBertClassifier

    if torch.cuda.is_available() is False and os.environ.get('REVIEW_ALLOW_CPU', '0') != '1':
        raise RuntimeError('CUDA unavailable. This training is designed for your 4GB GPU; set REVIEW_ALLOW_CPU=1 only if intentional.')

    def set_seed(x):
        random.seed(x); np.random.seed(x); torch.manual_seed(x)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

    set_seed(seed)
    ds, tokenizer, num_labels = load_and_prepare_dataset(cfg['dataset'], BACKBONE, 128)
    shuffled = ds['train'].shuffle(seed=DATA_SEED)
    if len(shuffled) < TRAIN_SIZE+VAL_SIZE:
        raise ValueError('Official training split too small')
    train_ds = shuffled.select(range(TRAIN_SIZE))
    if mode=='selection':
        ev_ds = shuffled.select(range(TRAIN_SIZE,TRAIN_SIZE+VAL_SIZE))
        split_label='training-derived validation'
    else:
        official_split='validation' if cfg['dataset']=='sst2' else 'test'
        ev_ds=ds[official_split].shuffle(seed=DATA_SEED).select(range(TEST_SIZE))
        split_label='held-out official '+official_split
    set_seed(seed)
    if cfg['method']=='prefix_only':
        model=PrefixOnlyBertClassifier(model_name=BACKBONE, num_labels=num_labels, prefix_length=cfg['prefix_length'], dropout=0.1)
    elif cfg['method']=='prompt_only':
        model=PromptOnlyBertClassifier(model_name=BACKBONE, num_labels=num_labels, prompt_length=cfg['prompt_length'], dropout=0.1)
    else:
        model=TrueHybridPrefixPromptBertClassifier(model_name=BACKBONE,num_labels=num_labels,prefix_length=cfg['prefix_length'],prompt_length=cfg['prompt_length'],dropout=0.1)
    param_counts=count_parameters(model)
    run_dir=OUT/'tmp_trainer'/mode/config_id(cfg)/('seed'+str(seed))
    def metrics(pred):
        logits,labels=pred
        if isinstance(logits,tuple): logits=logits[0]
        y=np.argmax(logits,axis=-1)
        return {'accuracy':float(accuracy_score(labels,y)),
                'f1_macro':float(f1_score(labels,y,average='macro')),
                'f1_weighted':float(f1_score(labels,y,average='weighted'))}
    args=TrainingArguments(output_dir=str(run_dir),num_train_epochs=3,
        per_device_train_batch_size=8,per_device_eval_batch_size=16,gradient_accumulation_steps=1,
        learning_rate=float(cfg['learning_rate']),weight_decay=0.01,warmup_ratio=0.1,
        lr_scheduler_type='linear',optim='adamw_torch',logging_steps=50,eval_strategy='epoch',
        save_strategy='no',report_to='none',remove_unused_columns=False,
        fp16=bool(torch.cuda.is_available()),seed=seed,data_seed=seed,save_safetensors=False)
    tr=Trainer(model=model,args=args,train_dataset=train_ds,eval_dataset=ev_ds,
               processing_class=tokenizer,compute_metrics=metrics)
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    before=time.time()
    train_result=tr.train()
    seconds=time.time()-before
    ev=tr.evaluate()
    if not math.isfinite(float(ev['eval_f1_macro'])) or not math.isfinite(float(train_result.training_loss)):
        raise ArithmeticError('Non-finite validation F1 or training loss, check optimization divergence')
    return dict(cfg, status='success', phase=mode, selection=selection, seed=seed,
        data_seed=DATA_SEED, train_size=len(train_ds), evaluation_size=len(ev_ds),evaluation_split=split_label,
        train_fingerprint=getattr(train_ds,'_fingerprint',''), eval_fingerprint=getattr(ev_ds,'_fingerprint',''),
        model_name=BACKBONE,max_length=128,epochs=3,train_batch_size=8,eval_batch_size=16,
        optimizer='adamw_torch', scheduler='linear',weight_decay=0.01,warmup_ratio=0.1,
        fp16=bool(torch.cuda.is_available()),gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        trainable_parameters=param_counts,train_time_seconds=float(seconds),
        train_loss=float(train_result.training_loss),
        peak_allocated_mb=float(torch.cuda.max_memory_allocated()/1024**2) if torch.cuda.is_available() else None,
        python_version=platform.python_version(), torch_version=torch.__version__, transformers_version=transformers.__version__,
        eval_accuracy=float(ev['eval_accuracy']),eval_f1_macro=float(ev['eval_f1_macro']),
        eval_f1_weighted=float(ev['eval_f1_weighted']),
        train_log_history=tr.state.log_history)


def single_worker(args):
    cfg=json.loads(args.cfg)
    seed=int(args.seed)
    file=path_for(cfg,args.mode,seed,args.selection)
    try:
        result=run_training(cfg,args.mode,seed,args.selection)
    except Exception:
        result=dict(cfg, phase=args.mode, selection=args.selection, seed=seed, status='failed',
                    error=traceback.format_exc())
    save_json(file,result)
    print(json.dumps({k:v for k,v in result.items() if k in ('dataset','method','learning_rate','prefix_length','prompt_length','status','eval_accuracy','eval_f1_macro','error')},indent=2))
    return 0 if result['status']=='success' else 1


def master_run(args, mode):
    if mode in ('matched','joint'):
        configs=[cfg for d in args.datasets for cfg in configurations(d,mode)]
        selection='matched'
        work=[(cfg,SELECTION_SEED) for cfg in configs]
        result_mode='selection'
    elif mode=='heldout':
        winners=choose(args.datasets,args.selection,args.allow_failed_grid)
        work=[]
        for w in winners:
            for seed in FINAL_SEEDS:
                work.append((w['selected_config'],seed))
        selection=args.selection
        result_mode='heldout'
    else: raise ValueError(mode)
    completed=0; attempted=0; failures=0
    for cfg,seed in work:
        file=path_for(cfg,result_mode,seed,selection)
        existing=get_json(file)
        if existing is not None and existing.get('status')=='success':
            print('SKIP already successful:',file.name, flush=True)
            completed+=1;continue
        if args.max_runs is not None and attempted>=args.max_runs:
            break
        print(f"TRAIN {result_mode} {len(work)} total: {config_id(cfg)} seed={seed}",flush=True)
        command=[sys.executable,str(SCRIPT),'--phase','worker','--mode',result_mode,'--selection',selection,
                 '--cfg',json.dumps(cfg),'--seed',str(seed)]
        rc=subprocess.run(command).returncode
        attempted+=1
        if rc: failures+=1;print('WARNING failed worker:',file.name,flush=True)
    report(args.datasets)
    print('Completed before this invocation:',completed, 'attempted this invocation:',attempted,'failed:',failures)
    return 0 if failures==0 else 1


def report(dataset_filter):
    grid=[];held=[];audit=[]
    for ds in dataset_filter:
        for method in METHODS:
            for cfg in [x for x in configurations(ds,'matched') if x['method']==method]:
                row=get_json(path_for(cfg,'selection',SELECTION_SEED))
                grid.append({'scope':'matched',**cfg,'status':row.get('status') if row else 'missing',
                    'validation_macro_f1':row.get('eval_f1_macro') if row else None,
                    'validation_accuracy':row.get('eval_accuracy') if row else None})
        for cfg in configurations(ds,'joint'):
            row=get_json(path_for(cfg,'selection',SELECTION_SEED))
            grid.append({'scope':'joint_extra',**cfg,'status':row.get('status') if row else 'missing',
                'validation_macro_f1':row.get('eval_f1_macro') if row else None,
                'validation_accuracy':row.get('eval_accuracy') if row else None})
    for selection in ('matched','joint'):
        s=get_json(OUT/f'selected_{selection}.json') or []
        for win in s:
            cfg=win['selected_config']
            for seed in FINAL_SEEDS:
                row=get_json(path_for(cfg,'heldout',seed,selection))
                held.append({'selection':selection,**cfg,'seed':seed,'status':row.get('status') if row else 'missing',
                    'heldout_accuracy':row.get('eval_accuracy') if row else None,
                    'heldout_macro_f1':row.get('eval_f1_macro') if row else None})
    def csv_write(filename,rows):
        if not rows:return
        keys=list(dict.fromkeys(k for row in rows for k in row.keys()))
        with open(OUT/filename,'w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    csv_write('selection_grid_status.csv',grid)
    csv_write('heldout_status.csv',held)
    summary=[]
    for selection in ('matched','joint'):
        for ds in dataset_filter:
            for method in METHODS:
                vals=[x for x in held if x['selection']==selection and x['dataset']==ds and x['method']==method and x['status']=='success']
                if not vals:continue
                if len(vals)!=len(FINAL_SEEDS): continue
                import statistics
                summary.append(dict(dataset=ds,method=method,selection=selection,
                    n=len(vals),mean_accuracy=statistics.mean(x['heldout_accuracy'] for x in vals),
                    sd_accuracy=statistics.stdev(x['heldout_accuracy'] for x in vals),
                    mean_macro_f1=statistics.mean(x['heldout_macro_f1'] for x in vals),
                    sd_macro_f1=statistics.stdev(x['heldout_macro_f1'] for x in vals)))
    csv_write('five_seed_heldout_summary.csv',summary)
    p={'matched_count':sum(x['scope']=='matched' for x in grid),'matched_success':sum(x['scope']=='matched' and x['status']=='success' for x in grid),
       'additional_joint_count':sum(x['scope']=='joint_extra' for x in grid),'additional_joint_success':sum(x['scope']=='joint_extra' and x['status']=='success' for x in grid),
       'heldout_finished_count':sum(x['status']=='success' for x in held),'five_seed_full_groups':len(summary)}
    save_json(OUT/'grid_status.json',p)
    print(json.dumps(p,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('plan','screen','full-joint','select','heldout','report','worker'))
    parser.add_argument('--dataset',choices=('both','sst2','imdb'),default='both')
    parser.add_argument('--selection',choices=('matched','joint'),default='matched')
    parser.add_argument('--max-runs',type=int,default=None,help='Safely limit newly launched training workers for a small smoke test')
    parser.add_argument('--allow-failed-grid',action='store_true',help='Explicitly allow configurations that were attempted but failed/diverged; never permits unattempted configurations')
    parser.add_argument('--cfg',help=argparse.SUPPRESS)
    parser.add_argument('--mode',choices=('selection','heldout'),help=argparse.SUPPRESS)
    parser.add_argument('--seed',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    args.datasets=DATASETS if args.dataset=='both' else (args.dataset,)
    if args.phase=='worker':return single_worker(args)
    if args.phase=='plan':plan(args.datasets);return 0
    if args.phase=='screen':return master_run(args,'matched')
    if args.phase=='full-joint':return master_run(args,'joint')
    if args.phase=='select':
        print(json.dumps(choose(args.datasets,args.selection,args.allow_failed_grid),indent=2));return 0
    if args.phase=='heldout':return master_run(args,'heldout')
    if args.phase=='report':report(args.datasets);return 0
    return 1


if __name__=='__main__':sys.exit(main())
