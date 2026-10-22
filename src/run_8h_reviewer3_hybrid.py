#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Symmetry-4631037: time-bounded fair-candidate-count Hybrid experiment.

Put this script in the EXISTING PROJECT/src/ folder (beside
run_review_lr_joint_fairness.py, models_symmetry.py, and data.py).
Run --phase plan first, then --phase all --budget-hours 6.5 --seeds 3.

Reuses 9 *completed* training-only validation trials for EACH ablation
method/dataset, and runs the 9 predeclared Hybrid candidates/dataset.
The total selection budget is 9 candidates per method/dataset. This is
NOT a full 3x3x3 joint Hybrid grid. Test-set evaluation ONLY happens after
all 9 validation trials/dataset are completed, with an independently
trained selected Hybrid and matched baseline seeds.

Never invents results or silently selects from incomplete validation grids.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve()
OUT = ROOT / 'outputs' / 'reviewer3_8h_fair_hybrid'
BASE = ROOT / 'outputs' / 'exp3_ablation_independent_tuning'
SELECTION_INPUT = BASE / 'ablation_tuning_results.json'
OFFICIAL_INPUT = BASE / 'ablation_tuned_official_results.json'
DATASETS = ('imdb', 'sst2')  # Prioritize IMDb: largest baseline Hybrid accuracy deficit.
SEEDS = (42, 123, 2026, 3407, 7777)
LRS = (1e-4, 3e-4, 5e-4)
# Predeclared three candidate pairs: 3 lengths x 3 LR = 9 for Hybrid.
PAIRS = ((8, 10), (16, 20), (32, 30))


def load(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return None


def _serializable(value):
    """Keep all finite metrics untouched; preserve non-finite diagnostic values
    explicitly as strings, not invalid JSON numbers. Intended for log history.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {k: _serializable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [_serializable(x) for x in value]
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    # Do not invent/fix a failed scientific metric; the training function already
    # requires finite train_loss and final eval_macro_f1 before returning success.
    tmp.write_text(json.dumps(_serializable(value), indent=2, ensure_ascii=False,
                              allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def atomic_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def name(cfg):
    return f"{cfg['dataset']}_hybrid_pfx{cfg['prefix_length']}_prm{cfg['prompt_length']}_lr{cfg['learning_rate']:.0e}".replace('+', '')


def target_path(cfg, phase, seed=42):
    return OUT / phase / (name(cfg) + '_seed' + str(seed) + '.json')


def grid(dataset):
    return [dict(dataset=dataset, method='hybrid', prefix_length=p, prompt_length=m, learning_rate=lr)
            for p, m in PAIRS for lr in LRS]


def check_old_row(row, dataset, method, phase):
    """Hard precondition before using saved evidence, not a retroactive refit."""
    assert row.get('status') == 'success'
    assert row.get('dataset') == dataset and row.get('method') == method
    assert row.get('model_name') == 'bert-base-uncased'
    assert row.get('data_seed') == 2026 and row.get('seed') == 42
    assert row.get('train_size') == 3000 and row.get('max_length') == 128
    assert row.get('epochs') == 3 and row.get('train_batch_size') == 8
    assert row.get('eval_batch_size') == 16
    if phase == 'selection':
        assert row.get('validation_size') == 500
        assert row.get('official_eval_used') is False
    else:
        assert row.get('eval_size') == 500
    assert row.get('learning_rate') in LRS
    assert math.isfinite(float(row['eval_f1_macro']))
    assert math.isfinite(float(row['eval_accuracy']))


def ablation_candidates(dataset, method):
    raw = load(SELECTION_INPUT)
    if not isinstance(raw, list):
        raise RuntimeError('Missing ORIGINAL ablation validation results: ' + str(SELECTION_INPUT))
    rows = [r for r in raw if r.get('dataset') == dataset and r.get('method') == method]
    if len(rows) != 9 or len({(r['learning_rate'], r['prefix_length'], r['prompt_length']) for r in rows}) != 9:
        raise RuntimeError(f'{dataset}/{method}: expected exactly 9 UNIQUE archived validation candidates, got {len(rows)}')
    for row in rows:
        check_old_row(row, dataset, method, 'selection')
    return rows


def hybrid_candidates(dataset):
    rows, missing, failed = [], [], []
    for cfg in grid(dataset):
        file = target_path(cfg, 'selection')
        row = load(file)
        if row is None:
            missing.append(name(cfg))
        elif row.get('status') != 'success':
            failed.append(name(cfg))
        else:
            try:
                assert row['seed'] == 42 and row['phase'] == 'selection'
                assert row['evaluation_split'] == 'training-derived validation'
                assert row['train_size'] == 3000 and row['evaluation_size'] == 500
                assert math.isfinite(float(row['eval_f1_macro']))
                rows.append(row)
            except (AssertionError, KeyError, ValueError, TypeError):
                failed.append(name(cfg))
    return rows, missing, failed


def best(rows):
    return max(rows, key=lambda x:(float(x['eval_f1_macro']),float(x['eval_accuracy']),-float(x['learning_rate'])))


def candidate_id(row):
    return f"pfx{row['prefix_length']}/prm{row['prompt_length']}/lr{row['learning_rate']:.0e}"


def do_select(dataset_filter):
    all_selected = load(OUT / 'selected_configs.json') or {}
    for dataset in dataset_filter:
        h_rows, missing, failed = hybrid_candidates(dataset)
        if missing or failed or len(h_rows) != 9:
            print(f'{dataset}: cannot SELECT until all 9 Hybrid validation trials complete. Missing={len(missing)} failed={len(failed)}.')
            continue
        old_prefix = ablation_candidates(dataset, 'prefix_only')
        old_prompt = ablation_candidates(dataset, 'prompt_only')
        winners = {method:best(rows) for method,rows in (
            ('prefix_only',old_prefix),('prompt_only',old_prompt),('hybrid',h_rows))}
        selection = {
            'dataset':dataset,
            'protocol':'9 candidate configurations each for prefix_only, prompt_only, hybrid; training-only validation; seed42; data seed 2026; 3 epochs',
            'method_candidate_counts':{'prefix_only':9,'prompt_only':9,'hybrid':9},
            'hybrid_pairs':PAIRS,
            'learning_rates':LRS,
            'is_full_joint_hybrid_grid':False,
            'selection_rule':'last-epoch validation Macro-F1; validation accuracy tie-break',
            'source_component_validation':str(SELECTION_INPUT.relative_to(ROOT)),
            'selected':{
                method:{
                    'method':method,
                    'prefix_length':int(row['prefix_length']),
                    'prompt_length':int(row['prompt_length']),
                    'learning_rate':float(row['learning_rate']),
                    'validation_macro_f1':float(row['eval_f1_macro']),
                    'validation_accuracy':float(row['eval_accuracy']),
                } for method,row in winners.items()
            },
        }
        if dataset in all_selected and all_selected[dataset]['selected'] != selection['selected']:
            raise RuntimeError(f'{dataset} winner changed from saved selection; manual audit required')
        all_selected[dataset] = selection
        print(dataset, 'SELECTION READY:', json.dumps(selection['selected'],indent=2))
    if all_selected:
        save(OUT / 'selected_configs.json', all_selected)
    return all_selected


def run_worker(cfg, phase, seed):
    path = target_path(cfg, phase, seed)
    try:
        from run_review_lr_joint_fairness import run_training
        row = run_training(cfg, phase, seed, '9_candidate_balanced_selection')
        row['audit_protocol'] = 'reviewer3_8h_fair_hybrid'
        row['candidate_policy'] = 'three paired prefix/prompt lengths x three learning rates'
        row['uses_heldout_for_selection'] = False
    except Exception:
        row = dict(cfg,phase=phase,seed=seed,status='failed',error=traceback.format_exc())
    save(path,row)
    print(json.dumps({k:v for k,v in row.items() if k in ('dataset','method','phase','seed','status','prefix_length','prompt_length','learning_rate','eval_accuracy','eval_f1_macro','error')},indent=2),flush=True)
    return 0 if row['status']=='success' else 1


def all_work(dataset_filter, phase, seeds):
    if phase == 'selection':
        return [(cfg,'selection',42) for dataset in dataset_filter for cfg in grid(dataset)]
    selections = do_select(dataset_filter)
    work=[]
    for dataset in dataset_filter:
        if dataset not in selections:
            print(f'{dataset}: HELD-OUT BLOCKED: incomplete training-only grid.')
            continue
        w=selections[dataset]['selected']['hybrid']
        cfg=dict(dataset=dataset,method='hybrid',prefix_length=w['prefix_length'],prompt_length=w['prompt_length'],learning_rate=w['learning_rate'])
        for seed in SEEDS[:seeds]:
            work.append((cfg,'heldout',seed))
    return work


def execute(dataset_filter, phase, seeds, deadline):
    work=all_work(dataset_filter,phase,seeds)
    launched=0
    completed_time=[]
    for cfg,mode,seed in work:
        path=target_path(cfg,mode,seed)
        old=load(path)
        if old and old.get('status')=='success':
            print('SKIP:',path.name,flush=True)
            continue
        remain=(deadline-time.monotonic()) if deadline else None
        # Use previous observed durations as guard, or conservative 22 min per run.
        # Never launch a job that is unlikely to finish inside budget.
        expected=(sum(completed_time)/len(completed_time)) if completed_time else (22*60)
        guard=max(expected*1.2,18*60)
        if remain is not None and remain < guard:
            print('STOP: time reserved for saving and auditing results. Remaining seconds:',round(remain),flush=True)
            break
        print('RUN',mode, name(cfg),'seed',seed, 'remaining minutes',round(remain/60,1) if remain else 'unlimited',flush=True)
        t0=time.monotonic()
        cmd=[sys.executable,str(SCRIPT),'--phase','worker','--cfg',json.dumps(cfg),'--worker-phase',mode,'--seed',str(seed)]
        rc=subprocess.run(cmd,check=False).returncode
        took=time.monotonic()-t0
        if rc==0:
            completed_time.append(took)
        else:
            failure=load(path) or {}
            message=str(failure.get('error',''))
            print('FAILED run; inspect JSON error:',path,flush=True)
            fatal_tokens=('ModuleNotFoundError', 'ImportError', 'CUDA unavailable',
                          'ConnectionError', 'SSLError', 'No such file or directory',
                          'not a valid model identifier')
            if any(t in message for t in fatal_tokens):
                print('STOP ON INFRASTRUCTURE ERROR — fix environment/data before retry.',flush=True)
                break
        launched+=1
        report(dataset_filter,seeds)
    print('NEW workers launched:',launched,flush=True)


def report(dataset_filter,seeds):
    picks=load(OUT/'selected_configs.json') or {}
    old=load(OFFICIAL_INPUT)
    summary=[]
    grid_status=[]
    for d in dataset_filter:
        hrows,missing,failed=hybrid_candidates(d)
        grid_status.append({'dataset':d,'hybrid_success':len(hrows),'hybrid_missing':len(missing),'hybrid_failed':len(failed),'baseline_selection_trials_each':9})
        if not isinstance(old,list) or d not in picks:
            continue
        win=picks[d]['selected']['hybrid']
        cfg=dict(dataset=d,method='hybrid',prefix_length=win['prefix_length'],prompt_length=win['prompt_length'],learning_rate=win['learning_rate'])
        h_by_seed={seed:load(target_path(cfg,'heldout',seed)) for seed in SEEDS[:seeds]}
        b_by_method={method:{int(r['seed']):r for r in old if r.get('dataset')==d and r.get('method')==method and r.get('status')=='success'} for method in ('prefix_only','prompt_only')}
        for m in ('hybrid','prefix_only','prompt_only'):
            rows=[(s,(h_by_seed[s] if m=='hybrid' else b_by_method[m].get(s))) for s in SEEDS[:seeds]]
            good=[(s,r) for s,r in rows if r and r.get('status')=='success']
            accs=[float(r['eval_accuracy']) for s,r in good]
            f1s=[float(r['eval_f1_macro']) for s,r in good]
            summary.append({'dataset':d,'method':m,'requested_seed_count':seeds,
                'completed_seed_count':len(good),'paired_seeds':','.join(str(s) for s,r in good),
                'fully_completed':len(good)==seeds,
                'source':'new tuned Hybrid runs' if m=='hybrid' else str(OFFICIAL_INPUT.relative_to(ROOT)),
                'mean_accuracy':statistics.mean(accs) if accs else None,
                'sd_accuracy':statistics.stdev(accs) if len(accs)>1 else None,
                'mean_macro_f1':statistics.mean(f1s) if f1s else None,
                'sd_macro_f1':statistics.stdev(f1s) if len(f1s)>1 else None})
    save(OUT/'grid_status.json',grid_status)
    save(OUT/'heldout_summary.json',summary)
    atomic_csv(OUT/'heldout_summary.csv',summary)
    # Draft text is produced ONLY for a fully completed, genuinely paired final evaluation.
    fully_done = (
        len([r for r in summary if r['fully_completed']]) == 3*len(dataset_filter)
        and len([r for r in grid_status if r['hybrid_success'] == 9]) == len(dataset_filter)
    )
    if fully_done:
        paragraphs = [
            'Supplementary matched-candidate-count sensitivity (NOT a full joint grid)',
            'Validation selection used exactly nine candidates per method/dataset, '
            'with training seed 42, training-derived validation of 500 examples, '
            '3,000 training examples, three epochs, and final-epoch Macro-F1 selection. '
            'For Hybrid, three predeclared paired (prefix,prompt) lengths were '
            'crossed with three learning rates; the component baselines each '
            'used three length choices crossed with the same three learning rates. '
            'The component baseline tuning/held-out results are reused from '
            'previously completed experiments; Hybrid was trained afresh. '
            'All held-out figures below are on the same fixed 500-example '
            'official split using matched model-training seeds. '
            'The comparison controls the NUMBER of tuning candidates, not '
            'the complete joint search space or tuning-history exposure.',
        ]
        for d in dataset_filter:
            w=picks[d]['selected']
            groups={r['method']:r for r in summary if r['dataset']==d}
            paragraphs.append(
                f"{d.upper()}: selection Macro-F1 — Hybrid={100*w['hybrid']['validation_macro_f1']:.2f}%, "
                f"Prefix-only={100*w['prefix_only']['validation_macro_f1']:.2f}%, "
                f"Prompt-only={100*w['prompt_only']['validation_macro_f1']:.2f}%. "
                f"Held-out accuracy (n={seeds} seeds, mean +/- SD) — "
                f"Hybrid={100*groups['hybrid']['mean_accuracy']:.2f} +/- {100*groups['hybrid']['sd_accuracy']:.2f}%; "
                f"Prefix-only={100*groups['prefix_only']['mean_accuracy']:.2f} +/- {100*groups['prefix_only']['sd_accuracy']:.2f}%; "
                f"Prompt-only={100*groups['prompt_only']['mean_accuracy']:.2f} +/- {100*groups['prompt_only']['sd_accuracy']:.2f}%. "
                f"Hybrid selected lengths p={w['hybrid']['prefix_length']}, m={w['hybrid']['prompt_length']} "
                f"and LR={w['hybrid']['learning_rate']:.0e}."
            )
        paragraphs.append(
            'Interpretation: The 9-candidate matching improves fairness of search '
            'opportunities relative to the original fixed-Hybrid ablation. It is '
            'still not a complete factorial joint prefix-by-prompt-by-learning-rate '
            'search, so no globally optimal Hybrid or universal synergy is claimed. '
            'Single-grid selection uses seed 42 and final estimates involve only '
            f'{seeds} model-training seeds per dataset; any wider robustness claim would require more runs.'
        )
        draft=OUT/'DRAFT_EVIDENCE_FOR_AUTHOR_VERIFICATION.txt'
        draft.write_text('\n\n'.join(paragraphs)+'\n',encoding='utf-8')
        print('COMPLETE: verify measurements and adapt the generated evidence draft:',draft,flush=True)
    for row in grid_status:print('GRID:',row,flush=True)
    for row in summary:print('HELD-OUT:',row,flush=True)
    if not summary:print('No final claims yet. Unfinished grid != completed comparison.',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('plan','selection','select','heldout','all','report','worker'),default='plan')
    ap.add_argument('--dataset',choices=('both','sst2','imdb'),default='both')
    ap.add_argument('--seeds',type=int,choices=(3,5),default=3)
    ap.add_argument('--budget-hours',type=float,default=7.0,
                    help='time cap for the current invocation; leave reserve for review and submission')
    ap.add_argument('--cfg',help=argparse.SUPPRESS)
    ap.add_argument('--worker-phase',choices=('selection','heldout'),help=argparse.SUPPRESS)
    ap.add_argument('--seed',type=int,help=argparse.SUPPRESS)
    args=ap.parse_args()
    datasets=DATASETS if args.dataset=='both' else (args.dataset,)
    if args.phase=='worker':
        if not args.cfg or not args.worker_phase or args.seed is None:
            ap.error('worker requires cfg, worker-phase, seed')
        return run_worker(json.loads(args.cfg),args.worker_phase,args.seed)
    if args.phase=='plan':
        for d in datasets:
            p=ablation_candidates(d,'prefix_only');m=ablation_candidates(d,'prompt_only');h,missing,failed=hybrid_candidates(d)
            print(d, '| archived Prefix/Prompt selection:',len(p),len(m),'| new Hybrid success:',len(h),'/9 | missing:',len(missing),'| failed:',len(failed))
        print('Matched selection: 9 candidate configs each for Hybrid, Prefix-only, Prompt-only; 3 LR and three predeclared lengths/pairs.')
        print('New training needed: max 18 Hybrid selection workers + 6 matched heldout (3 seeds per dataset).')
        print('Optional 4 additional heldout workers (5 seeds per dataset). This is NOT a full joint Hybrid search.')
        return 0
    if args.phase=='report': report(datasets,args.seeds);return 0
    if args.phase=='select':
        do_select(datasets);report(datasets,args.seeds);return 0
    if args.budget_hours<=0:
        ap.error('budget-hours must be positive')
    deadline=time.monotonic()+args.budget_hours*3600
    if args.phase in ('selection','all'):
        execute(datasets,'selection',args.seeds,deadline)
    if args.phase in ('heldout','all'):
        execute(datasets,'heldout',args.seeds,deadline)
    report(datasets,args.seeds)
    return 0


if __name__=='__main__':
    sys.exit(main())
