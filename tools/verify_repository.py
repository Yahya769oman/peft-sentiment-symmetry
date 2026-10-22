#!/usr/bin/env python3
"""Offline verification of saved code and main result summaries; no training."""
import ast
import json
import math
from pathlib import Path
from collections import Counter
from statistics import mean, stdev

ROOT=Path(__file__).resolve().parent.parent
SRC=ROOT/'src'
OUT=ROOT/'outputs'

expected={
    'full_ft': ('full_ft_fresh_process/full_ft_fresh_process_summary.json',
                {('sst2','full'):0.8848,('imdb','full'):0.8656}),
    'bert_peft': ('exp2_peft_official/exp2_peft_official_summary.json',
                  {('sst2','hybrid'):0.8644,('imdb','hybrid'):0.8080}),
    'ia3': ('ia3_official_5seed/ia3_official_5seed_summary.json',
             {('sst2','ia3'):0.8344,('imdb','ia3'):0.8024}),
    'ablation': ('exp3_ablation_independent_tuning/ablation_tuned_official_summary.json',
                  {('sst2','prefix_only'):0.8356,('sst2','prompt_only'):0.8648,
                   ('imdb','prefix_only'):0.8032,('imdb','prompt_only'):0.8124}),
}

def check():
    files=list(SRC.glob('*.py'))
    assert len(files)>=54, f'Missing scripts: found {len(files)}'
    for p in files:ast.parse(p.read_text(encoding='utf8'),filename=str(p))
    total_json=0
    for p in OUT.rglob('*.json'):
        json.loads(p.read_text(encoding='utf8'), parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
        total_json+=1
    detail_files={
        'full_ft':'full_ft_fresh_process/full_ft_fresh_process_results.json',
        'bert_peft':'exp2_peft_official/exp2_peft_official_results.json',
        'ia3':'ia3_official_5seed/ia3_official_5seed_results.json',
        'ablation':'exp3_ablation_independent_tuning/ablation_tuned_official_results.json',
    }
    summary_groups_checked=0
    for label,(rel,vals) in expected.items():
        rows=json.loads((OUT/rel).read_text(encoding='utf8'))
        details=json.loads((OUT/detail_files[label]).read_text(encoding='utf8'))
        for summary in rows:
            detail=[x for x in details if x.get('dataset')==summary['dataset']
                    and x.get('method')==summary['method'] and x.get('status')=='success']
            assert len(detail)==summary['n_runs']==5, (label,summary['dataset'],summary['method'],'missing seeds')
            assert len({r['seed'] for r in detail})==5,(label,'duplicate seeds')
            assert math.isclose(mean(x['eval_accuracy'] for x in detail),summary['eval_accuracy_mean'],abs_tol=1e-12), (label,'mean mismatch')
            assert math.isclose(stdev(x['eval_accuracy'] for x in detail),summary['eval_accuracy_std'],abs_tol=1e-12), (label,'SD mismatch')
            assert math.isclose(mean(x['eval_f1_macro'] for x in detail),summary['eval_f1_macro_mean'],abs_tol=1e-12), (label,'Macro-F1 mismatch')
            summary_groups_checked+=1
        for (dataset,method),value in vals.items():
            matched=[x for x in rows if x['dataset']==dataset and x['method']==method]
            assert len(matched)==1,(label,dataset,method)
            row=matched[0]
            assert row['n_runs']==5,(label,dataset,method,'expected five seeds')
            assert math.isclose(row['eval_accuracy_mean'],value,rel_tol=0,abs_tol=0.000001),(label,dataset,method,row['eval_accuracy_mean'],value)
    r=OUT/'reviewer3_wide_lr_matched_hybrid'/'selection'
    records=[json.loads(p.read_text(encoding='utf8')) for p in r.glob('*.json')]
    counts=Counter(x.get('status') for x in records)
    print('Python scripts parse: ',len(files))
    print('Strict JSON files: ',total_json)
    print('Recalculated five-seed summary groups: ',summary_groups_checked,'PASSED')
    print('Principal manuscript-value checks: PASSED')
    print('Wide-LR selection records:',len(records),'statuses:',dict(counts))
    print('NOTE: Wide-LR selection is incomplete; record counts are NOT held-out evaluation.')
    print('NOTE: This test does not execute model training or validate trained checkpoints.')
    return 0

if __name__=='__main__':raise SystemExit(check())
