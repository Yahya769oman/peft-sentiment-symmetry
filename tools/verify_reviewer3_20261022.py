#!/usr/bin/env python3
"""Offline structural consistency check for the 2026-10-22 Reviewer 3 supplementary experiment.
Does NOT independently rerun training or verify archived baseline experiment data.
Run from the repository root: python tools/verify_reviewer3_20261022.py
"""
import json, math, statistics
from pathlib import Path
root = Path(__file__).resolve().parents[1]
p = root / 'outputs' / 'reviewer3_8h_fair_hybrid'
assert p.is_dir(), 'Missing output directory'
loads=lambda f: json.loads(f.read_text(encoding='utf-8'))
summary=loads(p/'heldout_summary.json')
selected=loads(p/'selected_configs.json')
combined=loads(p/'heldout_summary_hybrid_both_datasets.json')
assert len(combined)==2 and {row['dataset'] for row in combined}=={'sst2','imdb'}
assert len(summary) in (3,6), 'Expected three or six dataset-method summary rows (some phase-specific exports cover one dataset)'
assert set(selected)=={'sst2','imdb'}
for dataset in ['sst2','imdb']:
    selections = list((p/'selection').glob(dataset+'_hybrid_*.json'))
    heldouts = list((p/'heldout').glob(dataset+'_hybrid_*.json'))
    assert len(selections)==9 and len(heldouts)==3, (dataset, len(selections),len(heldouts))
    srows=[loads(f) for f in selections]
    hrows=[loads(f) for f in heldouts]
    assert len({(r['prefix_length'],r['prompt_length'],r['learning_rate']) for r in srows})==9
    assert {r['seed'] for r in hrows}=={42,123,2026}
    assert all(r['status']=='success' and r['phase']=='selection' for r in srows)
    assert all(r['status']=='success' and r['phase']=='heldout' for r in hrows)
    assert all(r.get('uses_heldout_for_selection') is False for r in srows)
    for r in srows+hrows:
        assert r['dataset']==dataset and 0 <= r['eval_accuracy']<=1 and 0<=r['eval_f1_macro']<=1
    recovered = [r for r in srows if r.get('recovered_from_console')]
    assert len(recovered)==(2 if dataset=='sst2' else 0), (dataset,len(recovered))
    best=max(srows,key=lambda r:(r['eval_f1_macro'],r['eval_accuracy']))
    sel=selected[dataset]['selected']['hybrid']
    assert (best['prefix_length'], best['prompt_length'],best['learning_rate'])==(sel['prefix_length'], sel['prompt_length'],sel['learning_rate'])
    for metric,sm,ss in [('eval_accuracy','mean_accuracy','sd_accuracy'),('eval_f1_macro','mean_macro_f1','sd_macro_f1')]:
        row=next((r for r in summary if r['dataset']==dataset and r['method']=='hybrid'), None)
        vals=[r[metric] for r in hrows]
        combine_row=next(r for r in combined if r['dataset']==dataset)
        assert math.isclose(statistics.mean(vals),combine_row[sm],abs_tol=1e-10)
        assert math.isclose(statistics.stdev(vals),combine_row[ss],abs_tol=1e-10)
        if row is not None:
            assert row['fully_completed'] and row['completed_seed_count']==3
            assert math.isclose(statistics.mean(vals),row[sm],abs_tol=1e-10)
            assert math.isclose(statistics.stdev(vals),row[ss],abs_tol=1e-10)
        else:
            print(f'NOTE {dataset}: phase-specific heldout_summary.json does not contain this dataset; individual seed JSONs are present and checked. {metric} mean={statistics.mean(vals):.6f}')
    tr={r['train_fingerprint'] for r in hrows}; ev={r['eval_fingerprint'] for r in hrows}
    assert len(tr)==1 and len(ev)==1 and tr!=ev, (dataset,tr,ev)
    print(f'OK {dataset}: 9 distinct Hybrid selection records, 3 heldout seeds; recovered selection logs={len(recovered)}')
print('PASS: Supplementary Hybrid records, selected config and aggregate numbers are structurally consistent.')
print('LIMITATION: No external code execution, train rerun, archived baseline provenance, or secret scan is certified by this check.')
