"""Find reusable reviewer evidence in existing project outputs; does NOT train models.

Place in .../peft_sentiment_project/src and run:
    python audit_saved_review_evidence.py
Optionally:
    python audit_saved_review_evidence.py --project-root "C:\\path\\to\\peft_sentiment_project"

Writes outputs/reviewer_evidence_audit/ with manifest, epoch curves (if found),
prediction candidates, and a compact ZIP to upload for review.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

LIMIT_JSON_BYTES=16_000_000
SKIP_DIRS={'__pycache__','.git','.conda','site-packages','venv','.venv','node_modules'}
SKIP_SUFFIXES={'.pt','.pth','.ckpt','.safetensors','.bin','.png','.jpg','.jpeg','.pdf','.npy','.npz','.parquet','.arrow'}
EPOCH_KEYS=('eval_accuracy','eval_f1_macro','eval_loss','loss','learning_rate')
PRED_KEYS=('predictions','preds','y_pred','predicted_labels','label_ids','true_labels','y_true','labels','logits')

def classify_json(obj):
    epoch_rows=[]; pred_summary=[]
    def walk(v,trail='',depth=0):
        if depth>5:return
        if isinstance(v,dict):
            for k,x in v.items():
                low=k.lower()
                if low=='log_history' and isinstance(x,list):
                    for n,row in enumerate(x):
                        if isinstance(row,dict) and ('epoch' in row or 'step' in row) and any(z in row for z in EPOCH_KEYS):
                            epoch_rows.append({'location':trail+'.'+k,'index':n,**{z:row.get(z) for z in ('epoch','step')+EPOCH_KEYS}})
                if low in PRED_KEYS and isinstance(x,list) and len(x)>=100:
                    pred_summary.append({'field':(trail+'.'+k).lstrip('.'),'length':len(x), 'kind':'array'})
                if low in ('history','trainer_state','metrics','experiment','runs','log_history','records','results'):
                    walk(x,trail+'.'+k,depth+1)
        elif isinstance(v,list) and len(v)<=100000:
            for j,x in enumerate(v[:200]):
                if isinstance(x,dict):
                    if ('epoch' in x or 'step' in x) and any(z in x for z in EPOCH_KEYS):
                        epoch_rows.append({'location':trail,'index':j,**{z:x.get(z) for z in ('epoch','step')+EPOCH_KEYS}})
                    if any(z in x for z in PRED_KEYS):walk(x,trail+f'[{j}]',depth+1)
    walk(obj)
    # Dedupe from both explicit log_history and list traversal
    unique=[]; seen=set()
    for x in epoch_rows:
        key=(x.get('location'),x.get('index'),x.get('epoch'),x.get('step'))
        if key not in seen:seen.add(key);unique.append(x)
    return unique,pred_summary

def scan(root:Path):
    output=root/'outputs'
    if not output.is_dir():raise ValueError(f'Outputs folder not found: {output}')
    destination=output/'reviewer_evidence_audit';destination.mkdir(exist_ok=True)
    inventory=[]; curves=[]; pred_candidates=[]; log_candidates=[]; failures=[]
    for path in output.rglob('*'):
        if not path.is_file() or path.is_relative_to(destination) or any(s in SKIP_DIRS for s in path.parts):continue
        suf=path.suffix.lower();relative=path.relative_to(output).as_posix();size=path.stat().st_size
        if suf in SKIP_SUFFIXES:continue
        if suf not in ('.json','.csv','.log','.txt','.jsonl'):continue
        inventory.append({'path':relative,'bytes':size,'type':suf})
        if suf in ('.log','.txt') and any(k in relative.lower() for k in ('train','history','epoch','log')):
            log_candidates.append({'path':relative,'bytes':size,'reason':'possible epoch log; not automatically interpreted'})
        if suf in ('.csv','.jsonl') and any(k in relative.lower() for k in ('predict','logit','label','train_log','trainer_state','history')):
            log_candidates.append({'path':relative,'bytes':size,'reason':'potential structured log or predictions'})
        if suf!='.json' or size>LIMIT_JSON_BYTES:continue
        try:
            obj=json.loads(path.read_text(encoding='utf-8-sig'))
            ep,pred=classify_json(obj)
            for row in ep:curves.append({'source_file':relative,**row})
            for row in pred:pred_candidates.append({'source_file':relative,**row})
            if 'predic' in relative.lower() or 'logit' in relative.lower():
                if not pred:pred_candidates.append({'source_file':relative,'field':'filename only — verify content','length':None,'kind':'possible'})
        except (UnicodeDecodeError,ValueError,TypeError) as exc:
            failures.append({'path':relative,'error':str(exc)[:150]})
    out={
      'project_root':str(root),'files_indexed':len(inventory),
      'epoch_metric_records_detected':len(curves),
      'prediction_array_candidates_detected':len(pred_candidates),
      'potential_log_files_detected':len(log_candidates),
      'note':'Presence of a candidate does not prove paired example identifiers, independent held-out predictions, or valid convergence curves. No training or GPU execution occurred.',
      'inventory':inventory,'prediction_candidates':pred_candidates,
      'log_candidates':log_candidates,'unreadable_json':failures,
    }
    dest_manifest=destination/'reviewer_evidence_manifest.json'
    dest_manifest.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    dest_curve=destination/'review_epoch_metric_records.csv'
    fields=['source_file','location','index','epoch','step']+list(EPOCH_KEYS)
    with dest_curve.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');wr.writeheader();wr.writerows(curves)
    dest_pred=destination/'review_prediction_candidates.csv'
    with dest_pred.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['source_file','field','length','kind']);wr.writeheader();wr.writerows(pred_candidates)
    lines=[
      'SAVED EVIDENCE AUDIT — NO TRAINING PERFORMED',
      f'Project: {root}', f'Files indexed: {len(inventory)}',
      f'Epoch-level records found: {len(curves)}',
      f'Prediction array candidates: {len(pred_candidates)}',
      f'Potential log files: {len(log_candidates)}',
      '', 'Most relevant prediction candidates:',
      *[f"- {x['source_file']} ({x['field']}; n={x['length']})" for x in pred_candidates[:30]],
      '', 'Most relevant log candidates:',
      *[f"- {x['path']}" for x in log_candidates[:30]],
      '', 'WARNING: A file listing alone cannot establish the validity of McNemar/bootstrap.',
    ]
    txtfile=destination/'reviewer_evidence_summary.txt';txtfile.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    zipfile_path=destination/'reviewer_evidence_audit.zip'
    with zipfile.ZipFile(zipfile_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in (dest_manifest,dest_curve,dest_pred,txtfile):z.write(p,p.name)
    print('\n'.join(lines[:7]))
    print('Report:',txtfile)
    print('ZIP:',zipfile_path)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',default=None,help='Project folder containing outputs/ (auto: parent of src/)')
    args=parser.parse_args()
    project=Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parent
    if project.name.lower()=='src':project=project.parent
    try:scan(project)
    except Exception as e:
        print('ERROR:',e,file=sys.stderr);sys.exit(1)
