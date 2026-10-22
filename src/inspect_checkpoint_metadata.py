from pathlib import Path
import json, csv, zipfile, collections
ROOT=Path(__file__).resolve().parent
# This script is intended to live under project/src
project=Path(__file__).resolve().parent.parent
outputs=project/'outputs'
dest=outputs/'reviewer_checkpoint_audit'; dest.mkdir(parents=True,exist_ok=True)
rows=[]
for weights in sorted(outputs.rglob('*.safetensors')):
    folder=weights.parent
    metadata=[]
    for name in ('config.json','adapter_config.json','trainer_state.json','training_args.bin','tokenizer_config.json','special_tokens_map.json','preprocessor_config.json'):
        p=folder/name
        if p.exists(): metadata.append(name)
    state=folder/'trainer_state.json'
    state_info={}
    if state.exists():
        try:
            j=json.loads(state.read_text(encoding='utf-8'))
            state_info={'global_step':j.get('global_step'),'epoch':j.get('epoch'),'log_entries':len(j.get('log_history',[]))}
        except Exception as e: state_info={'read_error':str(e)}
    rows.append({'checkpoint':str(weights.relative_to(project)),'weight_mb':round(weights.stat().st_size/1048576,2),'files_in_checkpoint':'; '.join(sorted(p.name for p in folder.iterdir() if p.is_file())),'has_config':(folder/'config.json').exists(),'has_adapter_config':(folder/'adapter_config.json').exists(),'has_trainer_state':state.exists(),'trainer_state_info':json.dumps(state_info,ensure_ascii=False)})
fields=list(rows[0]) if rows else ['checkpoint']
with (dest/'checkpoint_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
summary={'project':str(project),'checkpoint_count':len(rows),'with_config':sum(x['has_config'] for x in rows),'with_adapter_config':sum(x['has_adapter_config'] for x in rows),'with_trainer_state':sum(x['has_trainer_state'] for x in rows),'note':'This is a metadata-only audit. No model loading, inference or training is performed. Checkpoint availability does not establish comparability to principal five-seed experiments.'}
(dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(dest/'checkpoint_metadata_audit.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ('checkpoint_manifest.csv','summary.json'):z.write(dest/name,arcname=name)
print('CHECKPOINT METADATA AUDIT — NO TRAINING OR INFERENCE')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('ZIP:',dest/'checkpoint_metadata_audit.zip')
