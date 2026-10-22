import os, sys, json, time, random, platform, subprocess
import numpy as np
import torch, transformers
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
CURRENT_DIR=os.path.dirname(os.path.abspath(__file__)); PROJECT_DIR=os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path: sys.path.append(CURRENT_DIR)
from data import load_and_prepare_dataset
from metrics import count_parameters
from models_symmetry import TrueHybridPrefixPromptBertClassifier
EXPERIMENT='hybrid_tuned_official'; OUTPUT_DIR=os.path.join(PROJECT_DIR,'outputs',EXPERIMENT)
DATASETS=['sst2','imdb']; SEEDS=[42,123,2026]; DATA_SEED=2026
MODEL_NAME='bert-base-uncased'; MAX_LENGTH=128; TRAIN_SIZE=3000; EVAL_SIZE=500; EPOCHS=3; TRAIN_BATCH=8; EVAL_BATCH=16; WEIGHT_DECAY=0.01; WARMUP_RATIO=0.1
CONFIGS={'sst2':{'learning_rate':3e-4,'prefix_length':32,'prompt_length':20},'imdb':{'learning_rate':5e-4,'prefix_length':32,'prompt_length':30}}
RESULTS_JSON=os.path.join(OUTPUT_DIR,'hybrid_tuned_official_results.json'); SUMMARY_JSON=os.path.join(OUTPUT_DIR,'hybrid_tuned_official_summary.json')
def set_seed(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def fixed_subset(split,n,seed): return split.shuffle(seed=int(seed)).select(range(min(int(n),len(split))))
def compute_metrics(eval_pred):
 logits,labels=eval_pred
 if isinstance(logits,tuple): logits=logits[0]
 preds=np.argmax(logits,axis=-1); p,r,f1c,s=precision_recall_fscore_support(labels,preds,labels=[0,1],average=None,zero_division=0)
 return {'accuracy':accuracy_score(labels,preds),'f1_macro':f1_score(labels,preds,average='macro'),'f1_weighted':f1_score(labels,preds,average='weighted'),'precision_class0':p[0],'recall_class0':r[0],'f1_class0':f1c[0],'support_class0':s[0],'precision_class1':p[1],'recall_class1':r[1],'f1_class1':f1c[1],'support_class1':s[1]}
def worker_path(d,s): return os.path.join(OUTPUT_DIR,f'worker_{d}_seed{s}.json')
def run_worker(d,seed):
 cfg=CONFIGS[d]; set_seed(seed)
 dataset,tokenizer,num_labels=load_and_prepare_dataset(dataset_name=d,model_name=MODEL_NAME,max_length=MAX_LENGTH)
 evsrc=dataset['validation'] if 'validation' in dataset else dataset['test']
 train_ds=fixed_subset(dataset['train'],TRAIN_SIZE,DATA_SEED); eval_ds=fixed_subset(evsrc,EVAL_SIZE,DATA_SEED)
 set_seed(seed)
 model=TrueHybridPrefixPromptBertClassifier(model_name=MODEL_NAME,num_labels=int(num_labels),prefix_length=cfg['prefix_length'],prompt_length=cfg['prompt_length'],dropout=0.1)
 params=count_parameters(model)
 args=TrainingArguments(output_dir=os.path.join(OUTPUT_DIR,f'{d}_seed{seed}'),num_train_epochs=EPOCHS,per_device_train_batch_size=TRAIN_BATCH,per_device_eval_batch_size=EVAL_BATCH,gradient_accumulation_steps=1,learning_rate=cfg['learning_rate'],weight_decay=WEIGHT_DECAY,warmup_ratio=WARMUP_RATIO,lr_scheduler_type='linear',optim='adamw_torch',logging_steps=50,eval_strategy='epoch',save_strategy='no',report_to='none',remove_unused_columns=False,fp16=torch.cuda.is_available(),seed=seed,data_seed=seed,save_safetensors=False)
 trainer=Trainer(model=model,args=args,train_dataset=train_ds,eval_dataset=eval_ds,processing_class=tokenizer,compute_metrics=compute_metrics)
 gpu_name=None; gpu_total=None
 if torch.cuda.is_available():
  torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); props=torch.cuda.get_device_properties(0); gpu_name=props.name; gpu_total=float(props.total_memory/(1024**2))
 started=time.time(); train_output=trainer.train(); train_seconds=time.time()-started
 peak_alloc=float(torch.cuda.max_memory_allocated()/(1024**2)) if torch.cuda.is_available() else 0.0
 peak_res=float(torch.cuda.max_memory_reserved()/(1024**2)) if torch.cuda.is_available() else 0.0
 metrics=trainer.evaluate()
 row={'experiment':EXPERIMENT,'dataset':d,'method':'hybrid_tuned','seed':seed,'data_seed':DATA_SEED,'train_size':len(train_ds),'eval_size':len(eval_ds),'model_name':MODEL_NAME,'max_length':MAX_LENGTH,'epochs':EPOCHS,'train_batch_size':TRAIN_BATCH,'eval_batch_size':EVAL_BATCH,'learning_rate':cfg['learning_rate'],'prefix_length':cfg['prefix_length'],'prompt_length':cfg['prompt_length'],'optimizer':'adamw_torch','lr_scheduler_type':'linear','weight_decay':WEIGHT_DECAY,'warmup_ratio':WARMUP_RATIO,'fp16':bool(torch.cuda.is_available()),**params,'train_time_seconds':float(train_seconds),'cuda_available':bool(torch.cuda.is_available()),'gpu_name':gpu_name,'gpu_total_memory_mb':gpu_total,'peak_train_gpu_allocated_mb':peak_alloc,'peak_train_gpu_reserved_mb':peak_res,'python_version':platform.python_version(),'torch_version':torch.__version__,'transformers_version':transformers.__version__,'status':'success'}
 if hasattr(train_output,'metrics') and 'train_loss' in train_output.metrics: row['train_loss']=float(train_output.metrics['train_loss'])
 for k,v in metrics.items(): row[k]=float(v) if isinstance(v,(int,float,np.integer,np.floating)) else v
 return row
def worker_main(d,seed):
 os.makedirs(OUTPUT_DIR,exist_ok=True)
 try: row=run_worker(d,int(seed))
 except Exception as e: row={'experiment':EXPERIMENT,'dataset':d,'method':'hybrid_tuned','seed':int(seed),'status':'failed','error':repr(e)}
 with open(worker_path(d,int(seed)),'w',encoding='utf-8') as f: json.dump(row,f,indent=2)
 print(json.dumps(row,indent=2))
 if row.get('status')!='success': raise SystemExit(1)
def collect_rows():
 rows=[]
 for d in DATASETS:
  for s in SEEDS:
   p=worker_path(d,s)
   if os.path.exists(p):
    with open(p,'r',encoding='utf-8') as f: rows.append(json.load(f))
 return rows
def summarize(rows):
 ok=[r for r in rows if r.get('status')=='success']; out=[]; metrics=['eval_accuracy','eval_f1_macro','eval_f1_weighted','train_time_seconds','peak_train_gpu_allocated_mb','peak_train_gpu_reserved_mb','trainable_params','trainable_ratio']
 for d in DATASETS:
  g=[r for r in ok if r['dataset']==d]
  if not g: continue
  cfg=CONFIGS[d]; s={'dataset':d,'method':'hybrid_tuned','learning_rate':cfg['learning_rate'],'prefix_length':cfg['prefix_length'],'prompt_length':cfg['prompt_length'],'n_runs':len(g),'seeds':','.join(str(r['seed']) for r in g)}
  for m in metrics:
   vals=[float(r[m]) for r in g if r.get(m) is not None]
   if vals: s[m+'_mean']=float(np.mean(vals)); s[m+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0
  out.append(s)
 with open(RESULTS_JSON,'w',encoding='utf-8') as f: json.dump(rows,f,indent=2)
 with open(SUMMARY_JSON,'w',encoding='utf-8') as f: json.dump(out,f,indent=2)
 return out
def master_main():
 os.makedirs(OUTPUT_DIR,exist_ok=True); print('='*78); print('FINAL TUNED HYBRID | OFFICIAL 3-SEED RERUN'); print('='*78)
 for d in DATASETS:
  for s in SEEDS:
   p=worker_path(d,s)
   if os.path.exists(p):
    try:
     with open(p,'r',encoding='utf-8') as f: old=json.load(f)
     if old.get('status')=='success': print(f'SKIP completed: {d} / seed={s}'); continue
    except Exception: pass
   cfg=CONFIGS[d]; print('\n'+'='*78); print(f"dataset={d} | seed={s} | lr={cfg['learning_rate']} | prefix={cfg['prefix_length']} | prompt={cfg['prompt_length']}"); print('='*78)
   rc=subprocess.run([sys.executable,os.path.abspath(__file__),'--worker',d,str(s)]).returncode
   if rc!=0: print(f'FAILED: {d} / seed={s}; continuing.')
 rows=collect_rows(); summary=summarize(rows); print('\n'+'='*78); print('FINAL TUNED HYBRID SUMMARY'); print('='*78)
 for s in summary: print(f"{s['dataset']:5s} | acc={s.get('eval_accuracy_mean',float('nan')):.4f} +/- {s.get('eval_accuracy_std',float('nan')):.4f} | f1={s.get('eval_f1_macro_mean',float('nan')):.4f} +/- {s.get('eval_f1_macro_std',float('nan')):.4f}")
 print(f'\nResults: {RESULTS_JSON}'); print(f'Summary: {SUMMARY_JSON}')
if __name__=='__main__':
 if len(sys.argv)==4 and sys.argv[1]=='--worker': worker_main(sys.argv[2],int(sys.argv[3]))
 else: master_main()
