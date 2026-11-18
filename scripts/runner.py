"""Portable wrapper around byte-identical prepared scripts. Training requires --run."""
import argparse
import contextlib
import importlib.util
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import threading
import time
import traceback
from common import ROOT,MODELS,CONFIGURATIONS,read,write,sha,require,source_check,input_check,splits,environment,check_prediction

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text);stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:stream.flush()

def worker(a,seed,folder):
    import numpy as np
    import pandas as pd
    import torch
    versions=environment();source_check();upstream=input_check(a.upstream)
    graph=torch.load(upstream/'BLTE_multiclass_graph.pt',map_location='cpu',weights_only=False)
    split_path,partition=splits(graph)[seed]
    record=read(ROOT/'provenance/CONFIGURATIONS.json')[a.configuration+'/'+a.model]
    script=ROOT/record['script'];require(sha(script)==record['sha256'],'Script changed')
    folder.mkdir(parents=True,exist_ok=False)
    metadata={'status':'RUNNING','configuration':a.configuration,'model':a.model,'label':record['label'],
      'split_seed':seed,'initialization_seed':seed,'epochs':300,'learning_rate':record['learning_rate'],
      'weight_decay':.0005,'base_loss':'weighted_ce','pcgrad':a.model=='full-xmoe','calibration':False,
      'graph_sha256':sha(upstream/'BLTE_multiclass_graph.pt'),'split_sha256':sha(split_path),
      'script_sha256':sha(script),'wrapper_sha256':sha(Path(__file__)),
      'versions':versions,'validation_selection':'highest validation Macro-F1',
      'execution_policy':'Original branch-specific seed/CUDA policy retained; no global deterministic override'}
    write(folder/'metadata.json',metadata)
    previous=Path.cwd()
    try:
        with (folder/'execution.log').open('w') as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            if a.model=='full-xmoe':
                sys.path.insert(0,str(script.parent))
                spec=importlib.util.spec_from_file_location('frozen_full_runner',script)
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                data=graph.clone()
                for key,attr in [('train','train_mask'),('validation','val_mask'),('test','test_mask')]:
                    mask=torch.zeros(data.num_nodes,dtype=torch.bool)
                    mask[torch.as_tensor(partition[key])]=True;setattr(data,attr,mask)
                result=module.run_once(released_data=data,csv_path=upstream/'BLTE_multiclass_v2.csv',
                    model_name='x_moe_gnn',protocol='artifact',seed=seed,epochs=300,device=torch.device('cuda'),
                    deterministic=False,explanation_path=None,checkpoint_path=folder/'checkpoint.pt',
                    prediction_path=folder/'predictions.csv',base_loss='weighted_ce',focal_gamma=2.,
                    calibrate_class_bias=False,expert_loss_weight=.25,scam_specialist_loss_weight=.5,
                    x_moe_expert_task_loss_weight=1.,x_moe_normal_phishing_penalty=.30,
                    learning_rate=record['learning_rate'])
                saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
                model=module.MODELS['x_moe_gnn'](data.num_node_features,3).cuda()
                model.load_state_dict(saved['state_dict']);model.eval()
                with torch.no_grad():
                    x=data.x.cuda();edge=data.edge_index.cuda();logits=model(x,edge)
                    val_pred=logits[data.val_mask.cuda()].argmax(1).cpu().numpy()
                    exp=model.explain(x,edge)
                    probs=torch.softmax(torch.log(exp['probabilities'].clamp_min(1e-12)),dim=1)[data.test_mask.cuda()].cpu().numpy()
                from sklearn.metrics import f1_score
                require(np.isclose(f1_score(data.y[data.val_mask].numpy(),val_pred,labels=[0,1,2],average='macro',zero_division=0),result['best_validation_macro_f1'],atol=1e-12,rtol=0),'Reload validation')
                saved_pred=pd.read_csv(folder/'predictions.csv')
                require(np.allclose(probs,saved_pred[['normal_probability','phishing_probability','scamming_probability']].to_numpy(),atol=1e-7,rtol=1e-5),'Reload probabilities')
                metadata.update({'internal_data_mode':'artifact with reconstructed masks','expert_loss_weight':.25,
                    'scam_specialist_loss_weight':.5,'expert_task_loss_weight':1.,'normal_phishing_penalty':.30,'checkpoint_reload':'PASS'})
            else:
                shutil.copy2(upstream/'BLTE_multiclass_graph.pt',folder/'BLTE_multiclass_graph.pt')
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                os.chdir(folder)
                values=runpy.run_path(str(script),init_globals={'AUDIT_SPLIT_FILE':str(split_path)},run_name='__main__')
                require(values['data'].x.is_cuda and values['epochs']==300,'CUDA/epoch mismatch')
                for key,attr in [('train','train_mask'),('validation','val_mask'),('test','test_mask')]:
                    require(np.array_equal(getattr(values['data'],attr).nonzero().view(-1).cpu().numpy(),partition[key]),'Mask mismatch')
                for group in values['optimizer'].param_groups:
                    require(group['lr']==record['learning_rate'] and group['weight_decay']==.0005,'Optimizer mismatch')
                probs=np.asarray(values['y_score'])
                pd.DataFrame({'node_id':partition['test'],'y_true':values['y_true'],'y_pred':values['y_pred'],
                    'prob_normal':probs[:,0],'prob_phishing':probs[:,1],'prob_scamming':probs[:,2]}).to_csv(folder/'predictions.csv',index=False)
                shutil.copy2(folder/values['best_model_path'],folder/'checkpoint.pt')
                shutil.copy2(folder/f"{record['label']}_training_history.csv",folder/'training_history.csv')
                result={'best_validation_macro_f1':float(values['best_val_f1'])}
            computed=check_prediction(folder/'predictions.csv',partition['test'],graph.y.numpy()[partition['test']])
            for key,val in computed.items():
                if key in result and isinstance(val,float):require(np.isclose(val,result[key],atol=1e-10,rtol=0),key)
            computed['best_validation_macro_f1']=float(result['best_validation_macro_f1'])
            if a.model!='full-xmoe':
                h=pd.read_csv(folder/'training_history.csv')
                require(len(h)==300 and np.isclose(h['Validation Macro F1'].max(),computed['best_validation_macro_f1'],atol=1e-12,rtol=0),'History/selection mismatch')
            write(folder/'metrics.json',computed)
            names=['predictions.csv','checkpoint.pt','metrics.json']
            if a.model!='full-xmoe':names.append('training_history.csv')
            metadata['output_sha256']={n:sha(folder/n) for n in names}
            metadata['status']='PASS_NEW_RUN_VALIDATED'
            print('PASS',a.configuration,a.model,seed,computed['macro_f1'],flush=True)
    except BaseException:
        metadata['status']='FAILED_OR_INTERRUPTED'
        (folder/'ERROR.txt').write_text(traceback.format_exc());raise
    finally:
        os.chdir(previous);write(folder/'metadata.json',metadata)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--configuration',choices=CONFIGURATIONS,required=True)
    p.add_argument('--seeds',default=','.join(map(str,range(100,110))))
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--run',action='store_true',help='Explicitly start training; otherwise print plan only')
    p.add_argument('--worker-seed',type=int,help=argparse.SUPPRESS)
    a=p.parse_args();source_check()
    seeds=[int(s.strip()) for s in a.seeds.split(',')]
    require(len(set(seeds))==len(seeds) and set(seeds)<=set(range(100,110)) and len(seeds)>0,'Seeds must be unique in 100–109')
    a.upstream=a.upstream.resolve();a.output_dir=a.output_dir.resolve()
    require(not a.output_dir.is_relative_to(ROOT/'sources'),'Output cannot be inside frozen sources')
    require(not a.output_dir.is_relative_to(ROOT/'results/reference'),'Reference evidence is read-only to the runner')
    if a.worker_seed is not None:
        require(a.run and a.worker_seed in seeds,'Worker requires explicit --run')
        worker(a,a.worker_seed,a.output_dir/f'seed_{a.worker_seed}');return
    print('PLAN:',a.configuration,a.model,'seeds=',seeds,'epochs=300; no CPU fallback',flush=True)
    if not a.run:
        print('PLAN ONLY: no training or output writes. Add --run to execute.');return
    environment();input_check(a.upstream);splits()
    require(not a.output_dir.exists(),'Output directory exists. Choose a new directory; no overwrite/resume.')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for seed in seeds:
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--model',a.model,'--configuration',a.configuration,
            '--seeds',a.seeds,'--upstream',str(a.upstream),'--output-dir',str(a.output_dir),'--worker-seed',str(seed),'--run']
        with subprocess.Popen(cmd) as proc:
            start=time.monotonic()
            try:
                while True:
                    try:code=proc.wait(timeout=30);break
                    except subprocess.TimeoutExpired:print(f'Seed {seed}: running {int(time.monotonic()-start)} seconds',flush=True)
                require(code==0,f'Worker failed: {seed}; keep evidence and inspect ERROR.txt')
            except BaseException:
                proc.terminate();proc.wait();raise
if __name__=='__main__': main()
