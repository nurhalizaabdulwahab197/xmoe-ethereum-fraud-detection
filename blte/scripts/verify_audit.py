"""Verify 120 historical runs against original hashes. Default is read-only.
--copy-to creates a NEW evidence directory only after all checks pass.
"""
import argparse
import csv
import shutil
from pathlib import Path
from common import ROOT,read,write,sha,require,within,splits,check_prediction

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-root',type=Path,required=True)
    p.add_argument('--copy-to',type=Path)
    p.add_argument('--include-checkpoints',action='store_true')
    a=p.parse_args();audit=a.audit_root.resolve()
    require(audit.is_dir(),'Audit root missing')
    if a.copy_to:
        a.copy_to=a.copy_to.resolve()
        require(not a.copy_to.exists(),'Destination exists; no overwrite')
        require(not a.copy_to.is_relative_to(audit),'Destination must be outside source audit')
    input_hashes=read(ROOT/'provenance/INPUT_SHA256.json')
    for name,h in input_hashes.items():require(sha(audit/'upstream-paper'/name)==h,'Input mismatch: '+name)
    import numpy as np
    import pandas as pd
    labels=pd.read_csv(audit/'upstream-paper/BLTE_multiclass_v2.csv',usecols=['attack_encoded'])['attack_encoded'].to_numpy()
    saved_splits=splits()
    expected=read(ROOT/'provenance/EXPECTED_AUDIT_EVIDENCE.json')['files']
    for n,h in expected.items():require(sha(within(audit,n))==h,'Evidence changed: '+n)
    print('PASS original summary evidence chain:',len(expected),'files',flush=True)
    batch_records=read(ROOT/'provenance/PRIMARY_BATCHES.json')
    with (ROOT/'results/tables/one_hundred_twenty_runs.csv').open() as f:
        rows=list(csv.DictReader(f))
    expected_rows={(r['configuration'],r['model'],int(r['split_seed'])):r for r in rows}
    require(len(expected_rows)==len(rows)==120,'Primary summary keys')
    mapping={'GCN':'gcn','GAT':'gat','GraphSAGE':'graphsage','MLP':'mlp','XMoE_CE':'xmoe-ce','Full_XMoE':'full-xmoe'}
    cfg={'author':'released-code','paper':'paper-described'}
    configs=read(ROOT/'provenance/CONFIGURATIONS.json')
    planned=[];seen=set();bytes_to_copy=0
    for batch in batch_records:
        bpath=within(audit,batch['file']);require(sha(bpath)==batch['sha256'],'Batch changed')
        state=read(bpath);require(state['status']==batch['status'] and len(state['runs'])==batch['count'],'Batch status/count')
        for run in state['runs']:
            model=run.get('model_label',run.get('model'));conf=run['configuration'];seed=run['split_seed'];key=(conf,model,seed)
            require(key not in seen,'Duplicate run');seen.add(key)
            folder=within(audit,'audit_runs/'+Path(run['folder']).name)
            meta=read(folder/'RUN_METADATA.json');require(sha(folder/'RUN_METADATA.json')==run['metadata_sha256'],'Metadata changed')
            require(meta['status']==('PASS_FULL_XMOE_BLTE_RUN' if model=='Full_XMoE' else 'PASS_RUN_VALIDATED'),'Incomplete run')
            require(meta['configuration']==conf and meta['split_seed']==seed and meta['initialization_seed']==seed and meta['epochs']==300,'Run identity')
            require(meta['graph_sha256']==input_hashes['BLTE_multiclass_graph.pt'],'Run graph')
            require(meta['split_sha256']==sha(saved_splits[seed][0]),'Run split')
            rec=configs[cfg[conf]+'/'+mapping[model]]
            sourcehash=meta.get('adapter_sha256',meta.get('script_sha256',meta.get('prepared_sha256')))
            require(sourcehash==rec['sha256'],'Training script mismatch')
            require(meta.get('torch')=='2.0.1+cu118' and meta.get('pyg')=='2.3.1','Recorded versions')
            if model in ('MLP','XMoE_CE','Full_XMoE'):
                require(meta['learning_rate']==rec['learning_rate'],'LR metadata')
                require(meta['pcgrad']==(model=='Full_XMoE') and meta['calibration'] is False,'Training variant metadata')
                frozen=read(ROOT/'provenance/SOURCE_SHA256.json')
                for name,h in meta.get('model_source_sha256',{}).items():
                    require(frozen['sources/prepared/model_source/models/'+name]==h,'Model source mismatch')
            if model=='Full_XMoE':
                for name,val in {'expert_loss_weight':.25,'scam_specialist_loss_weight':.5,'x_moe_expert_task_loss_weight':1.,'x_moe_normal_phishing_penalty':.30}.items():
                    require(meta[name]==val,'Full coefficient mismatch')
                frozen=read(ROOT/'provenance/SOURCE_SHA256.json')
                for name,h in meta['source_sha256'].items():require(frozen['sources/full/'+name]==h,'Full source mismatch')
            for name,h in meta['output_sha256'].items():require(sha(within(folder,name))==h,'Output changed: '+str(folder/name))
            prediction=folder/f'{model}_test_predictions.csv';stored=read(folder/'VALIDATED_METRICS.json')
            require(prediction.name in meta['output_sha256'] and 'VALIDATED_METRICS.json' in meta['output_sha256'],'Missing output hashes')
            computed=check_prediction(prediction,saved_splits[seed][1]['test'],labels[saved_splits[seed][1]['test']])
            reference=expected_rows[key]
            require(Path(reference['folder']).name==folder.name,'Summary folder mismatch')
            for name,val in computed.items():
                if name=='confusion_matrix':continue
                require(np.isclose(val,float(reference[name]),atol=1e-10,rtol=0),'Summary mismatch: '+name)
                if name in stored:require(np.isclose(val,stored[name],atol=1e-10,rtol=0),'Stored metric mismatch: '+name)
            for i in range(3):
                for j in range(3):require(computed['confusion_matrix'][i][j]==int(reference[f'cm_{i}_{j}']),'Summary confusion matrix')
            if 'confusion_matrix' in stored:require(computed['confusion_matrix']==stored['confusion_matrix'],'Saved confusion matrix')
            selected=[('predictions.csv',prediction),('metrics.json',folder/'VALIDATED_METRICS.json')]
            if model=='Full_XMoE':
                reload=read(folder/'RELOAD_VALIDATION.json')
                require(reload['status']=='PASS_FULL_XMOE_RELOAD','Reload status')
                for name in ['test_node_ids','predictions','probabilities','metrics_recomputed']:require(reload[name]=='PASS','Reload check')
                require(reload['checkpoint_sha256']==sha(folder/'best_full_xmoe_model.pt'),'Reload checkpoint hash')
                selected.append(('reload_validation.json',folder/'RELOAD_VALIDATION.json'))
                checkpoint=folder/'best_full_xmoe_model.pt'
            else:
                history=folder/f'{model}_training_history.csv';h=pd.read_csv(history)
                require(len(h)==300 and np.isclose(h['Validation Macro F1'].max(),stored['best_validation_macro_f1'],atol=1e-12,rtol=0),'History/checkpoint selection')
                selected.append(('training_history.csv',history))
                checkpoints=[within(folder,n) for n in meta['output_sha256'] if n.endswith('.pt') and n!='BLTE_multiclass_graph.pt']
                require(len(checkpoints)==1,'Ambiguous model checkpoint');checkpoint=checkpoints[0]
            if a.include_checkpoints:selected.append(('checkpoint.pt',checkpoint))
            relative=Path(cfg[conf])/mapping[model]/f'seed_{seed}'
            planned.append((folder,relative,selected,meta,run['metadata_sha256']))
            bytes_to_copy+=sum(p.stat().st_size for _,p in selected)
            print(f'[{len(seen)}/120] PASS {conf}/{model}/{seed}',flush=True)
    require(seen==set(expected_rows),'Missing/unexpected primary keys')
    print('PASS_120_LOCAL_EVIDENCE; payload MiB:',round(bytes_to_copy/2**20,2),flush=True)
    if not a.copy_to:
        print('READ ONLY: no files written; no training, commit or push.');return
    a.copy_to.mkdir(parents=True,exist_ok=False)
    imported=[]
    try:
        for original,relative,selected,meta,original_sha in planned:
            dest=a.copy_to/relative;dest.mkdir(parents=True)
            for name,path in selected:
                shutil.copy2(path,dest/name);require(sha(path)==sha(dest/name),'Copy hash mismatch')
            # Relative publication metadata; preserve original metadata digest separately.
            clean={k:meta[k] for k in ['configuration','split_seed','initialization_seed','epochs','graph_sha256','split_sha256','torch','pyg','cuda_runtime'] if k in meta}
            clean.update({'model':str(relative.parts[1]),'status':'PASS_IMPORTED_EVIDENCE','original_metadata_sha256':original_sha,
              'original_run_name':original.name,'checkpoint_included':a.include_checkpoints,
              'output_sha256':{n:sha(dest/n) for n,_ in selected}})
            write(dest/'metadata.json',clean)
            imported.append({'path':relative.as_posix(),'metadata_sha256':sha(dest/'metadata.json'),'original_metadata_sha256':original_sha})
        write(a.copy_to/'IMPORT_VALIDATION.json',{'status':'PASS_120_IMPORTED_EVIDENCE','runs':imported,'training':False,'source_modified':False,
            'checkpoints_included':a.include_checkpoints,'note':'Historical metadata digest retained; original local audit remains the archive for unbundled files.'})
        print('PASS_120_EVIDENCE_COPIED:',a.copy_to)
    except BaseException:
        write(a.copy_to/'IMPORT_INCOMPLETE.json',{'status':'INCOMPLETE_KEEP_FOR_REVIEW'});raise
if __name__=='__main__':main()
