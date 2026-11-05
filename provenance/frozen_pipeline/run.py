"""One CLI, one config, one original training engine, two explicit data adapters."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys

from adapters import checked_entry, load_dataset
from shared_engine import reference, memory_options, run_reference, safe_result, signed_rank_p
from utility import read, sha, verify_code
from rq1_safe_io import atomic_json
from resource_guard import supervise
from attempts import resolve_attempt, inventory, bind_study

ROOT = Path(__file__).resolve().parent


def configuration():
    verify_code(ROOT)
    config = read(ROOT / 'common_config.json')
    if config['optimizer'] != {'name':'Adam','learning_rate':.001,'weight_decay':.0005}:
        raise ValueError('Optimizer config does not match the pinned source constants')
    expected = config['runtime_versions']
    if sys.version.split()[0] != expected['python'].split()[0]:
        raise ValueError('Use the shared locked Python environment')
    for name in ('torch','torch-geometric','numpy','pandas','scikit-learn','scipy','psutil'):
        if importlib.metadata.version(name) != expected[name]:
            raise ValueError('Shared environment differs: ' + name)
    return config


def worker(args, config, registry):
    import math
    import torch
    import psutil
    torch.set_num_threads(config['threads'])
    if args.mode == 'bridge':
        if args.dataset != 'blte' or 'prepared_common' in registry[args.dataset]:
            raise ValueError('The pre-test implementation bridge uses BLTE on both paths')
        from bridge import run_bridge
        result = run_bridge(args.model,args.seed,registry,config)
    else:
        ref = reference()
        values, meta = load_dataset(args.dataset,registry,args.seed,ref)
        if args.mode == 'probe':
            from original_step import original_training_step
            ref.set_seed(args.seed,True)
            x,edge,y,train = values[:4]
            model = memory_options(ref.MODELS[args.model](meta['input_dim'],3).cpu(),args.model)
            counts=torch.bincount(y[train],minlength=3)
            weights=counts.sum()/counts
            weights/=weights.sum()
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0005)
            options={k:v for k,v in config['training_options'].items() if k != 'calibrate_class_bias'}
            details=original_training_step(model,args.model,x,edge,y,train,optimizer,weights,**options)
            if not math.isfinite(details['base_loss']) or any(not torch.isfinite(p).all() or
                    (p.grad is not None and not torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError('Nonfinite technical probe update')
            result={'status':'one_shared_pipeline_step_passed','model':args.model,'seed':args.seed,
                    'step':details,'dataset_metadata':meta,'test_evaluated':False,'validation_evaluated':False}
        elif args.mode in ('validate','reproduce'):
            from validation_only import run_validation, compare_checkpoint
            if args.mode=='reproduce' and (args.dataset!='blte' or args.model!='x_moe_gnn' or
                    args.seed!=100 or args.reference_checkpoint is None):
                raise ValueError('Reproduction mode requires BLTE X-MoE seed100 and its original reference checkpoint')
            result = run_validation(ref,args.model,args.seed,values,args.output_dir,
                epochs=config['epochs'] if args.mode=='reproduce' else args.check_epochs,
                feature_names=meta['feature_names'],training_options=config['training_options'])
            result['dataset_metadata'] = meta
            if args.mode == 'reproduce':
                result['reference_checkpoint_comparison'] = compare_checkpoint(
                    args.output_dir/'best_validation_checkpoint.pt',args.reference_checkpoint)
        elif args.mode == 'train':
            result=safe_result(run_reference(ref,args.model,args.seed,values,args.output_dir,
                epochs=config['epochs'],feature_names=meta['feature_names'], training_options=config['training_options']))
            result.update(status='shared_pipeline_performance_run_completed',dataset=args.dataset,
                dataset_metadata=meta,test_evaluated=True,validation_evaluated=True,
                checkpoint_sha256=sha(args.output_dir/'best_checkpoint.pt'),
                prediction_sha256=sha(args.output_dir/'test_predictions.csv'))
        else:
            raise ValueError('Unsupported worker mode')
    result.update(dataset=args.dataset,model=args.model,seed=args.seed,mode=args.mode,
        shared_config_sha256=sha(ROOT/'common_config.json'),release_manifest_sha256=sha(ROOT/'release_manifest.json'),
        registry_sha256=sha(args.registry),runner_sha256=sha(Path(__file__)))
    result.update(runtime_versions=config['runtime_versions'],threads=config['threads'],
                  optimizer=config['optimizer'],effective_training_options=config['training_options'])
    result.update(actual_torch_threads=torch.get_num_threads(),actual_torch_interop_threads=torch.get_num_interop_threads())
    memory=psutil.Process().memory_info()
    peak=getattr(memory,'peak_wset',None)
    result['worker_windows_peak_working_set_gib']=peak/1024**3 if peak else None
    atomic_json(args.output_dir/'result.json',result)
    atomic_json(args.output_dir/'worker_result.json',result)


def complete(folder,dataset,model,seed,args):
    try:
        return complete_checked(folder,dataset,model,seed,args)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def complete_checked(folder,dataset,model,seed,args):
    if not (folder/'result.json').exists() or not (folder/'resource_probe.json').exists():
        return False
    r=read(folder/'result.json')
    return (r['status']=='shared_pipeline_performance_run_completed' and
        read(folder/'resource_probe.json')['status']=='passed' and r['dataset']==dataset and
        r['model']==model and r['seed']==seed and r['epochs']==300 and r['test_evaluated'] is True and
        r['shared_config_sha256']==sha(ROOT/'common_config.json') and
        r['release_manifest_sha256']==sha(ROOT/'release_manifest.json') and r['registry_sha256']==sha(args.registry) and
        r['checkpoint_sha256']==sha(folder/'best_checkpoint.pt') and
        r['prediction_sha256']==sha(folder/'test_predictions.csv') and
        r['training_history_sha256']==sha(folder/'training_history.csv'))


def summarize(output,dataset,config,args):
    runs=[]
    for seed in config['seeds']:
        for model in config['models']:
            folder, done = resolve_attempt(output,model,seed,lambda p: complete(p,dataset,model,seed,args))
            if done:
                runs.append(read(folder/'result.json'))
    summary={'dataset':dataset,'completed_runs':len(runs),'planned_runs':50,'all_complete':len(runs)==50,
             'scope':config['scope'],'models':{}}
    entry = read(args.registry)[dataset]
    if 'prepared_common' in entry:
        summary['scope'] = 'Common three-feature structural study; fresh training on each source graph with dataset-native labels. Not original BLTE23 reproduction.'
        summary['representation'] = entry['profile']
    metrics=('accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','normal_f1',
             'phishing_f1','scamming_f1','phishing_ap','scamming_ap')
    for model in config['models']:
        group=[r for r in runs if r['model']==model]
        if group:
            summary['models'][model]={'seeds':[r['seed'] for r in group], 'metrics':{
                m:{'mean':statistics.mean(r[m] for r in group),
                   'sample_std':statistics.stdev(r[m] for r in group) if len(group)>1 else None} for m in metrics}}
    if runs:
        if len(runs)==50:
            pairs=[]
            by_key={(r['model'],r['seed']):r for r in runs}
            for baseline in ('mlp','gcn','gat','graphsage'):
                differences=[by_key['x_moe_gnn',s]['macro_f1']-by_key[baseline,s]['macro_f1'] for s in config['seeds']]
                pairs.append({'baseline':baseline,'mean_delta':statistics.mean(differences),
                    'raw_signed_rank_p':signed_rank_p(differences),'wins':sum(d>0 for d in differences),
                    'ties':sum(d==0 for d in differences),'losses':sum(d<0 for d in differences)})
            previous=0.
            for rank,row in enumerate(sorted(pairs,key=lambda r:r['raw_signed_rank_p'])):
                previous=max(previous,min(1.,(4-rank)*row['raw_signed_rank_p']))
                row['holm_adjusted_p']=previous
            summary['macro_f1_comparisons']=pairs
        atomic_json(output/'summary.json',summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['blte','rq1'],required=True)
    parser.add_argument('--mode',choices=['verify','bridge','probe','validate','reproduce','train'],default='verify')
    parser.add_argument('--reference-checkpoint',type=Path,
                        help='Original locally generated BLTE checkpoint for validation-only reproduction')
    parser.add_argument('--check-epochs',type=int,default=5,
                        help='Validation-only technical budget (1..5); final training remains 300 epochs')
    parser.add_argument('--registry',type=Path,default=ROOT/'datasets.local.json')
    parser.add_argument('--models',default='mlp,gcn,gat,graphsage,x_moe_gnn')
    parser.add_argument('--seeds',default='100')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--graph',type=Path)
    parser.add_argument('--evaluation',type=Path)
    parser.add_argument('--threads',type=int,default=None)
    parser.add_argument('--activation-checkpointing',action='store_true')
    parser.add_argument('--gat-edge-chunk-size',type=int,default=0)
    args=parser.parse_args()
    config,registry=configuration(),read(args.registry)
    if args.threads is None:
        args.threads=config['threads']
    if not 1 <= args.check_epochs <= 5:
        parser.error('Technical validation budget must be 1..5 epochs')
    if args.threads!=config['threads']:
        parser.error('Use the shared thread setting')
    if args.worker:
        if args.model not in config['models'] or args.seed not in config['seeds']:
            parser.error('Unplanned model/seed')
        try:
            worker(args,config,registry)
        except Exception as error:
            atomic_json(args.output_dir/'worker_error.json',{'error':str(error)})
            raise
        return
    models,seeds=args.models.split(','),[int(s) for s in args.seeds.split(',')]
    if not models or not seeds or len(set(models))!=len(models) or len(set(seeds))!=len(seeds):
        parser.error('Nonempty unique model and seed lists required')
    if any(m not in config['models'] for m in models) or any(s not in config['seeds'] for s in seeds):
        parser.error('Unplanned model or seed')
    if args.mode=='bridge' and (args.dataset!='blte' or 'prepared_common' in registry[args.dataset]):
        parser.error('Bridge compares two implementations on BLTE, not two datasets')
    if args.mode=='reproduce' and (args.dataset!='blte' or models!=['x_moe_gnn'] or seeds!=[100]
            or args.reference_checkpoint is None or not args.reference_checkpoint.is_file()):
        parser.error('Use reproduce only for BLTE --models x_moe_gnn --seeds 100 with an existing --reference-checkpoint')
    for seed in seeds:
        checked_entry(args.dataset,registry,seed)
    if args.mode=='verify':
        print(json.dumps({'status':'shared_pipeline_and_dataset_verified','dataset':args.dataset,
            'input_dimension':registry[args.dataset]['input_dim'],
            'representation':registry[args.dataset].get('profile','dataset_native'),
            'training_performed':False,'test_evaluated':False},indent=2))
        return
    if args.output_dir is None:
        parser.error('Use an explicit new --output-dir for each dataset/mode')
    output=args.output_dir.resolve()
    if args.mode == 'train':
        bind_study(output,{'dataset':args.dataset,'mode':'train','shared_config_sha256':sha(ROOT/'common_config.json'),
            'registry_sha256':sha(args.registry),'release_manifest_sha256':sha(ROOT/'release_manifest.json'),
            'models':config['models'],'seeds':config['seeds'],'epochs':config['epochs']})
    planned=[]
    for seed in seeds:
        for model in models:
            if args.mode == 'train':
                folder, done = resolve_attempt(output,model,seed,lambda p: complete(p,args.dataset,model,seed,args))
            else:
                folder, done = output/f'{model}_seed{seed}', False
            if args.mode != 'train' and folder.exists():
                raise ValueError(f'Preserve existing technical output: {folder}')
            planned.append((seed,model,folder,done))
    for seed,model,folder,done in planned:
        if folder.exists() and not done:
            raise ValueError(f'Preserve existing incomplete/technical output: {folder}')
    for seed,model,folder,done in planned:
        if done:
            print(f'SKIP completed {args.dataset} {model} seed {seed}',flush=True)
            continue
        entry=registry[args.dataset]
        args.model,args.seed,args.output_dir=model,seed,folder
        if 'prepared_common' in entry:
            args.graph = Path(entry['prepared_common'])
            args.evaluation = args.graph / f'seed{seed}'
        else:
            args.graph=Path(entry['graph'] if args.dataset=='blte' else read(entry['study_lock'])['graph'])
            args.evaluation=Path(entry['csv']).parent if args.dataset=='blte' else Path(read(entry['study_lock'])['inputs'])
        limits = config['resource_limits']
        args.max_rss_gib=limits['max_rss_gib'];args.min_available_gib=limits['min_available_gib']
        args.timeout_seconds=min(1200,limits['timeout_seconds']) if args.mode=='bridge' else limits['timeout_seconds']
        args.activation_checkpointing=model=='gat';args.gat_edge_chunk_size=32768 if model=='gat' else 0
        extra=['--dataset',args.dataset,'--mode',args.mode,'--registry',str(args.registry.resolve()),'--seed',str(seed),
               '--check-epochs',str(args.check_epochs)]
        if args.reference_checkpoint is not None:
            extra.extend(['--reference-checkpoint',str(args.reference_checkpoint.resolve())])
        print(f'START {args.dataset} {model} seed {seed}: {args.mode}',flush=True)
        try:
            supervise(args,worker_script=Path(__file__),worker_arguments=extra,
                      run_kind={'train':'final_evaluation','validate':'development_pilot',
                                'reproduce':'development_pilot'}.get(args.mode,'resource_probe'))
        finally:
            if args.mode == 'train':
                inventory(output,[(m,s) for s in config['seeds'] for m in config['models']],
                          lambda p,m,s: complete(p,args.dataset,m,s,args))
        if args.mode=='train':
            if not complete(folder,args.dataset,model,seed,args):
                raise ValueError('Final result did not pass integrity verification')
            summarize(output,args.dataset,config,args)
    print('Requested shared-pipeline stage completed.',flush=True)


if __name__=='__main__':
    main()
