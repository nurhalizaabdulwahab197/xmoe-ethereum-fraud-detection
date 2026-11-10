"""RQ1 journal parity: preflight -> isolated smoke runs -> full 300-epoch matrix."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'provenance/frozen_pipeline'))
from data import load, inspect_graph, sha
from plan import matrix, run_id


def find_graph():
    roots=[Path.home()/'Documents',Path.home()/'Desktop']
    candidates=[]
    for base in roots:
        if not base.exists():continue
        for current, directories, files in os.walk(base):
            directories[:]=[d for d in directories if not d.startswith('.') and d not in ['node_modules','__pycache__','site-packages']]
            if 'features_raw.npy' in files:
                candidate=Path(current)
                try:inspect_graph(candidate)
                except (OSError,ValueError,KeyError):continue
                candidates.append(candidate)
    if not candidates:
        raise ValueError('Original RQ1 features_raw.npy, labels.npy, candidate_labeled_mask.npy and edge_index.npy were not found. The demo inference ZIP is insufficient for new training. Supply --graph PATH to the original prepared graph folder.')
    return sorted(candidates,key=lambda p:len(str(p)))[0]


def guarded(command, log, max_rss_gib, min_available_gib):
    import psutil
    started=time.monotonic();peak=0
    with Path(log).open('w',encoding='utf-8') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
        error=None
        try:
            process=psutil.Process(child.pid)
            while child.poll() is None:
                try:rss=process.memory_info().rss
                except psutil.NoSuchProcess:break
                peak=max(peak,rss)
                available=psutil.virtual_memory().available
                if rss > max_rss_gib*1024**3 or available < min_available_gib*1024**3:
                    error='RAM guard stopped training; use a larger machine, never reduce the dataset silently.';break
                if time.monotonic()-started > 21600:
                    error='Six-hour per-run timeout exceeded.';break
                time.sleep(.5)
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
        finally:
            if error and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        code=child.wait()
    profile={'peak_rss_bytes':peak,'elapsed_seconds':time.monotonic()-started,'returncode':code,'guard_failure':error}
    Path(str(log)+'.resource.json').write_text(json.dumps(profile,indent=2)+'\n')
    if error or code:
        raise RuntimeError((error or 'Worker failed')+'; inspect '+str(log))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['preflight','smoke','full','summarize'],default='preflight')
    p.add_argument('--graph',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'results/new_runs/rq1_journal_parity')
    p.add_argument('--threads',type=int,default=24)
    p.add_argument('--include-gpu',action='store_true')
    p.add_argument('--max-rss-gib',type=float,default=20)
    p.add_argument('--min-available-gib',type=float,default=4)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:
        job=json.loads(args.worker.read_text())
        from train import train
        values,metadata=load(Path(job['graph']),job['row']['seed'])
        train(values,metadata,job['row'],Path(job['folder']),job['epochs'],args.threads)
        return
    from train import code_inventory
    from utility import verify_code
    verify_code(ROOT/'provenance/frozen_pipeline')
    from report import audit_result, summarize
    if args.stage=='summarize':
        summarize(args.output,args.include_gpu);return
    if args.threads<1 or args.max_rss_gib<=0 or args.min_available_gib<0:
        raise ValueError('Invalid resource configuration.')
    if args.graph is None:
        print('Searching Documents and Desktop for verified original RQ1 raw graph files...',flush=True)
    graph=args.graph.resolve() if args.graph else find_graph()
    inspect_graph(graph)
    print('RQ1_GRAPH_VERIFIED: '+str(graph),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    expected={'source_sha256':code_inventory(),'graph':str(graph),'threads':args.threads,'gpu_included':args.include_gpu,
              'runtime_versions': {name:importlib.metadata.version(name) for name in ['torch','torch-geometric','numpy','pandas','scikit-learn','scipy','psutil']},
              'python_version':sys.version.split()[0],
              'epochs':300,'seeds':list(range(100,110)),'exploratory':True,
              'strict_context_policy':'Exclude all unlabeled context nodes and incident edges at every strict stage.',
              'strict_feature_scope':'Same frozen full-observation features as clean; not temporal or feature-recomputation isolation.'}
    lock=args.output/'experiment_lock.json'
    if lock.exists() and json.loads(lock.read_text())!=expected:
        raise ValueError('Experiment configuration or source differs; use a new --output. Existing results preserved.')
    lock.write_text(json.dumps(expected,indent=2)+'\n')
    jobs=matrix([100] if args.stage=='smoke' else range(100,110),args.include_gpu)
    if args.include_gpu:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable.')
    if args.stage=='preflight':
        verified=[]
        for seed in range(100,110):
            values,meta=load(graph,seed)
            verified.append({'seed':seed,'train_nodes':int(values[3].sum()),'validation_nodes':int(values[4].sum()),'test_nodes':int(values[5].sum())})
            del values
            print('PASS exact raw data, split and train-only scaler: seed '+str(seed),flush=True)
        (args.output/'preflight.json').write_text(json.dumps({'passed':True,'experiment_lock_sha256':sha(lock),'seeds':verified,'training_performed':False},indent=2)+'\n')
        print('PREFLIGHT PASSED. Next: same command with --stage smoke.');return
    preflight=args.output/'preflight.json'
    if not preflight.exists() or json.loads(preflight.read_text()).get('experiment_lock_sha256')!=sha(lock):
        raise ValueError('Run --stage preflight first for these exact source files/configuration.')
    if args.stage=='full':
        for row in matrix([100],args.include_gpu):
            audit_result(args.output/'smoke'/run_id(row),row,1,expected['source_sha256'])
    for number,row in enumerate(jobs,1):
        folder=args.output/args.stage/run_id(row)
        epochs=1 if args.stage=='smoke' else 300
        if (folder/'result.json').exists():
            audit_result(folder,row,epochs,expected['source_sha256']);print('VERIFIED RESUME '+run_id(row),flush=True);continue
        if folder.exists():
            raise ValueError('An interrupted/failed run exists: '+str(folder)+'. Preserve its log and move the folder aside before retrying.')
        job_dir=args.output/'jobs'/args.stage;job_dir.mkdir(parents=True,exist_ok=True)
        key=run_id(row).replace('/','__');job=job_dir/(key+'.json')
        job.write_text(json.dumps({'graph':str(graph),'row':row,'folder':str(folder.resolve()),'epochs':epochs})+'\n')
        print(f'RUN {number}/{len(jobs)} '+run_id(row),flush=True)
        guarded([sys.executable,str(Path(__file__).resolve()),'--worker',str(job.resolve()),'--threads',str(args.threads)],
                job_dir/(key+'.log'),args.max_rss_gib,args.min_available_gib)
        audit_result(folder,row,epochs,expected['source_sha256'])
    if args.stage=='full':summarize(args.output,args.include_gpu)
    else:print('SMOKE PASSED. These one-epoch scores are not manuscript results. Next: --stage full.')


if __name__=='__main__':main()
