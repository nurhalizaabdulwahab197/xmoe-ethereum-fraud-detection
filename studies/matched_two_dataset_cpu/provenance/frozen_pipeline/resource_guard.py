"""Guarded one-step RQ1 CPU resource probe. No validation/test evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def latest_epoch_log(path):
    """Read append-only log; do not lock the worker's replaceable progress JSON."""
    with Path(path).open('rb') as f:
        f.seek(max(0, f.seek(0, 2) - 8192))
        lines = f.read().decode('utf-8', errors='replace').splitlines()
    return next((line for line in reversed(lines) if ': epoch ' in line and 'validation Macro-F1=' in line), None)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify_model_snapshot(root):
    manifest = load(root / 'source_manifest.json')
    for name, expected in manifest['source_files_sha256'].items():
        path = root / 'frozen_models' / Path(name).name if name.startswith('src/models/') else root / 'provenance/blte_reproduce_reference.py'
        if sha(path) != expected:
            raise ValueError('Frozen model/reference changed: ' + name)
    if sha(root / 'frozen_training_step.py') != manifest['adapter_sha256']:
        raise ValueError('Frozen training adapter changed')
    return manifest


def worker(args):
    root = Path(__file__).resolve().parent
    manifest = verify_model_snapshot(root)
    graph = load(args.graph / 'graph_inputs_report.json')
    evaluation = load(args.evaluation / 'evaluation_setup_report.json')
    if graph['status'] != 'common_graph_inputs_prepared' or evaluation['status'] != 'shared_split_and_train_only_preprocessing_prepared':
        raise ValueError('Completed graph and evaluation artifacts are required')
    if sha(args.graph / 'graph_inputs_report.json') != evaluation['input_sha256']['graph_report']:
        raise ValueError('Graph/evaluation provenance mismatch')
    if 42 not in evaluation['development_initialization_seeds']:
        raise ValueError('Seed 42 is not authorized by the saved development protocol')
    paths = {'features': args.evaluation / 'features_normalized.npy',
             'train': args.evaluation / 'train_mask.npy',
             'labels': args.graph / 'labels.npy', 'edges': args.graph / 'edge_index.npy'}
    hashes = {'features': evaluation['artifact_sha256']['features_normalized.npy'],
              'train': evaluation['artifact_sha256']['train_mask.npy'],
              'labels': graph['artifact_sha256']['labels.npy'], 'edges': graph['artifact_sha256']['edge_index.npy']}
    for key, path in paths.items():
        if sha(path) != hashes[key]:
            raise ValueError('Input artifact changed: ' + key)
    print('Input hashes verified. Importing CPU model libraries...', flush=True)
    import numpy as np
    import torch
    import torch_geometric
    from frozen_models import GCN, GAT, GraphSAGE, ExplainableMixtureOfExpertsGNN
    from frozen_training_step import original_training_step
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    x_np = np.load(paths['features'], mmap_mode='r', allow_pickle=False)
    y_np = np.load(paths['labels'], mmap_mode='r', allow_pickle=False)
    edge_np = np.load(paths['edges'], mmap_mode='r', allow_pickle=False)
    mask_np = np.load(paths['train'], mmap_mode='r', allow_pickle=False)
    if x_np.shape != tuple(evaluation['normalized_feature_shape']) or x_np.dtype != np.float32:
        raise ValueError('Normalized feature shape/type differs')
    n = x_np.shape[0]
    if edge_np.shape != tuple(graph['edge_index_shape']) or edge_np.dtype != np.int64 or y_np.shape != (n,) or y_np.dtype != np.int64 or mask_np.shape != (n,):
        raise ValueError('Graph tensor shape/type differs')
    if mask_np.dtype not in (np.dtype('uint8'), np.dtype('bool')) or not np.isin(mask_np, [0, 1]).all():
        raise ValueError('Invalid training mask')
    if not np.isfinite(x_np).all() or edge_np.min() < 0 or edge_np.max() >= n:
        raise ValueError('Nonfinite features or invalid edge index')
    x = torch.from_numpy(x_np.copy()).cpu()
    y = torch.from_numpy(y_np.copy()).cpu()
    edge_index = torch.from_numpy(edge_np.copy()).cpu()
    training = torch.from_numpy(mask_np.astype(bool)).cpu()
    if int(training.sum()) != evaluation['partition_counts']['train']['total'] or (y[training] < 0).any():
        raise ValueError('Training labels/mask differ')
    counts = torch.bincount(y[training], minlength=3)
    if (counts == 0).any() or counts.tolist() != [evaluation['partition_counts']['train']['classes'][str(c)] for c in range(3)]:
        raise ValueError('Training class counts differ')
    weights = counts.sum() / counts
    weights = weights / weights.sum()
    configuration = load(root / 'reference_training_config.json')
    model_class = {'gcn': GCN, 'gat': GAT, 'graphsage': GraphSAGE, 'x_moe_gnn': ExplainableMixtureOfExpertsGNN}[args.model]
    model = model_class(x.shape[1], 3).cpu()
    if args.gat_edge_chunk_size:
        from gat_chunked_messages import enable_gat_chunked_messages
        enable_gat_chunked_messages(model, args.gat_edge_chunk_size)
    if args.activation_checkpointing:
        from gat_activation_checkpoint import enable_gat_activation_checkpointing
        enable_gat_activation_checkpointing(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['learning_rate'], weight_decay=configuration['weight_decay'])
    initial = [p.detach().clone() for p in model.parameters()]
    options = {k: v for k, v in configuration['x_moe'].items() if k != 'gradient_method'} if args.model == 'x_moe_gnn' else {}
    print(f'Loaded full graph: {n:,} nodes, {edge_index.shape[1]:,} edges. Starting ONE training-mask optimizer step for {args.model}...', flush=True)
    started = time.perf_counter()
    result = original_training_step(model, args.model, x, edge_index, y, training, optimizer, weights, **options)
    if not all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError('Nonfinite parameter or gradient')
    if not any(not torch.equal(old, new.detach()) for old, new in zip(initial, model.parameters())):
        raise ValueError('No optimizer parameter update')
    import psutil
    worker_memory = psutil.Process().memory_info()
    windows_peak = getattr(worker_memory, 'peak_wset', None)
    report = {'status': 'one_full_graph_optimizer_step_passed', 'model': args.model,
              'activation_checkpointing': args.activation_checkpointing,
              'gat_edge_chunk_size': args.gat_edge_chunk_size,
              'chunk_helper_sha256': sha(root / 'gat_chunked_messages.py') if args.gat_edge_chunk_size else None,
              'checkpoint_helper_sha256': sha(root / 'gat_activation_checkpoint.py') if args.activation_checkpointing else None,
              'worker_pid': os.getpid(), 'worker_rss_after_step_gib': worker_memory.rss / 1024**3,
              'worker_windows_peak_working_set_gib': windows_peak / 1024**3 if windows_peak is not None else None,
              'nodes': n, 'directed_edges': edge_index.shape[1], 'training_accounts': int(training.sum()),
              'training_class_counts': counts.tolist(), 'training_class_weights': weights.tolist(),
              'optimizer_steps_completed': 1, 'optimizer_step_seconds': time.perf_counter() - started,
              'training_base_loss_diagnostic': result['base_loss'], 'objective_tasks': result['task_count'],
              'seed': 42, 'device': 'cpu', 'threads': args.threads,
              'validation_mask_opened': False, 'test_mask_opened': False,
              'validation_evaluated': False, 'test_evaluated': False, 'checkpoint_saved': False,
              'experiment_training_completed': False, 'final_seeds_used': False,
              'source_commit': manifest['source_commit'], 'reference_training_config_sha256': sha(root / 'reference_training_config.json'),
              'input_sha256': hashes, 'torch': str(torch.__version__), 'torch_geometric': torch_geometric.__version__,
              'python': sys.version, 'probe_code_sha256': sha(Path(__file__))}
    temporary_result = args.output_dir / 'worker_result.tmp'
    temporary_result.write_text(json.dumps(report, indent=2))
    temporary_result.replace(args.output_dir / 'worker_result.json')
    print('One full-graph optimizer step completed. No validation/test metrics or checkpoint.', flush=True)


def sample_process_tree(monitored, known):
    """Include Windows venv redirectors' actual Python worker descendants."""
    import psutil
    try:
        members = [monitored] + monitored.children(recursive=True)
    except psutil.NoSuchProcess:
        members = list(known.values())
    total, launcher, alive = 0, 0, 0
    for member in members:
        known[member.pid] = member
        try:
            if not member.is_running():
                continue
            rss = member.memory_info().rss
        except psutil.NoSuchProcess:
            continue
        total += rss
        alive += 1
        if member.pid == monitored.pid:
            launcher = rss
    return total, launcher, alive


def stop_process(process, known=None):
    """Terminate worker descendants as well as the venv launcher."""
    import psutil
    members = dict(known or {})
    try:
        root = psutil.Process(process.pid)
        for member in [root] + root.children(recursive=True):
            members[member.pid] = member
    except psutil.NoSuchProcess:
        pass
    for member in reversed(list(members.values())):
        try:
            member.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(members.values()), timeout=5)
    for member in alive:
        try:
            member.kill()
        except psutil.NoSuchProcess:
            pass
    if alive:
        psutil.wait_procs(alive, timeout=5)
    if process.poll() is None:
        process.kill()
    process.wait(timeout=15)


def supervise(args, worker_script=None, worker_arguments=None, run_kind='resource_probe'):
    import psutil
    if args.output_dir.exists():
        raise ValueError('Output exists; preserve it and select a new output directory')
    if args.max_rss_gib <= 0 or args.min_available_gib < 0 or args.timeout_seconds <= 0 or args.threads < 1:
        raise ValueError('Invalid resource limits')
    args.output_dir.mkdir(parents=True)
    started, peak, lowest = time.monotonic(), 0, psutil.virtual_memory().available
    launcher_peak, max_members, known, monitor_error = 0, 0, {}, None
    command = [sys.executable, '-B', str(Path(worker_script or __file__).resolve()), '--worker',
               '--model', args.model, '--graph', str(args.graph.resolve()), '--evaluation', str(args.evaluation.resolve()),
               '--threads', str(args.threads), '--output-dir', str(args.output_dir.resolve())]
    if args.activation_checkpointing:
        command.append('--activation-checkpointing')
    if args.gat_edge_chunk_size:
        command.extend(['--gat-edge-chunk-size', str(args.gat_edge_chunk_size)])
    command.extend(worker_arguments or [])
    process, reason, interrupted = None, None, False
    print(f'{args.model}: guarded CPU probe. RSS limit={args.max_rss_gib} GiB; free-RAM floor={args.min_available_gib} GiB; time limit={args.timeout_seconds}s.', flush=True)
    with (args.output_dir / 'worker_console.log').open('w', encoding='utf-8') as log:
        try:
            if lowest < args.min_available_gib * 1024**3:
                reason = 'not_started_insufficient_available_ram'
            else:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                monitored = psutil.Process(process.pid)
                last_log = 0
                while process.poll() is None:
                    available = psutil.virtual_memory().available
                    lowest = min(lowest, available)
                    rss, launcher_rss, members = sample_process_tree(monitored, known)
                    peak = max(peak, rss)
                    launcher_peak = max(launcher_peak, launcher_rss)
                    max_members = max(max_members, members)
                    elapsed = time.monotonic() - started
                    if rss > args.max_rss_gib * 1024**3:
                        reason = 'stopped_memory_limit'
                    elif available < args.min_available_gib * 1024**3:
                        reason = 'stopped_low_available_ram'
                    elif elapsed > args.timeout_seconds:
                        reason = 'stopped_time_limit'
                    if reason:
                        stop_process(process, known)
                        break
                    if elapsed - last_log >= 10:
                        print(f'{args.model}: {elapsed:.0f}s elapsed; process-tree RSS={rss/1024**3:.2f} GiB; sampled peak={peak/1024**3:.2f} GiB; available RAM={available/1024**3:.2f} GiB', flush=True)
                        if run_kind in ('development_pilot', 'final_evaluation'):
                            try:
                                progress = latest_epoch_log(args.output_dir / 'worker_console.log')
                                if progress:
                                    print(progress, flush=True)
                            except OSError:
                                pass  # Live progress display is not a training dependency.
                        last_log = elapsed
                    time.sleep(0.25)
                process.wait()
        except KeyboardInterrupt:
            interrupted, reason = True, 'interrupted_by_user'
        except Exception as error:
            reason, monitor_error = 'monitor_failed', str(error)
        finally:
            if process:
                stop_process(process, known)
    worker_file = args.output_dir / 'worker_result.json'
    detail = load(worker_file) if worker_file.exists() else None
    status = reason or ('passed' if process and process.returncode == 0 and detail else 'worker_failed')
    if status == 'passed' and detail.get('worker_windows_peak_working_set_gib') is not None and detail['worker_windows_peak_working_set_gib'] > args.max_rss_gib:
        status = 'completed_but_os_peak_exceeded_limit'
    report = {'status': status, 'model': args.model, 'device': 'cpu', 'threads': args.threads,
              'gat_edge_chunk_size': args.gat_edge_chunk_size,
              'run_kind': run_kind,
              'activation_checkpointing': args.activation_checkpointing,
              'elapsed_seconds': time.monotonic() - started, 'sampled_peak_worker_rss_gib': peak / 1024**3,
              'memory_measurement': 'sum of RSS across worker launcher and recursive descendants',
              'sampled_peak_process_tree_rss_gib': peak / 1024**3,
              'sampled_peak_launcher_rss_gib': launcher_peak / 1024**3,
              'max_monitored_processes': max_members, 'monitor_error': monitor_error,
              'lowest_sampled_system_available_gib': lowest / 1024**3,
              'resource_limits': {'max_worker_rss_gib': args.max_rss_gib, 'min_system_available_gib': args.min_available_gib,
                                  'max_process_tree_rss_gib': args.max_rss_gib,
                                  'timeout_seconds': args.timeout_seconds, 'poll_seconds': 0.25},
              'worker_exit_code': process.returncode if process else None,
              'worker_result': detail,
              'validation_evaluated': bool(detail and detail.get('validation_evaluated')) or
                  (run_kind in ('development_pilot', 'final_evaluation') and (args.output_dir / 'progress.json').exists()),
              'test_evaluation_started': (args.output_dir / 'test_evaluation_started.json').exists(),
              'test_evaluated': bool(detail and detail.get('test_evaluated')), 'development_run_completed': run_kind == 'development_pilot' and status == 'passed',
              'experiment_training_completed': run_kind == 'final_evaluation' and status == 'passed',
              'final_run_completed': run_kind == 'final_evaluation' and status == 'passed',
              'checkpoint_saved': run_kind in ('development_pilot', 'final_evaluation') and any(
                  (args.output_dir / name).exists() for name in ('best_validation_checkpoint.pt','best_checkpoint.pt')),
              'probe_code_sha256': sha(Path(__file__)),
              'limitation': 'Process-tree RSS can double-count shared pages and sampling can miss transients. Resource success alone does not establish superiority; inspect the final run metrics.' if run_kind in ('development_pilot', 'final_evaluation') else 'Process-tree RSS can conservatively double-count shared pages and sampling can miss transients. Windows worker peak working set is reported separately when available. A one-step pass does not establish 300-epoch stability, runtime or model performance.'}
    (args.output_dir / 'resource_probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('status', 'model', 'elapsed_seconds', 'sampled_peak_process_tree_rss_gib', 'test_evaluated')}, indent=2))
    if status != 'passed':
        print('Inspect worker_console.log and resource_probe.json; do not raise limits or start full experiments automatically.', flush=True)
        raise SystemExit(130 if interrupted else 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['gcn', 'gat', 'graphsage', 'x_moe_gnn'], default='graphsage')
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--max-rss-gib', type=float, default=20)
    parser.add_argument('--min-available-gib', type=float, default=4)
    parser.add_argument('--timeout-seconds', type=float, default=1200)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--activation-checkpointing', action='store_true',
                        help='Recompute frozen GAT convolution activations during backward; preserve RNG and state keys')
    parser.add_argument('--gat-edge-chunk-size', type=int, default=0, help='Bound edge-message temporaries for the CPU GAT baseline; zero disables')
    args = parser.parse_args()
    if args.activation_checkpointing and args.model != 'gat':
        parser.error('Activation checkpointing is supported only for the frozen GAT baseline')
    if args.gat_edge_chunk_size < 0 or (args.gat_edge_chunk_size and args.model != 'gat'):
        parser.error('Positive edge chunks are supported only for GAT')
    if args.worker:
        try:
            worker(args)
        except Exception as error:
            (args.output_dir / 'worker_error.json').write_text(json.dumps({'status': 'worker_failed', 'error': str(error)}, indent=2))
            raise
    else:
        supervise(args)


if __name__ == '__main__':
    main()
