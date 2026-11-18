import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from attempts import resolve_attempt, bind_study
from shared_engine import reference
from bridge import instrumented, compare_state
from validation_only import run_validation, compare_checkpoint
from utility import read

ROOT = Path(__file__).resolve().parent


class ValidationTests(unittest.TestCase):
    def test_checkpoint_expert_rename_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old=root/'old.pt';new=root/'new.pt'
            torch.save({'input_dim':23,'state_dict':{'v8_expert.weight':torch.tensor([1.,2.])}},old)
            torch.save({'input_dim':23,'state_dict':{'conflict_aware_expert.weight':torch.tensor([1.,2.])}},new)
            result=compare_checkpoint(new,old)
            self.assertTrue(result['exact_equal'])
            self.assertEqual(result['maximum_absolute_difference'],0.)
            torch.save({'input_dim':23,'state_dict':{'conflict_aware_expert.weight':torch.tensor([1.,3.])}},new)
            with self.assertRaisesRegex(ValueError,'not exactly equal'):
                compare_checkpoint(new,old)

    def fixture(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        x = torch.randn(60, 5)
        edge = torch.stack([torch.arange(60), torch.roll(torch.arange(60), 1)])
        y = torch.tensor([0, 1, 2] * 20)
        train = torch.arange(60) < 42
        validation = (torch.arange(60) >= 42) & (torch.arange(60) < 51)
        test = torch.arange(60) >= 51
        return (x, edge, y, train, validation, test)

    def test_validation_matches_reference_prefix_and_does_not_score_test(self):
        values = self.fixture()
        config = read(ROOT / 'common_config.json')
        for name in config['models']:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as tmp:
                ref = reference()
                fn, ns = instrumented(ref)
                ns['clean_data'] = lambda *a: values
                expected = fn(None, Path('.'), name, 'clean', 100, 3, torch.device('cpu'), True,
                              **config['training_options'])
                ref.score = lambda *a: (_ for _ in ()).throw(AssertionError('Test scorer called'))
                folder = Path(tmp)
                result = run_validation(ref, name, 100, values, folder, 3,
                                        [f'f{i}' for i in range(5)], config['training_options'])
                saved = torch.load(folder/'best_validation_checkpoint.pt',weights_only=True)
                compare_state(saved['state_dict'], expected['selected_state'])
                self.assertEqual(result['best_validation_metrics']['macro_f1'], expected['best_validation_f1'])
                self.assertTrue(result['checkpoint_reload_exact'])
                self.assertFalse(result['test_evaluated'])
                self.assertFalse((folder/'test_predictions.csv').exists())

    def test_changing_only_test_labels_cannot_affect_validation_training(self):
        values = self.fixture()
        changed = list(values)
        changed[2] = values[2].clone()
        changed[2][values[5]] = (changed[2][values[5]] + 1) % 3
        config = read(ROOT/'common_config.json')
        with tempfile.TemporaryDirectory() as tmp:
            states = []
            for i, tensors in enumerate((values, changed)):
                folder = Path(tmp)/str(i)
                folder.mkdir()
                run_validation(reference(),'x_moe_gnn',100,tensors,folder,3,[f'f{i}' for i in range(5)],config['training_options'])
                states.append(torch.load(folder/'best_validation_checkpoint.pt',weights_only=True)['state_dict'])
            self.assertEqual(compare_state(*states,rtol=0,atol=0),0.)


class AttemptTests(unittest.TestCase):
    def test_preserve_failure_then_skip_verified_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root/'gcn_seed100'
            original.mkdir()
            (original/'failure.log').write_text('preserve')
            folder, done = resolve_attempt(root,'gcn',100,lambda p:(p/'complete').exists())
            self.assertFalse(done)
            self.assertEqual(folder.name,'gcn_seed100_attempt002')
            folder.mkdir()
            (folder/'complete').touch()
            selected, done = resolve_attempt(root,'gcn',100,lambda p:(p/'complete').exists())
            self.assertTrue(done)
            self.assertEqual(selected,folder)
            self.assertEqual((original/'failure.log').read_text(),'preserve')

    def test_multiple_completions_are_not_selected_by_performance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in ('gcn_seed100','gcn_seed100_attempt002'):
                (root/n).mkdir()
            with self.assertRaisesRegex(ValueError,'Multiple verified'):
                resolve_attempt(root,'gcn',100,lambda p:True)

    def test_study_identity_rejects_changed_dataset_or_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'output'
            bind_study(root,{'dataset':'blte','config':'a'})
            bind_study(root,{'dataset':'blte','config':'a'})
            with self.assertRaisesRegex(ValueError,'another dataset'):
                bind_study(root,{'dataset':'rq1','config':'a'})

    def test_supervisor_reports_actual_checkpoint_name(self):
        # Exercise supervisor outcome metadata without launching any ML training.
        import resource_guard
        import argparse
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper = root/'helper.py'
            helper.write_text("import pathlib,sys,json\np=pathlib.Path(sys.argv[sys.argv.index('--output-dir')+1]); (p/'best_checkpoint.pt').write_text('synthetic'); (p/'worker_result.json').write_text(json.dumps({'validation_evaluated':True,'test_evaluated':True}))\n")
            args = argparse.Namespace(output_dir=root/'out',model='mlp',graph=root,evaluation=root,
                threads=1,activation_checkpointing=False,gat_edge_chunk_size=0,
                max_rss_gib=20,min_available_gib=0,timeout_seconds=30)
            resource_guard.supervise(args,worker_script=helper,run_kind='final_evaluation')
            report = read(args.output_dir/'resource_probe.json')
            self.assertTrue(report['checkpoint_saved'])
            self.assertTrue(report['validation_evaluated'])
            self.assertTrue(report['test_evaluated'])


if __name__ == '__main__':
    unittest.main()
