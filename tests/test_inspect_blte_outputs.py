"""Verify new-output grouping and prevent smoke/duplicate rows entering summaries."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'scripts/inspect_blte_outputs.py'
spec = importlib.util.spec_from_file_location('inspect_blte_outputs', MODULE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

class InspectOutputsTest(unittest.TestCase):
    def rows(self, family='clean'):
        return [dict(model='x_moe_gnn', protocol='strict_inductive' if family=='strict' else 'clean', seed=str(seed), epochs='300', macro_f1=str(.5+(seed-100)*.01),device='cpu') for seed in range(100,110)]
    def test_complete_ten_seed_summary(self):
        group,status,values=helper.summarize(self.rows(),'clean','metrics')[0]
        self.assertEqual(group,('x_moe_gnn','cpu'))
        self.assertEqual(status,'COMPLETE 10 SEEDS')
        self.assertAlmostEqual(values['macro_f1'][0],.545)
        self.assertEqual(values['macro_f1'][2],10)
    def test_duplicate_seed_rejected(self):
        rows=self.rows()
        with self.assertRaisesRegex(ValueError,'Duplicate'):helper.summarize(rows+[rows[0]],'clean','metrics')
    def test_lambda_groups_not_pooled(self):
        rows=[]
        for coefficient in ['0.00','0.30']:
            rows += [dict(r,x_moe_normal_phishing_penalty=coefficient) for r in self.rows()]
        self.assertEqual(len(helper.summarize(rows,'sensitivity','metrics')),2)
    def test_connectivity_fpr_requires_normal_only_stratum(self):
        rows=[dict(r,variant_id='a0_full',test_isolated_nodes='3',isolated_normal_nodes='3',isolated_normal_recall='0.9') for r in self.rows()]
        values=helper.summarize(rows,'ablation','connectivity')[0][2]
        self.assertAlmostEqual(values['isolated_normal_fpr_normal_only_stratum'][0],.1)
        for r in rows:r['isolated_normal_nodes']='2'
        values=helper.summarize(rows,'ablation','connectivity')[0][2]
        self.assertNotIn('isolated_normal_fpr_normal_only_stratum',values)
    def test_raw_reader_excludes_predictions_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=self.rows('strict')
            with (root/'full.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            (root/'full_summary.csv').write_text('metric,mean\nmacro_f1,.5\n')
            (root/'predictions.csv').write_text('node_index,true_label,predicted_label\n1,0,0\n')
            self.assertEqual(len(helper.collect(root,'strict')),10)
    def test_smoke_csv_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=self.rows();rows[0]['epochs']='2'
            with (root/'smoke.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'Non-300'):helper.collect(root,'clean')

if __name__ == '__main__':unittest.main()
