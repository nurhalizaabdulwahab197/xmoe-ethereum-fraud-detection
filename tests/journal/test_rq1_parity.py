"""Scientific checks for RQ1 parity; synthetic scores are never research evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'experiments/rq1_parity'
sys.path.insert(0,str(HERE))
from protocol import make_views, masks_for
from plan import matrix, run_id, VARIANTS
from train import train, ref
from report import audit_result, flip_p, holm, summarize


def fixture(context=True):
    rng=np.random.default_rng(42)
    n=24 if context else 18
    y=np.array([0,1,2]*6+([-1]*6 if context else []),dtype=np.int64)
    masks=[]
    for start,end in [(0,12),(12,15),(15,18)]:
        m=np.zeros(n,dtype=bool);m[start:end]=True;masks.append(torch.from_numpy(m))
    # Directed edges, duplicate links, self-loop, query-query, validation-test,
    # context-train and context-query relationships all included deliberately.
    edges=torch.tensor([[0,1,2,3,12,15,13,16,12,15,0,0,5,17],
                        [1,2,0,3,0,0,14,17,15,12,12,15,0,17]],dtype=torch.long)
    if context:edges=torch.cat([edges,torch.tensor([[18,0,19,15],[0,18,15,20]])],dim=1)
    return (torch.from_numpy(rng.normal(size=(n,5)).astype(np.float32)),edges,torch.from_numpy(y),*masks)


class ParityTests(unittest.TestCase):
    def test_matrix(self):
        rows=matrix();self.assertEqual(len(rows),230)
        self.assertEqual(len({run_id(r) for r in rows}),230)
        self.assertEqual(len(matrix(gpu=True)),270)
        self.assertEqual({r['model'] for r in matrix(gpu=True) if r['device']=='cuda'}, {'gcn','gat','graphsage','x_moe_gnn'})

    def test_visibility_and_context(self):
        x,e,y,t,v,q=fixture()
        features,edges,audit=make_views(x,e,y,(t,v,q))
        for i,keep in enumerate([t,t|v,t|q]):
            self.assertTrue(torch.equal(features[i][keep],x[keep]))
            self.assertEqual(torch.count_nonzero(features[i][~keep]),0)
            self.assertTrue(keep[edges[i]].all())
        self.assertTrue(all(a['unlabeled_context_edges']==0 for a in audit.values()))
        # Original edge order, duplicates and direction are preserved.
        self.assertEqual(edges[0].tolist(),[[0,1,2,3,5],[1,2,0,3,0]])
        with self.assertRaises(ValueError):make_views(x,e,y,(t,t,q))

    def test_split(self):
        y=np.array([0,1,2]*100+[-1]*200)
        a=masks_for(y,100);b=masks_for(y,100)
        self.assertTrue(all(np.array_equal(x,z) for x,z in zip(a,b)))
        self.assertTrue(np.array_equal(sum(m.astype(int) for m in a),y>=0))
        self.assertTrue(all(not m[y<0].any() for m in a))

    def test_sign_flip_and_holm(self):
        self.assertEqual(flip_p([1]*10),2/1024)
        self.assertEqual(flip_p([0]*10),1)
        rows=holm([{'raw_p':.01},{'raw_p':.04},{'raw_p':.03}])
        self.assertEqual([round(r['holm_adjusted_p'],2) for r in rows],[.03,.06,.06])

    def test_incomplete_summary_is_not_approved(self):
        with tempfile.TemporaryDirectory() as folder:
            status=summarize(folder)
            self.assertEqual(status['status'],'incomplete')
            self.assertFalse(status['cpu_tables_ready'])
            self.assertEqual(len(status['missing']),230)
            self.assertFalse((Path(folder)/'tables/paired_statistics.csv').exists())

    def test_complete_report_with_mocked_run_audit(self):
        # Tests report mechanics only; all fabricated rows stay in a temp directory.
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            records={}
            for row in matrix():
                folder=Path(directory)/'full'/run_id(row)
                folder.mkdir(parents=True)
                (folder/'result.json').write_text('{"synthetic_report_fixture":true}')
                records[str(folder)]={**row,'macro_f1':.5+.001*(row['seed']-100),
                    'phishing_f1':.4,'phishing_ap':.3,'scamming_f1':.6,'scamming_ap':.5,
                    'output_sha256':{'test_predictions.csv':'synthetic-same-control-'+str(row['seed'])}}
            with patch('report.audit_result',side_effect=lambda folder,*args:records[str(folder)]):
                status=summarize(directory)
            self.assertTrue(status['cpu_tables_ready'])
            self.assertFalse(status['gpu_parity_completed'])
            import pandas as pd
            stats=pd.read_csv(Path(directory)/'tables/paired_statistics.csv')
            self.assertEqual(len(stats),24)
            self.assertTrue((stats.holm_adjusted_p==1).all())

    def test_every_configuration_trains_and_audits(self):
        values=fixture()
        with tempfile.TemporaryDirectory() as directory:
            for row in matrix([100]):
                folder=Path(directory)/run_id(row)
                result=train(values,{'synthetic':True},row,folder,epochs=2,threads=2)
                audit_result(folder,row,2)
                self.assertTrue(np.isfinite(result['macro_f1']))
                self.assertEqual(result['test_nodes'],3)
                self.assertEqual(result['provenance']['input'],{'synthetic':True})
            row=matrix([100])[0];folder=Path(directory)/run_id(row)
            (folder/'test_predictions.csv').write_text('changed')
            with self.assertRaises(ValueError):audit_result(folder,row,2)


if __name__=='__main__':unittest.main()
