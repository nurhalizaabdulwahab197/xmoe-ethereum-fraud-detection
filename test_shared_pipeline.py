import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from bridge import run_bridge
from shared_engine import reference, run_reference
from utility import read, sha


class SharedTests(unittest.TestCase):
    def fixture(self,root):
        rng=np.random.default_rng(7)
        frame=pd.DataFrame(rng.normal(size=(60,23)),columns=[f'f{i}' for i in range(23)])
        y=torch.tensor([0,1,2]*20)
        edge=torch.stack([torch.arange(40),torch.roll(torch.arange(40),-1)])
        frame['Address']=[f'0x{i:040x}' for i in range(60)]
        frame['Scam']=0;frame['attack']='source';frame['attack_encoded']=y.numpy()
        path=root/'input.csv';frame.to_csv(path,index=False)
        graph=root/'graph.pt';torch.save(Data(x=torch.randn(60,23),y=y,edge_index=edge),graph)
        return {'blte':{'csv':str(path),'graph':str(graph),'csv_sha256':sha(path),'graph_sha256':sha(graph),
            'input_dim':23,'label_policy':'synthetic only'}}

    def test_bridge_all_models_no_test_scoring(self):
        torch.set_num_threads(1)
        config=read(Path(__file__).with_name('common_config.json'))
        real_reference=reference
        def forbidden_reference():
            ref=real_reference()
            ref.score=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('Bridge invoked test scoring'))
            return ref
        with tempfile.TemporaryDirectory() as tmp:
            registry=self.fixture(Path(tmp))
            for model in config['models']:
                with self.subTest(model=model), patch('bridge.reference',side_effect=forbidden_reference):
                    result=run_bridge(model,100,registry,config)
                    self.assertEqual(result['status'],'real_blte_pre_test_bridge_passed')
                    self.assertFalse(result['test_evaluated'])
                    self.assertEqual(len(result['epoch_checks']),3)

    def test_input_dimension_checkpoint_metadata_is_not_hardcoded_five(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registry=self.fixture(root)
            ref=reference()
            from adapters import load_dataset
            values,meta=load_dataset('blte',registry,100,ref)
            output=root/'result';output.mkdir()
            run_reference(ref,'mlp',100,values,output,epochs=1,feature_names=meta['feature_names'])
            # Only this test's own generated checkpoint is read with NumPy metric scalars.
            checkpoint=torch.load(output/'best_checkpoint.pt',weights_only=False,map_location='cpu')
            self.assertEqual(checkpoint['input_dim'],23)
            self.assertEqual(checkpoint['feature_names'],meta['feature_names'])

    def test_bridge_detects_changed_adapter_input(self):
        torch.set_num_threads(1)
        config=read(Path(__file__).with_name('common_config.json'))
        from adapters import load_dataset as real_loader
        def changed(*args):
            values,meta=real_loader(*args)
            values=list(values);values[0]=values[0]+1
            return values,meta
        with tempfile.TemporaryDirectory() as tmp:
            registry=self.fixture(Path(tmp))
            with patch('bridge.load_dataset',side_effect=changed),self.assertRaises(AssertionError):
                run_bridge('mlp',100,registry,config)


if __name__=='__main__':
    unittest.main()
