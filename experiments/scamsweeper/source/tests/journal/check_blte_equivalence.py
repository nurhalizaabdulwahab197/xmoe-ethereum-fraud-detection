import importlib.util,sys,tempfile,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/journal'))
from test_rq1_parity import fixture,train,ref
from plan import matrix,run_id
sys.modules['runner']=ref

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
legacy=module('_legacy_blte_ablation',root/'tests/journal/blte_reference/ablation.py')
strict=module('_legacy_blte_strict',root/'tests/journal/blte_reference/strict_inductive.py')
values=fixture(context=False)
ref.clean_data=lambda *args: values
rows=[r for r in matrix([100]) if r['family'] in ['clean','strict','ablation']]
checks=[]
with tempfile.TemporaryDirectory() as tmp:
 for row in rows:
  actual=train(values,{'synthetic':True},row,Path(tmp)/run_id(row),epochs=3,threads=2)
  captured=[]
  original_score=ref.score
  def capture_score(y, probability):
   captured.append(np.asarray(probability).copy())
   return original_score(y, probability)
  ref.score=capture_score
  if row['family']=='ablation':
   wanted=legacy.run_variant_once(None,Path('unused'),row['variant'],'clean',100,3,ref.torch.device('cpu'),True,.25,.5,1.,.3)
  elif row['family']=='clean':
   from shared_engine import run_reference
   out=Path(tmp)/('reference_'+row['model']);out.mkdir()
   wanted=run_reference(ref,row['model'],100,values,out,epochs=3,feature_names=['f'+str(i) for i in range(5)])
  else:
   wanted=strict.run_once(None,Path('unused'),row['model'],100,3,ref.torch.device('cpu'),True,'weighted_ce',2.,False,.25,.5,1.,.3,None)
  ref.score=original_score
  import pandas as pd
  actual_prob=pd.read_csv(Path(tmp)/run_id(row)/'test_predictions.csv')[['normal_probability','phishing_probability','scamming_probability']].to_numpy()
  np.testing.assert_allclose(actual_prob,captured[0],atol=1e-7,rtol=1e-6)
  probability_error=float(np.abs(actual_prob-captured[0]).max())
  for key in ['accuracy','macro_f1','phishing_ap','scamming_ap','best_validation_macro_f1']:
   assert abs(actual[key]-wanted[key])<1e-10,(row,key,actual[key],wanted[key])
  checks.append({'family':row['family'],'model':row['model'],'variant':row['variant'],'passed':True,'maximum_probability_error':probability_error})
print(json.dumps({'passed':True,'synthetic_only':True,'checks':checks},indent=2))
