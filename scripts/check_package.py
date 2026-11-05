"""Check publication/source hashes and audit all 100 saved result sets."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utility import read,sha,verify_code
from audit_results import audit

def main():
    verify_code(ROOT)
    manifest=read(ROOT/'PACKAGE_MANIFEST.json')
    for name,expected in manifest['files_sha256'].items():
        if sha(ROOT/name)!=expected:
            raise ValueError('Publication file changed: '+name)
    for dataset in ['blte','rq1']:
        result=audit(ROOT/'results/reference'/dataset,dataset,published=True)
        print(json.dumps(result))
    print('PASS: publication inventory and all 100 saved runs; no training or model inference')

if __name__=='__main__':
    main()
