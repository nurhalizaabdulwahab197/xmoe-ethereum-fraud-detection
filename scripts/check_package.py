"""Read-only source syntax, package hashes and primary summary checks; no GPU required."""
import argparse
import ast
import csv
from common import ROOT,read,require,sha,source_check

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    n=source_check()
    for path in ROOT.rglob('*.py'):
        if any(x.startswith('.venv') for x in path.parts):continue
        ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    for name,h in read(ROOT/'provenance/PACKAGE_SHA256.json').items():
        require(sha(ROOT/name)==h,'Package file changed: '+name)
    with (ROOT/'results/tables/one_hundred_twenty_runs.csv').open() as f:rows=list(csv.DictReader(f))
    identities={(r['configuration'],r['model'],int(r['split_seed'])) for r in rows}
    expected={(c,m,s) for c in ('author','paper') for m in ('MLP','GCN','GAT','GraphSAGE','XMoE_CE','Full_XMoE') for s in range(100,110)}
    require(len(rows)==120 and identities==expected,'Primary identities')
    print('PASS_PACKAGE_STATIC:',n,'frozen source files; 120 primary identities')
    print('This does not verify missing per-run payloads or GPU execution. Run verify_audit.py locally next.')
if __name__=='__main__':main()
