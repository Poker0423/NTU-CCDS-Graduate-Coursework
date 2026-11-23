import argparse
from common import ROOT, connect

def preflight():
    parser=argparse.ArgumentParser(); parser.add_argument('--check-reset',action='store_true'); args=parser.parse_args()
    for f in ['data/raw/sales.csv','data/parquet/retail_reference.parquet','sql/session.cli','presentation/index.html']:
        if not (ROOT/f).is_file(): raise FileNotFoundError(f+' missing; run setup.ps1')
    with connect() as c:
        assert c.execute('SELECT count(*) FROM sales').fetchone()==(400000,)
        assert c.execute('SELECT version()').fetchone()==('v2.0.0',)
    if args.check_reset and (ROOT/'data/parquet/retail.parquet').exists():
        raise RuntimeError('Run reset_demo.ps1 before recording')
    print('Preflight PASS. Database available; no handle retained. Offline recording ready.')

if __name__=='__main__': preflight()
