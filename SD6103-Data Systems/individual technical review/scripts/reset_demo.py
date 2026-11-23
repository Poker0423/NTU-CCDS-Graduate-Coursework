from common import ROOT, connect

def reset():
    # Acquire the actual write connection before changing any outputs; fail on another writer.
    with connect() as c:
        assert c.execute('SELECT count(*) FROM sales').fetchone()[0]==400000
    target=ROOT/'data/parquet/retail.parquet'
    if target.resolve().parent != (ROOT/'data/parquet').resolve():
        raise RuntimeError('Unsafe output path; refusing reset')
    if target.exists(): target.unlink()
    print('Reset PASS. Only data/parquet/retail.parquet removed; raw data, database, reference Parquet and evidence retained.')

if __name__=='__main__': reset()
