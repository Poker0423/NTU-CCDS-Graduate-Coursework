from common import ROOT, connect, run_file

def build():
    (ROOT/'database').mkdir(exist_ok=True)
    (ROOT/'data/parquet').mkdir(parents=True,exist_ok=True)
    with connect() as c:
        run_file(c,'sql/00_initialize.sql')
        print('Persistent database initialized:',c.execute('SELECT count(*) FROM sales').fetchone()[0])
    with connect(read_only=True) as c:
        assert c.execute('SELECT count(*) FROM sales').fetchone()[0]==400000
    print('Closed and reopened database: PASS')

if __name__=='__main__': build()
