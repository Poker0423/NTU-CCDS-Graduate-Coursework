"""Execute assertions and every recording SQL on installed DuckDB, never mock results."""
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from decimal import Decimal
from datetime import datetime
from zoneinfo import ZoneInfo
from common import ROOT, DB, connect, run_file, enter

def validate():
    enter(); t0=time.perf_counter()
    out=ROOT/'verification'; out.mkdir(exist_ok=True)
    checks=[]; log=[]; evidence={}
    def check(name,condition,detail=''):
        checks.append((name,'PASS' if condition else 'FAILED',str(detail)))
        if not condition: raise AssertionError(name+': '+str(detail))
    def cli(args,stdin=None):
        p=subprocess.run([str(ROOT/'tools/duckdb.exe'),'-no-agent','-no-init','-bail',*args],
            input=stdin,text=True,encoding='utf-8',capture_output=True,cwd=ROOT)
        if p.returncode: raise RuntimeError(p.stderr+'\n'+p.stdout)
        return p.stdout
    try:
        import duckdb
        check('Python DuckDB pinned',duckdb.__version__=='2.0.0',duckdb.__version__)
        version=cli(['--version']).strip()
        check('Official CLI version matches Python','v2.0.0' in version,version)
        archive=ROOT/'tools/downloads/duckdb-cli-windows-amd64.tar.gz'
        check('Official CLI archive SHA256',hashlib.sha256(archive.read_bytes()).hexdigest()==
            '91663c2f36c2a334b0e5e639006d8c29c29e013af0c477d80bb931df68870312')
        m=json.loads((ROOT/'data/dataset_manifest.json').read_text(encoding='utf-8'))
        for name,meta in m['files'].items():
            p=ROOT/'data/raw'/name
            check('Raw data hash: '+name,hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'])
        # Oracle recomputed independently from serialized CSV integer cents.
        with (ROOT/'data/raw/customers.csv').open(newline='',encoding='utf-8') as f:
            regions={int(r['customer_id']):r['region'] for r in csv.DictReader(f)}
        with (ROOT/'data/raw/products.csv').open(newline='',encoding='utf-8') as f:
            categories={int(r['product_id']):r['category'] for r in csv.DictReader(f)}
        oracle=defaultdict(lambda:[0,0]); dec=defaultdict(int); total=0; rowcount=0
        with (ROOT/'data/raw/sales.csv').open(newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                cents=int(Decimal(r['unit_price'])*100)*int(r['quantity'])-int(Decimal(r['discount_amount'])*100)
                key=(regions[int(r['customer_id'])],categories[int(r['product_id'])])
                oracle[key][0]+=1; oracle[key][1]+=cents; total+=cents; rowcount+=1
                if r['sale_date']>='2025-12-01' and key[1]=='Electronics': dec[key[0]]+=cents
        check('Independent CSV oracle vs generator totals',total==m['total_revenue_cents'] and rowcount==400000,total)
        with connect() as c:
            check('SELECT 42',c.execute('SELECT 42').fetchone()==(42,))
            schemas={name:c.execute('DESCRIBE '+name).fetchall() for name in ['sales','customers','products']}
            evidence['schemas']=schemas
            for name,n in [('sales',400000),('customers',15000),('products',200)]:
                check('Persistent table count '+name,c.execute('SELECT count(*) FROM '+name).fetchone()[0]==n,n)
            st={r[0]:r[1] for r in schemas['sales']}
            check('Explicit dates and monetary types',st['sale_date']=='DATE' and st['unit_price']=='DECIMAL(10,2)' and st['discount_amount']=='DECIMAL(10,2)',st)
            check('Relational keys unique',all(c.execute(f'SELECT count(*)=count(DISTINCT {col}) FROM {tab}').fetchone()[0]
                  for tab,col in [('sales','sale_id'),('customers','customer_id'),('products','product_id')]))
            check('No orphan keys',c.execute('SELECT count(*) FROM sales s LEFT JOIN customers c USING(customer_id) LEFT JOIN products p USING(product_id) WHERE c.customer_id IS NULL OR p.product_id IS NULL').fetchone()[0]==0)
            check('Revenue nonnegative',c.execute('SELECT min(quantity*unit_price-discount_amount)>=0 FROM sales').fetchone()[0])
            c.execute('CREATE TABLE IF NOT EXISTS verification_roundtrip(i INTEGER)')
            c.execute('DELETE FROM verification_roundtrip'); c.execute('INSERT INTO verification_roundtrip VALUES (42)')
        # No Python handle remains when the CLI writes the same file.
        text=cli([str(DB),'-c','SELECT * FROM verification_roundtrip; INSERT INTO verification_roundtrip VALUES (43);'])
        check('Python-written database read and modified by CLI','42' in text)
        with connect() as c:
            check('CLI-written row persists and Python reopens',c.execute('SELECT sum(i) FROM verification_roundtrip').fetchone()==(85,))
            c.execute('DROP TABLE verification_roundtrip')
            c.execute('BEGIN'); c.execute('CREATE TABLE verification_rollback(i INTEGER)'); c.execute('ROLLBACK')
            check('Transaction rollback',c.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='verification_rollback'").fetchone()==(0,))
            for filename in ['01_intro','02_csv_analytics','03_parquet','04_explain','05_analyze','06_join_plan_optional']:
                results=run_file(c,'sql/'+filename+'.sql')
                log.append('\n### '+filename+'\n')
                for desc,rows in results:
                    log.append(str([d[0] for d in desc])+'\n')
                    for row in rows: log.append(' | '.join(map(str,row))+'\n')
                check('Demo SQL executes via Python: '+filename,True)
                if filename in ['04_explain','05_analyze','06_join_plan_optional']:
                    plan='\n'.join(str(row[-1]) for desc,rows in results for row in rows)
                    (out/(filename+'_plan.txt')).write_text(plan,encoding='utf-8')
                    evidence[filename]=plan
            actual=c.execute('SELECT region,category,count(*),sum(revenue) FROM retail GROUP BY region,category').fetchall()
            check('All 20 groups equal independent integer-cent CSV oracle',len(actual)==20 and all([n,int(v*100)]==oracle[(r,cat)] for r,cat,n,v in actual))
            check('JOIN preserves cardinality and exact revenue',c.execute('SELECT count(*),sum(revenue) FROM retail').fetchone()==(400000,Decimal(total)/100))
            actual_dec=c.execute("SELECT region,sum(revenue) FROM read_parquet('data/parquet/retail.parquet') WHERE sale_date>=DATE '2025-12-01' AND sale_date<DATE '2026-01-01' AND category='Electronics' GROUP BY region").fetchall()
            check('Parquet filtered result equals CSV oracle',all(int(v*100)==dec[r] for r,v in actual_dec) and len(actual_dec)==4)
            for path in ['data/parquet/retail.parquet','data/parquet/retail_reference.parquet']:
                diff=c.execute(f"SELECT count(*) FROM ((SELECT * FROM retail_base EXCEPT ALL SELECT * FROM read_parquet('{path}')) UNION ALL (SELECT * FROM read_parquet('{path}') EXCEPT ALL SELECT * FROM retail_base))").fetchone()[0]
                check('Full multiset equality '+path,diff==0)
            columns="columns={'sale_id':'INTEGER','sale_date':'DATE','customer_id':'INTEGER','product_id':'INTEGER','quantity':'SMALLINT','unit_price':'DECIMAL(10,2)','discount_amount':'DECIMAL(10,2)','region':'VARCHAR','category':'VARCHAR','revenue':'DECIMAL(12,2)'}"
            check('Like-for-like CSV/Parquet full equality',c.execute(f"SELECT count(*) FROM ((SELECT * FROM retail_base EXCEPT ALL SELECT * FROM read_csv('data/raw/retail_flat.csv',header=true,{columns})) UNION ALL (SELECT * FROM read_csv('data/raw/retail_flat.csv',header=true,{columns}) EXCEPT ALL SELECT * FROM retail_base))").fetchone()==(0,))
            plan=evidence['04_explain']; prof=evidence['05_analyze']; jp=evidence['06_join_plan_optional']
            check('Parquet scan node', 'READ_PARQUET' in plan.upper())
            check('Projection pushed into scan','Projections:' in plan and 'region' in plan and 'revenue' in plan)
            check('Date and category filters pushed into scan','Filters:' in plan and '2025-12-01' in plan and 'Electronics' in plan)
            check('Hash grouping and actual cardinalities','HASH_GROUP_BY' in prof.upper() or 'Hash Group By' in prof)
            check('Actual JOIN operator','HASH_JOIN' in jp.upper() or 'Hash Join' in jp)
            # Metadata eligibility is separate from the actual profiler row-group counter.
            meta=c.execute("SELECT row_group_id,row_group_num_rows,stats_min_value,stats_max_value FROM parquet_metadata('data/parquet/retail.parquet') WHERE path_in_schema='sale_date' ORDER BY row_group_id").fetchall()
            eligible=[r for r in meta if r[3]>='2025-12-01' and r[2]<'2026-01-01']
            check('Multiple date-ordered row groups and pruning opportunity',len(meta)==25 and len(eligible)==4,(len(meta),len(eligible)))
            evidence['row_groups']={'total':len(meta),'date_eligible':len(eligible),'metadata':meta}
            evidence['top5']=c.execute('SELECT region,category,count(*),sum(revenue) FROM retail GROUP BY region,category ORDER BY 4 DESC,1,2 LIMIT 5').fetchall()
            evidence['december']=actual_dec
            evidence['total_revenue']=str(Decimal(total)/100)
            evidence['sizes']={name:(ROOT/name).stat().st_size for name in ['data/raw/sales.csv','data/raw/customers.csv','data/raw/products.csv','data/raw/retail_flat.csv','data/parquet/retail_reference.parquet','data/parquet/retail.parquet']}
            evidence['csv_to_parquet_size_ratio']=evidence['sizes']['data/raw/retail_flat.csv']/evidence['sizes']['data/parquet/retail.parquet']
        # Complete CLI dry-run, twice, catches dot commands, paths and overwrite/repetition errors.
        batch='\n'.join('.read sql/'+x for x in ['session.cli','01_intro.sql','02_csv_analytics.sql','03_parquet.sql','04_explain.sql','05_analyze.sql','06_join_plan_optional.sql'])+'\n.quit\n'
        for i in [1,2]:
            start=time.perf_counter(); transcript=cli([str(DB)],batch)
            (out/f'cli_dry_run_{i}.txt').write_text(transcript,encoding='utf-8')
            check('Complete native CLI dry run '+str(i),'400000' in transcript and '35178394.19' in transcript and 'Row Groups Scanned: 4 / 25' in transcript,f'{time.perf_counter()-start:.3f} seconds')
        check('No database WAL after graceful close',not PathExists(str(DB)+'.wal'))
        evidence['elapsed_seconds']=round(time.perf_counter()-t0,3)
    except Exception as e:
        checks.append(('Unexpected error','FAILED',repr(e)))
        raise
    finally:
        stamp=datetime.now().astimezone().isoformat()
        (out/'demo_results.txt').write_text('Actual execution: '+stamp+'\n'+''.join(log),encoding='utf-8')
        (out/'results.json').write_text(json.dumps(evidence,indent=2,default=str),encoding='utf-8')
        lines=['# Executed verification report','',f'Run: {stamp}',f'Python: {sys.version}',
               '', '| Check | Status | Detail |','|---|---|---|']
        lines += ['| '+n+' | '+s+' | '+d.replace('|','/').replace('\n',' ')+' |' for n,s,d in checks]
        lines += ['',f'Automated checks: {sum(s=="PASS" for _,s,_ in checks)}/{len(checks)} PASS.',
          '', 'Evidence: results.json, demo_results.txt, cli_dry_run_1.txt, cli_dry_run_2.txt, *_plan.txt.',
          '', 'Timing is an observation on this laptop, not a comparative benchmark. Operator output rows are not physical rows read.']
        (out/'test_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'PASS: {len(checks)} executed checks in {time.perf_counter()-t0:.2f}s')

def PathExists(path):
    from pathlib import Path
    return Path(path).exists()

if __name__=='__main__': validate()
