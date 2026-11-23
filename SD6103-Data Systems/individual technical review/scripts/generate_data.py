"""Deterministic, entirely synthetic retail data; Python stdlib only."""
import calendar
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from common import ROOT, enter

SEED=6103
N_SALES=400_000
N_CUSTOMERS=15_000
N_PRODUCTS=200

def money(cents):
    return f'{cents//100}.{cents%100:02d}'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def generate():
    enter()
    rng=random.Random(SEED)
    raw=ROOT/'data/raw'
    raw.mkdir(parents=True,exist_ok=True)
    regions=['APAC','Europe','Americas','MEA']
    categories=['Electronics','Home','Sports','Books','Clothing']
    region_of={}
    with (raw/'customers.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n'); w.writerow(['customer_id','region','segment'])
        for cid in range(1,N_CUSTOMERS+1):
            region=rng.choices(regions,weights=[42,27,21,10],k=1)[0]
            region_of[cid]=region
            w.writerow([cid,region,rng.choices(['Consumer','Business'],[85,15],k=1)[0]])
    catalog={}
    with (raw/'products.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n'); w.writerow(['product_id','product_name','category','list_price'])
        for pid in range(1,N_PRODUCTS+1):
            cat=categories[(pid-1)//40]
            low,high={'Electronics':(8500,95000),'Home':(1500,22000),'Sports':(2000,28000),
                      'Books':(800,6500),'Clothing':(1200,18000)}[cat]
            price=rng.randint(low,high)
            catalog[pid]=(cat,price)
            w.writerow([pid,f'{cat}_{pid:03d}',cat,money(price)])
    groups=defaultdict(lambda:[0,0])
    total_cents=0; quantity=0; dates=Counter(); category_totals=Counter(); region_totals=Counter()
    with (raw/'sales.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['sale_id','sale_date','customer_id','product_id','quantity','unit_price','discount_amount'])
        for sid in range(1,N_SALES+1):
            month=rng.choices(range(1,13),[8,7,8,8,8,8,8,8,9,10,13,15],k=1)[0]
            day=rng.randint(1,calendar.monthrange(2025,month)[1])
            cid=rng.randint(1,N_CUSTOMERS)
            cat_index=rng.choices(range(5),[27,23,18,12,20],k=1)[0]
            pid=cat_index*40+rng.randint(1,40)
            cat,price=catalog[pid]
            qty=rng.choices([1,2,3,4,5],[54,26,12,6,2],k=1)[0]
            # Whole-line discount, rounded half up to cents with integer arithmetic.
            discount_bps=rng.choices([0,500,1000,1500],[45,25,22,8],k=1)[0]
            gross=qty*price
            discount=(gross*discount_bps+5000)//10000
            net=gross-discount
            date=f'2025-{month:02d}-{day:02d}'
            w.writerow([sid,date,cid,pid,qty,money(price),money(discount)])
            groups[(region_of[cid],cat)][0]+=1
            groups[(region_of[cid],cat)][1]+=net
            total_cents+=net; quantity+=qty; dates[date]+=1
            category_totals[cat]+=net; region_totals[region_of[cid]]+=net
    manifest={'synthetic':True,'seed':SEED,'sales':N_SALES,'customers':N_CUSTOMERS,'products':N_PRODUCTS,
        'date_min':min(dates),'date_max':max(dates),'currency':'synthetic SGD; one currency; no tax or shipping',
        'total_revenue_cents':total_cents,'total_quantity':quantity,
        'groups':[{'region':r,'category':cat,'transactions':n,'revenue_cents':c}
                  for (r,cat),(n,c) in sorted(groups.items())],
        'category_revenue_cents':dict(category_totals),'region_revenue_cents':dict(region_totals),
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(raw.glob('*.csv')) if p.name in ['sales.csv','customers.csv','products.csv']},
        'method':'Python random.Random(6103), fixed generation order, integer-cent prices/discounts. Weighted months/categories/regions; dates sampled in 2025. No external data.'}
    (ROOT/'data/dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Generated {N_SALES:,} synthetic sales; exact revenue SGD {money(total_cents)}')
    return manifest

if __name__=='__main__': generate()
