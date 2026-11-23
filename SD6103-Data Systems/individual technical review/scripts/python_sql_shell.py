"""Emergency interactive API shell. All output comes from executed SQL."""
import sys
from pathlib import Path
from common import ROOT, connect, run_file

def display(desc,rows):
    if not desc: return
    if len(desc)==2 and desc[0][0] in ['explain_key']:
        for row in rows: print(row[1])
        return
    print(' | '.join(d[0] for d in desc))
    for row in rows[:20]: print(' | '.join(str(v) for v in row))
    if len(rows)>20: print(f'({len(rows)} actual rows; first 20 displayed)')

def main():
    errors=0; buffer=''
    with connect() as c:
        print('DuckDB Python API 2.0.0 | actual SQL | persistent database review')
        print('Use .read sql/01_intro.sql ... .read sql/05_analyze.sql; .quit exits.')
        while True:
            try: line=input('DuckDB-API> ' if not buffer else '       ...> ')
            except EOFError: break
            if not buffer and line.strip() in ['.quit','.exit']: break
            try:
                if not buffer and line.startswith('.read '):
                    name=line[6:].strip()
                    target=(ROOT/name).resolve()
                    if target.parent!=(ROOT/'sql').resolve() or target.suffix!='.sql':
                        raise ValueError('Only project sql/*.sql files are accepted')
                    print('\n'.join(s for s in target.read_text(encoding='utf-8').splitlines() if not s.startswith('.')))
                    for desc,rows in run_file(c,name): display(desc,rows)
                else:
                    buffer+=line+'\n'
                    if not buffer.rstrip().endswith(';'): continue
                    for statement in c.extract_statements(buffer):
                        c.execute(statement); display(c.description,c.fetchall())
                    buffer=''
            except Exception as e:
                errors+=1; buffer=''; print('ERROR:',e,file=sys.stderr)
    return 1 if errors else 0

if __name__=='__main__': sys.exit(main())
