from pathlib import Path
import csv,json,hashlib,sys
sys.path.insert(0,'src');from fidelity import write_json
root=Path('build/expanded');checks=[]
for name in ['comparison','grouped','intervals','fast_pairwise','leave_one_domain_out','document_stability','round1_comparison']:
    a=list(csv.DictReader((root/'analysis_before'/f'{name}.csv').open()));b=list(csv.DictReader((root/'analysis'/f'{name}.csv').open()))
    filtered=[r for r in a if r.get('grouping')!='language']
    assert filtered==b,name
    checks.append(dict(file=name+'.csv',before_rows=len(a),after_rows=len(b),all_retained_rows_exact=True))
a=json.loads((root/'analysis_before/statistics.json').read_text());b=json.loads((root/'analysis/statistics.json').read_text());a.pop('source_sha256');b.pop('source_sha256');assert a==b
write_json(root/'analysis_simplification_audit.json',dict(passed=True,change='Remove language groups that duplicate the one-to-one domain/language mapping; retain all domain, context and joint groups',checks=checks))
print('All retained analysis values and confidence bounds are identical')
