from pathlib import Path
import csv,json,sys
sys.path.insert(0,'src')
from analyze import cluster_interval,PAIR_METRICS,FAST,PEERS
from fidelity import write_json
root=Path('build/expanded');rows=json.loads((root/'full/results.json').read_text());models={r['model']:r for r in rows if r['run']!='bf16_repeat'}
def direction(lo,hi):return 'positive' if lo>0 else ('negative' if hi<0 else 'uncertain')
reference={}
for row in csv.DictReader((root/'analysis/fast_pairwise.csv').open()):
    if row['grouping']=='all' and float(row['confidence'])>.99:
        reference[row['peer'],row['metric']]=direction(float(row['low']),float(row['high']))
checks=[]
for peer in PEERS:
    paired=[dict(a,ref_nll=b['nll'],kl_sum=a['kl_sum']-b['kl_sum'],flips=a['flips']-b['flips']) for a,b in zip(models[FAST]['documents'],models[peer]['documents'])]
    for seed in [20261017,20261018]:
        point,ci=cluster_interval(paired,pair=True,seed=seed,confidence=1-.05/12)
        for metric,value,lo,hi in zip(PAIR_METRICS,point,ci.low,ci.high):
            if metric not in ['delta_nll','kl_difference','flip_difference']:continue
            state=direction(lo,hi)
            checks.append(dict(peer=peer,seed=seed,metric=metric,estimate=float(value),low=float(lo),high=float(hi),
                               direction=state,reference_direction=reference[peer,metric],same_direction=state==reference[peer,metric]))
write_json(root/'bootstrap_seed_sensitivity.json',dict(seeds=[20261016,20261017,20261018],resamples_per_seed=19999,
    confidence=1-.05/12,all_primary_conclusions_unchanged=all(c['same_direction'] for c in checks),checks=checks))
print('Seed sensitivity unchanged:',all(c['same_direction'] for c in checks))
