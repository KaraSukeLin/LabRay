"""Check migration file hashes and recompute saved headline statistics; CPU only."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def load(p):
    return json.loads(p.read_text())

def records(p):
    return [json.loads(line) for line in p.open()]

def table(p):
    return list(csv.DictReader(p.open()))

def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),(a,b)

hashes=ROOT/'migration/SHA256SUMS'
if hashes.exists():
    checked=0
    for line in hashes.read_text().splitlines():
        expected,name=line.split('  ',1)
        h=hashlib.sha256()
        with (ROOT/name).open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''):h.update(block)
        assert h.hexdigest()==expected,name
        checked+=1
else:
    checked=0

formal=ROOT/'build/benchmark/ifeval_modes_formal_20261017'
experiment=load(formal/'experiment.json');total=0
for row in table(formal/'analysis_full_20261020/accuracy.csv'):
    data=[r for r in records(formal/row['model']/'questions.jsonl') if r['mode']==row['mode']]
    assert len(data)==1623
    assert len({(r['id'],r['seed']) for r in data})==1623
    assert all(r['correct']==all(r['strict_instructions']) for r in data)
    close(sum(r['correct'] for r in data)/len(data),row['prompt_strict_accuracy'])
    close(sum(sum(r['strict_instructions']) for r in data)/sum(len(r['strict_instructions']) for r in data),row['instruction_strict_accuracy'])
    total+=len(data)
assert total==16230

formal_models=load(ROOT/'build/autoquant/combined/provenance.json')['formal_models']
rows={r['model']:r for r in load(ROOT/'build/autoquant/combined/results.json') if r['run']!='bf16_repeat'}
for name in formal_models:
    r=rows[name];parts=r['documents'];n=sum(x['tokens'] for x in parts)
    assert len(parts)==288 and n==55296
    for key,value in [('ppl',math.exp(sum(x['nll'] for x in parts)/n)),('kl_bf16_to_candidate',sum(x['kl_sum'] for x in parts)/n),('top1_flip_rate',sum(x['flips'] for x in parts)/n)]:close(value,r[key])

terminal=ROOT/'build/terminal_bench/low_mtp3/formal_three_seed_20261022'
matrix=table(terminal/'analysis/matrix_801.csv')
assert len(matrix)==801
by_task=collections.defaultdict(list)
for r in matrix:by_task[r['task']].append(r)
assert len(by_task)==89 and all(len(v)==9 for v in by_task.values())
common={task:rr for task,rr in by_task.items() if all(r['score']!='' for r in rr)}
assert len(common)==53
scored=[r for r in matrix if r['score']!=''];assert len(scored)==533
for row in table(terminal/'analysis/model_summary.csv'):
    observed=[r for rr in common.values() for r in rr if r['model']==row['model']]
    assert len(observed)==159
    close(sum(float(r['score']) for r in observed)/159,row['mean_pass_rate'])

# Recompute service residency from interval unions; it is not compute utilization.
service=table(terminal/'report_review_20261023/service_intervals.csv')
intervals=collections.defaultdict(list)
for row in service:
    intervals[row['gpu']].append((float(row['started_at']),float(row['stopped_at'])))
seconds=0.
for values in intervals.values():
    merged=[]
    for start,end in sorted(values):
        if merged and start<=merged[-1][1]:
            merged[-1][1]=max(merged[-1][1],end)
        else:
            merged.append([start,end])
    seconds+=sum(end-start for start,end in merged)
wall=max(float(r['stopped_at']) for r in service)-min(float(r['started_at']) for r in service)
residency=seconds/(wall*5)
close(residency,load(terminal/'report_review_20261023/gpu_schedule_recomputed.json')['service_residency_fraction'])

# A sensitivity analysis stores replacements only. It is not added to primary counts.
patch=records(formal/'cap_recovery_sensitivity_20261020/replacements.jsonl')
assert len(patch)==3
for r in patch:
    original=next(x for x in records(formal/r['model']/'questions.jsonl') if (x['id'],x['mode'],x['seed'])==(r['id'],r['mode'],r['seed']))
    assert original['truncated'] and not r['truncated']
    assert original['prompt_token_ids_sha256_json']==r['prompt_token_ids_sha256_json']
print(json.dumps(dict(hash_checked_files=checked,ifeval_rows=total,fidelity_models=9,fidelity_documents_per_model=288,terminal_slots=801,terminal_scored=533,terminal_primary_tasks=53,sensitivity_replacements=3,service_residency_fraction=residency,inference=False),indent=2))
