from pathlib import Path
import json,hashlib,sys
sys.path.insert(0,'src')
from fidelity import aggregate,write_json
root=Path('build/expanded')
before={r['run']:r for r in json.loads((root/'full/results.json').read_text())}
after={r['run']:r for r in json.loads((root/'ablation/results.json').read_text())}
legacy={r['run']:r for r in json.loads(Path('build/final/results.json').read_text())}
tokens=json.loads((root/'ablation_tokens.json').read_text());indices=tokens['selected_full_indices'];offset=tokens['legacy_offset']
keys=['tokens','ppl','reference_ppl','ppl_delta','kl_bf16_to_candidate','top1_flip_rate']
checks=[]
assert set(before)==set(after)==set(legacy)
for name,b in before.items():
    a=after[name];old=legacy[name]
    assert a['status']=='ok' and a['worker_exit_code']==0
    assert a['tokens']==2736 and len(a['documents'])==20
    expected_hashes=[b['distribution_sha256'][i] for i in indices]
    assert a['distribution_sha256'][:offset]==expected_hashes,name
    assert a['distribution_sha256'][offset:]==old['distribution_sha256'],name
    for j,i in enumerate(indices):
        assert a['documents'][j]==b['documents'][i],(name,i)
    expanded_before=aggregate([b['documents'][i] for i in indices]);expanded_after=aggregate(a['documents'][:offset])
    assert expanded_before==expanded_after
    old_metrics=aggregate(a['documents'][offset:])
    differences={k:abs(old_metrics[k]-old[k]) for k in keys}
    assert max(differences.values())<1e-12,(name,differences)
    checks.append(dict(run=name,worker_exit_code=a['worker_exit_code'],expanded_distribution_hashes=offset,
                       expanded_document_metrics_exact=True,legacy_distribution_hashes=len(old['distribution_sha256']),
                       legacy_metric_errors=differences,passed=True))
assert json.loads((root/'ablation/sanity.json').read_text())['passed']
assert after['bf16_reference']['distribution_sha256']==after['bf16_repeat']['distribution_sha256']
write_json(root/'ablation.json',dict(passed=True,runs=checks,expanded_hash_matches=9*offset,
    legacy_hash_matches=9*8,all_hash_matches=9*20,full_models=8,ablation_scored_tokens=2736,
    full_scored_tokens=55296,
    source_before_sha256=hashlib.sha256((root/'fidelity_pre_simplification.py').read_bytes()).hexdigest(),
    source_after_sha256=hashlib.sha256(Path('src/fidelity.py').read_bytes()).hexdigest(),
    dataset_processing=json.loads((root/'dataset_rebuild_audit.json').read_text()),
    calibration_processing=json.loads((root/'calibration_simplification_audit.json').read_text())))
print('All9processes passed:108expanded +72legacy full-distribution hashes exact')
