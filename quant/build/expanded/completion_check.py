from pathlib import Path
import csv,hashlib,json,re,sys
import numpy as np
sys.path.insert(0,'src')
from fidelity import aggregate,write_json
R=Path.cwd();E=R/'build/expanded';checks={}
def check(name,condition):
    checks[name]=bool(condition)
    assert condition,name
full=json.loads((E/'full/results.json').read_text());after=json.loads((E/'ablation/results.json').read_text());data=json.loads((R/'assets/expanded_v2/tokens.json').read_text())
models={p.name for p in Path('/root/model').iterdir() if p.is_dir() and not p.name.startswith('.')}
check('all_eight_models_actual',len(models)==8 and {r['model'] for r in full}==models)
check('nine_clean_full_workers',len(full)==9 and all(r['status']=='ok' and r['worker_exit_code']==0 for r in full))
check('at_least_50000_each',all(r['tokens']==55296 and len(r['documents'])==288 for r in full))
source_before=hashlib.sha256((E/'fidelity_pre_simplification.py').read_bytes()).hexdigest();source_after=hashlib.sha256((R/'src/fidelity.py').read_bytes()).hexdigest();token_sha=hashlib.sha256((R/'assets/expanded_v2/tokens.json').read_bytes()).hexdigest()
check('full_source_and_tokens_identical',all(r['source_sha256']==source_before and r['tokens_sha256']==token_sha for r in full))
check('prepared_tokens_used_unchanged',(R/'assets/expanded_v2/tokens.json').read_bytes()==(E/'full/tokens.json').read_bytes())
check('mtp_off_bf16_kv_v2',all(r['mtp'] is False and r['kv_cache_dtype']=='bfloat16' and r['v2_model_runner']=='1' and r['vocab_size']==248320 for r in full+after))
check('all_same_document_score_positions',all([p['id'] for p in r['documents']]==[s['id'] for s in data['samples']] and all(p['tokens']==192 for p in r['documents']) for r in full))
first,second=full[:2];sanity=json.loads((E/'full/sanity.json').read_text())
check('independent_sanity_exact',sanity['passed'] and first['distribution_sha256']==second['distribution_sha256'] and second['kl_bf16_to_candidate']==0 and second['top1_flip_rate']==0 and second['ppl_delta']==0)
for r in full:
    log=(E/'full'/r['run']/'run.log').read_text()
    check('clean_log_'+r['run'],'Fatal Python error' not in log and 'Traceback' not in log and 'Using V2 Model Runner' in log)
    expected=aggregate(r['documents'])
    check('aggregate_'+r['run'],all(abs(expected[k]-r[k])<1e-12 for k in expected))
check('saved_full_reference_shapes',all(np.load(E/'full/bf16_reference'/f'{i}.npy',mmap_mode='r').shape==(192,248320) for i in range(288)))
check('ablation_all_models_clean',len(after)==9 and all(r['status']=='ok' and r['worker_exit_code']==0 and r['tokens']==2736 and r['source_sha256']==source_after for r in after))
for r in after:
    log=(E/'ablation'/r['run']/'run.log').read_text()
    check('clean_ablation_'+r['run'],'Fatal Python error' not in log and 'Traceback' not in log)
for file in ['ablation.json','independent_metrics_audit.json','dataset_rebuild_audit.json','calibration_simplification_audit.json','analysis_simplification_audit.json']:
    check(file,json.loads((E/file).read_text())['passed'])
check('all_180_distribution_hashes_match',json.loads((E/'ablation.json').read_text())['all_hash_matches']==180)
check('calibration_overlap_zero',all(r['matching_documents']==0 for r in json.loads((E/'calibration_overlap.json').read_text())['results']))
check('unique_document_windows',json.loads((E/'dataset_audit.json').read_text())['unique_target_sequences']==288)
check('bootstrap_seed_stable',json.loads((E/'bootstrap_seed_sensitivity.json').read_text())['all_primary_conclusions_unchanged'])
comparison=list(csv.DictReader((E/'analysis/comparison.csv').open()));check('comparison_eight_success',len(comparison)==8 and {r['model'] for r in comparison}==models and all(r['status']=='ok' for r in comparison))
for name,n in [('grouped',160),('intervals',960),('fast_pairwise',420),('leave_one_domain_out',80),('document_stability',12)]:
    rows=list(csv.DictReader((E/'analysis'/f'{name}.csv').open()));check(name+'_complete',len(rows)==n)
    if name in ['intervals','fast_pairwise','leave_one_domain_out']:
        check(name+'_finite',all(np.isfinite(float(r[k])) for r in rows for k in ['estimate','low','high']))
required=['src/fidelity.py','src/prepare_dataset.py','src/analyze.py','src/audit_calibration.py','src/plot_intervals.py','scripts/run_expanded.sh','assets/expanded_config.json','assets/expanded_v2/manifest.json','build/REPORT.md','build/REPORT_round1.md','build/expanded/REPORT.md','build/expanded/model_checks.csv','build/expanded/source_versions.json','build/expanded/package_provenance.json','build/expanded/environment.json','build/expanded/environment.freeze.txt','build/expanded/analysis/fast_intervals.png','build/expanded/analysis/fast_intervals.pdf','build/fidelity-public-v2.1.tar.gz']
check('all_deliverables_exist',all((R/p).is_file() and (R/p).stat().st_size>0 for p in required))
check('targeted_seven_tests','7 passed' in (E/'targeted_tests.txt').read_text())
links=[]
for p in [R/'README.md',R/'build/REPORT.md',E/'REPORT.md',R/'assets/expanded_v2/README.md',R/'assets/expanded_v2/LICENSES.md']:
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        if not target.startswith(('http:','https:','#')):
            check('link_'+str(p.relative_to(R))+'_'+target,(p.parent/target.split('#')[0]).exists());links.append(target)
for f in json.loads((E/'model_fingerprints.json').read_text()):
    p=Path('/root/model')/f['model']/f['file'];check('model_file_'+f['model']+'_'+f['file'],p.is_file() and p.stat().st_size==f['bytes'])
write_json(E/'completion_audit.json',dict(passed=True,checks=checks,models=sorted(models),full_scored_tokens_per_model=55296,documents=288,ablation_scored_tokens_per_model=2736,full_workers=9,ablation_workers=9,distribution_hashes_matched=180,source_before_sha256=source_before,source_after_sha256=source_after,tokens_sha256=token_sha,dataset_archive_sha256=hashlib.sha256((R/'build/fidelity-public-v2.1.tar.gz').read_bytes()).hexdigest(),checked_local_links=len(links),external_publication=False))
print('Completion audit passed:',len(checks),'checks')
