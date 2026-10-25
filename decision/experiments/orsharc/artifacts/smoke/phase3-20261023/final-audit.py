import copy,sys
from pathlib import Path
sys.path.insert(0,'experiments/orsharc/src')
from b_orsharc import gates
from b_orsharc.common import read_json,write_json,digest,code_hash
from b_orsharc.data import validate_prepared
root=Path('experiments/orsharc');r=root/'artifacts/smoke/phase3-20261023'
original=gates.read_json
cases=['zero_update','frozen_change','environment_drift','stale_code','stale_configuration']
for case in cases:
 def altered(path):
  value=copy.deepcopy(original(path))
  if Path(path).name=='integrity.json':
   if case=='zero_update':value['max_abs_updates'][next(iter(value['max_abs_updates']))]=0
   if case=='frozen_change':value['frozen_changed']=['backbone']
  if Path(path).name=='runtime.json' and case=='environment_drift':value['mismatches']={'triton':'wrong version'}
  if Path(path).name=='evidence-index.json':
   if case=='stale_code':value['verified_code_sha256']='0'*64
   if case=='stale_configuration':value['verified_configuration_sha256']={}
  return value
 gates.read_json=altered
 try:gates.admission_verify(r/'evidence-index.json')
 except AssertionError:print('PASS: admission rejected',case)
 else:raise AssertionError(case)
 finally:gates.read_json=original
write_json(r/'negative-gate-checks.json',{'status':'PASS','rejected':cases})
a=read_json(r/'admission.json')
assert a['formal_experiment_admission'] and a['admission_code_sha256']==code_hash()
assert all(digest(p)==sha for p,sha in a['evidence'].items())
assert all(digest(root/name)==sha for name,sha in a['configuration_sha256'].items())
validate_prepared(root/'artifacts/prepared')
old=read_json(r/'completion-audit.json')
old.update({'code_sha256':code_hash(),'admission_evidence_sha256':a['evidence_sha256'],'final_evidence_hashes_verified':True,'stale_code_configuration_rejection':'PASS'})
write_json(r/'completion-audit.json',old)
print('PASS: final admission, all evidence/configuration hashes, prepared files and negative gates')
