from pathlib import Path
import json,hashlib,sys,math
import numpy as np
from scipy.special import rel_entr
sys.path.insert(0,'src')
from fidelity import write_json
root=Path('build/expanded');data=json.loads((root/'ablation_tokens.json').read_text())
full=json.loads((root/'full/eNeural-Qwen3.8-27B-fast/result.json').read_text())
captured=json.loads((root/'independent_fast/result.json').read_text())
assert captured['status']=='ok'
log=(root/'independent_fast.log').read_text()
assert 'Fatal Python error' not in log and 'Traceback' not in log
checks=[];max_error=0
for j,i in enumerate(data['selected_full_indices']):
    sample=data['samples'][j];ids=data['input_ids'][j];start=data['score_starts'][j]
    a=np.load(root/'full/bf16_reference'/f'{i}.npy',mmap_mode='r');b=np.load(root/'independent_fast'/f'{j}.npy',mmap_mode='r')
    assert hashlib.sha256(b.tobytes()).hexdigest()==full['distribution_sha256'][i]
    assert hashlib.sha256(b.tobytes()).hexdigest()==captured['distribution_sha256'][j]
    targets=np.asarray(ids[start:]);nll=-np.take_along_axis(b,targets[:,None],axis=1).astype(np.float64).sum()
    ref_nll=-np.take_along_axis(a,targets[:,None],axis=1).astype(np.float64).sum()
    kl=0.
    for lo in range(0,len(targets),16):
        p=np.exp(a[lo:lo+16].astype(np.float64));q=np.exp(b[lo:lo+16].astype(np.float64))
        kl+=rel_entr(p,q).sum()
    flips=int(np.sum(a.argmax(axis=1)!=b.argmax(axis=1)))
    independent=dict(tokens=len(targets),nll=float(nll),ref_nll=float(ref_nll),kl_sum=float(kl),flips=flips)
    actual=full['documents'][i];errors={k:abs(independent[k]-actual[k]) for k in independent};err=max(errors.values());max_error=max(max_error,err)
    assert err<1e-10,(sample['id'],errors)
    checks.append(dict(sample=sample['id'],full_index=i,errors=errors,passed=True))
write_json(root/'independent_metrics_audit.json',dict(passed=True,method='SciPy.special.rel_entr on independently recaptured full arrays, no renormalization; take_along_axis NLL; independent argmax flips',max_absolute_sum_error=float(max_error),checks=checks))
print('Independent full-vocabulary metric audit passed, max sum error:',max_error)
