import gc,json,sys,time,shutil
from pathlib import Path
sys.path.insert(0,'experiments/orsharc/src')
from b_orsharc.runner import runtime_sources,model_config
from b_orsharc.common import rows,write_json,write_rows
runtime_sources(model_config('elephant'))
import torch,peft
from qev.engine import DecisionEngine
from qev.calibrate import Calibration
from qev.schema import SystemOneRequest
from b_orsharc.tasks import questions
p=Path('experiments/orsharc/artifacts/smoke/phase3-20261023/measure-2')
panel=rows(p/'panel.jsonl')
qs=questions('benchmark','single_choice')
def predict(e):
 result=[]
 for r in panel:
  begin=time.perf_counter()
  out,stats=e.decide(SystemOneRequest(state=r['record']['state'],questions=qs),debug=True)
  probs=out.answers['decision'].model_dump()['probabilities']
  result.append({'example_id':r['example_id'],'probabilities':probs,'prediction':max(probs,key=probs.__getitem__),'seconds':time.perf_counter()-begin})
 return result
started=time.perf_counter()
e=DecisionEngine('/root/model/orsharc-elephant',dtype='float32',head_dtype='float32',calibration=Calibration())
e.model=peft.PeftModel.from_pretrained(e.model,str(p/'native/adapter')).eval()
a=predict(e)
write_rows(p/'fp32-active.jsonl',a)
merged=e.model.merge_and_unload(safe_merge=True)
b=predict(e)
write_rows(p/'fp32-merged-memory.jsonl',b)
destination=p/'native/merged-fp32'
merged.save_pretrained(str(destination),safe_serialization=True)
e.tokenizer.save_pretrained(str(destination))
shutil.copy2('/root/model/orsharc-elephant/readout.json',destination/'readout.json')
del e,merged
gc.collect();torch.cuda.empty_cache()
e=DecisionEngine(str(destination),dtype='float32',head_dtype='float32',calibration=Calibration())
c=predict(e)
write_rows(p/'fp32-merged-reloaded.jsonl',c)
res={}
for name,compared in [('in_memory',b),('reloaded',c)]:
 d=max(abs(x['probabilities'][k]-y['probabilities'][k]) for x,y in zip(a,compared) for k in x['probabilities'])
 changes=[x['example_id'] for x,y in zip(a,compared) if x['prediction']!=y['prediction']]
 res[name]={'max_probability_difference':d,'label_changes':changes,'status':'PASS' if d<=1e-4 and not changes else 'FAIL'}
write_json(p/'fp32-probe.json',{'results':res,'seconds':time.perf_counter()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'tolerance':1e-4})
print(res,flush=True)
