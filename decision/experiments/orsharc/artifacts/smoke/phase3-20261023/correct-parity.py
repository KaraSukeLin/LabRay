import gc,sys,time,json,shutil
from pathlib import Path
sys.path.insert(0,'experiments/orsharc/src')
from b_orsharc.runner import runtime_sources,model_config
from b_orsharc.common import rows,write_json,write_rows,checkpoint_hash
runtime_sources(model_config('elephant'))
import torch
from train import common
from qev.readout import Readout,ReadoutConfig
from qev.engine import DecisionEngine
from qev.calibrate import Calibration
from qev.schema import SystemOneRequest
from b_orsharc.tasks import questions
p=Path('experiments/orsharc/artifacts/smoke/phase3-20261023/measure-2')
panel=rows(p/'panel.jsonl')
for name in ['merged.jsonl','parity.json','serving.json','profile.json']:
 shutil.copy2(p/name,p/('missing-attention-'+name))
for name in ['merged','merged-fp32']:
 d=p/'native'/name
 shutil.copy2(d/'readout.json',p/(name+'-missing-attention-readout.json'))
 shutil.copy2(p/'native/adapter/readout.json',d/'readout.json')
b=common.load_bundle(str(p/'native/merged'))
b.readout=Readout.attach(b.model,ReadoutConfig.load(p/'native/merged'),slot_ids=b.slot_ids,device=torch.device('cuda:0'))
results=[]
b.model.eval()
with torch.inference_mode():
 for r in panel:
  row=r['record'];ex=common.render_example(row,b.template_for(row),0,seed=None)
  t=time.perf_counter();logits,mask=common.batch_scores(b,b.model,[ex],[row],torch.device('cuda:0'))
  probs=logits[0,:len(ex.labels)].float().softmax(-1).cpu().tolist()
  results.append({'example_id':r['example_id'],'labels':ex.labels,'probabilities':dict(zip(ex.labels,probs)),'prediction':ex.labels[max(range(len(probs)),key=probs.__getitem__)],'tokens':r['tokens'],'seconds':time.perf_counter()-t})
write_rows(p/'merged.jsonl',results)
active=rows(p/'active.jsonl');parity={}
for name,tol in [('reloaded',1e-6),('merged',0.02)]:
 comp=rows(p/(name+'.jsonl'))
 diff=max(abs(a['probabilities'][k]-b['probabilities'][k]) for a,b in zip(active,comp) for k in a['probabilities'])
 changes=[a['example_id'] for a,b in zip(active,comp) if a['prediction']!=b['prediction']]
 parity[name]={'status':'PASS' if diff<=tol and not changes else 'FAIL','max_abs_probability_difference':diff,'label_changes':changes,'atol':tol,'rtol':0,'n':len(comp)}
write_json(p/'parity.json',parity);print(parity,flush=True)
profile=json.load(open(p/'profile.json'));profile['checkpoint']=checkpoint_hash(p/'native/merged');write_json(p/'profile.json',profile)
del b;gc.collect();torch.cuda.empty_cache()
e=DecisionEngine(str(p/'native/merged-fp32'),dtype='float32',head_dtype='float32',calibration=Calibration())
results=[]
for r in panel:
 out,stats=e.decide(SystemOneRequest(state=r['record']['state'],questions=questions('benchmark','single_choice')),debug=True)
 probs=out.answers['decision'].model_dump()['probabilities']
 results.append({'example_id':r['example_id'],'probabilities':probs,'prediction':max(probs,key=probs.__getitem__)})
write_rows(p/'fp32-merged-correct-readout.jsonl',results)
a=rows(p/'fp32-active.jsonl')
print('FP32 corrected',max(abs(x['probabilities'][k]-y['probabilities'][k]) for x,y in zip(a,results) for k in x['probabilities']),sum(x['prediction']!=y['prediction'] for x,y in zip(a,results)),flush=True)
