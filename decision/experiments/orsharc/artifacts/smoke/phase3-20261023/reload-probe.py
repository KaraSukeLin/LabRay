import sys,json,torch,time
from pathlib import Path
sys.path.insert(0,'experiments/orsharc/src')
from b_orsharc.runner import runtime_sources,model_config
from b_orsharc.common import write_json
runtime_sources(model_config('elephant'))
from train.common import load_bundle
from safetensors import safe_open
p=Path('experiments/orsharc/artifacts/smoke/phase3-20261023/measure-2')
b=load_bundle(str(p/'native/merged-fp32'),dtype='float32')
errors=[]; info={}
with safe_open(str(p/'native/merged-fp32/model.safetensors'),framework='pt') as f:
 for n,v in b.model.named_parameters():
  key=n.replace('model.layers.','model.language_model.layers.').replace('model.embed_tokens.','model.language_model.embed_tokens.').replace('model.norm.','model.language_model.norm.')
  saved=f.get_tensor(key)
  if not torch.equal(v.detach().cpu(),saved):errors.append({'name':n,'dtype':str(v.dtype),'saved_dtype':str(saved.dtype),'difference':float((v.detach().cpu().float()-saved.float()).abs().max())})
print('changed-on-load',errors,flush=True)
info={'changed_on_load':errors,'config':b.model.config.to_dict(),'dtype':str(b.model.dtype)}
write_json(p/'reload-diagnostic.json',info)
