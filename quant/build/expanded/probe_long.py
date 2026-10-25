import os,sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path('src').resolve()))
from fidelity import full_logprobs,write_json
import numpy as np
from vllm import LLM,SamplingParams

def main():
    data=json.loads(Path('assets/expanded_v2/tokens.json').read_text())
    sample=next(s for s in data['samples'] if s['context']==2048)
    ids=sample['input_ids'];start=sample['score_start']-1
    llm=LLM(model='/root/model/Qwen-Qwen3.8-27B',dtype='bfloat16',tensor_parallel_size=1,seed=0,max_model_len=2304,max_num_seqs=1,max_num_batched_tokens=2304,gpu_memory_utilization=.75,enforce_eager=True,enable_prefix_caching=False,enable_chunked_prefill=True,speculative_config=None,kv_cache_dtype='bfloat16',language_model_only=True,max_logprobs=-1,kernel_config={'linear_backend_per_quant':{'fp8_block_w8a8':'triton'}},logprobs_mode='raw_logprobs',generation_config='vllm')
    flat=llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=-1,flat_logprobs=True,detokenize=False),use_tqdm=False)[0]
    a=full_logprobs(flat.prompt_logprobs,ids,248320)[start:].copy();del flat
    dense=llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=1,prompt_logprob_token_ids=list(range(248320)),prompt_logprob_start=start,detokenize=False),use_tqdm=False)[0].prompt_token_id_logprobs
    finite=np.isfinite(a)&np.isfinite(dense)
    result=dict(sample_id=sample['id'],context=sample['context'],shape=a.shape,exact=bool(np.array_equal(a,dense)),max_abs=float(np.abs(a[finite]-dense[finite]).max()),flat_sha256=hashlib.sha256(a.tobytes()).hexdigest(),dense_sha256=hashlib.sha256(dense.tobytes()).hexdigest())
    write_json(Path('build/expanded/long_equivalence.json'),result);print(result,flush=True)
    llm.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()
