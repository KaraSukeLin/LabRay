import os,sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path('src').resolve()))
from fidelity import full_logprobs,compare_arrays,aggregate,write_json
import numpy as np
from vllm import LLM,SamplingParams
def main():
    ids=json.loads(Path('build/final/tokens.json').read_text())['input_ids'][0]
    llm=LLM(model='/root/model/Qwen-Qwen3.8-27B',dtype='bfloat16',tensor_parallel_size=1,seed=0,max_model_len=4096,max_num_seqs=1,max_num_batched_tokens=4096,gpu_memory_utilization=.85,enforce_eager=True,enable_prefix_caching=False,enable_chunked_prefill=True,speculative_config=None,kv_cache_dtype='bfloat16',language_model_only=True,max_logprobs=-1,kernel_config={'linear_backend_per_quant':{'fp8_block_w8a8':'triton'}},logprobs_mode='raw_logprobs',generation_config='vllm')
    flat=llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=-1,flat_logprobs=True,detokenize=False),use_tqdm=False)[0]
    a=full_logprobs(flat.prompt_logprobs,ids,248320)
    outputs=[]
    for start in [0,20]:
     out=llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=1,prompt_logprob_token_ids=list(range(248320)),prompt_logprob_start=start,detokenize=False),use_tqdm=False)[0]
     b=out.prompt_token_id_logprobs
     outputs.append(dict(start=start,shape=None if b is None else b.shape,max_abs=None if b is None else float(np.nanmax(np.abs(a[start:]-b))),exact=False if b is None else bool(np.array_equal(a[start:],b))))
    print(outputs,flush=True)
    write_json(Path('build/expanded/dense_probe.json'),outputs)
    llm.llm_engine.engine_core.shutdown()

if __name__ == "__main__":
    main()
