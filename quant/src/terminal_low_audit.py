"""CPU audit of actual Low requests, IDs, parser output, and thinking history."""
import argparse
from collections import defaultdict
import copy
from pathlib import Path

from terminal_bench import BASE, read, save, sha
from agentic_fidelity import render_ids


def audit(folder):
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning import ReasoningParserManager
    cfg=read(folder/'scope.json')['config']
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    groups=defaultdict(list)
    details=[]
    for path in sorted(folder.glob('services/gpu*/*/audit/*.json')):
        event=read(path)
        assert event['status']=='complete' and event['http_status']==200,path
        request,response=event['request'],event['response']
        assert event['model']==folder.name
        assert request['model']==response['model']==folder.name
        assert request['reasoning_effort']=='low'
        assert request['chat_template_kwargs']==cfg['chat_template_kwargs']
        assert event['engine_sampling']==dict(cfg['sampling'],seed=cfg['seeds'][0])
        ids=render_ids(tokenizer,request)
        assert ids==response['prompt_token_ids'],path
        rendered=tokenizer.apply_chat_template(request['messages'],tokenize=False,add_generation_prompt=True,**cfg['chat_template_kwargs'])
        assert 'Reasoning effort is set to low.' in rendered
        assert 'Reasoning effort is set to xhigh.' not in rendered
        assert rendered.endswith('<think>\n')
        choice=response['choices'][0]
        assert choice['finish_reason']=='stop' and choice['token_ids']
        message=choice['message']
        raw=tokenizer.decode(choice['token_ids'],skip_special_tokens=False)
        for eos in read(BASE/'generation_config.json')['eos_token_id']:
            raw=raw.removesuffix(tokenizer.decode([eos],skip_special_tokens=False))
        parser=ReasoningParserManager.get_reasoning_parser('qwen3')(tokenizer,chat_template_kwargs=cfg['chat_template_kwargs'])
        parser.adjust_initial_state_from_prompt(ids)
        reasoning,content=parser.extract_reasoning(raw,ChatCompletionRequest(**copy.deepcopy(request)))
        assert (reasoning or '')==(message.get('reasoning') or message.get('reasoning_content') or '')
        assert (content or '')==(message.get('content') or '')
        identity=event['identity']['x-terminal-identity']
        mode=identity.endswith('/True')
        for assistant in [m for m in request['messages'] if m['role']=='assistant']:
            if mode: assert assistant.get('reasoning_content')
            else: assert not assistant.get('reasoning_content')
        groups[identity].append(event)
        details.append(dict(path=str(path),sha256=sha(path),prompt_tokens=len(ids),output_tokens=len(choice['token_ids']),
            prompt_ids_match=True,parser_match=True,engine_sampling_match=True,interleaved=mode,
            tokens_per_second_including_prefill=len(choice['token_ids'])/event['elapsed_seconds']))
    assert len(groups)==2
    for identity,events in groups.items():
        events.sort(key=lambda event:event['started_at'])
        assert len(events)==2
        before,after=events
        old,new=before['request']['messages'],after['request']['messages']
        assert len(new)==len(old)+2 and new[:len(old)]==old
        reply=before['response']['choices'][0]['message']
        assert new[-2]['role']=='assistant' and new[-2]['content']==reply['content']
        assert new[-1]['role']=='user' and 'LOW_MTP_OK' in new[-1]['content']
        if identity.endswith('/True'):
            assert new[-2]['reasoning_content']==(reply.get('reasoning') or reply.get('reasoning_content'))
    for mode in [0,1]:
        result=read(folder/f'interleaved-{mode}/result.json')
        assert result['passed'] and result['accepted']>0 and result['drafted']>0
        assert not read(folder/f'interleaved-{mode}/protocol/proot/cleanup.json')['after']
    runtime=[]
    for service in read(folder/'services/services.json'):
        log=(Path(service['folder'])/'server.log').read_text()
        assert f"model='/root/model/{folder.name}'" in log
        assert "num_spec_tokens=3" in log and 'tensor_parallel_size=1' in log
        assert 'kv_cache_dtype=bfloat16' in log
        expected_quant='None' if folder.name=='Qwen-Qwen3.8-27B' else 'compressed-tensors'
        assert 'quantization='+expected_quant in log
        runtime.append(dict(gpu=service['gpu'],model=folder.name,quantization=expected_quant,
            mtp=3,tp=1,kv_dtype='bfloat16',fp4_weight_only_marlin='Weight-only FP4 compression' in log,
            source=str(Path(service['folder'])/'server.log')))
    save(folder/'RUNTIME.json',dict(passed=True,services=runtime))
    save(folder/'AUDIT.json',dict(passed=True,requests=len(details),details=details,
        scope='Bounded synthetic real-terminal preflight; no benchmark score or full trajectory fidelity claim.'))
    print('PASS: four actual Low/MTP3 requests, both histories, prompt IDs, parser, sampling and cleanup')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    audit(parser.parse_args().folder)
