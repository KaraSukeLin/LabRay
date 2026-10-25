#!/usr/bin/env python3
"""Audit actual Qwen protocol records; prepare explicit smoke/gate readiness files."""
import argparse
import copy
from collections import defaultdict
import json
import re
from pathlib import Path
import sys

from agentic_fidelity import render_ids
from terminal_bench import BASE, CONFIG, ROOT, read, save, sha


def audit_api(folder):
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning import ReasoningParserManager
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    config=read(CONFIG)
    details=[]
    groups=defaultdict(list)
    failures=[]
    model_diagnostics=[]
    timeout_sessions=set()
    for result_path in folder.rglob('result.json'):
        result=read(result_path)
        if (result.get('exception_info') or {}).get('exception_type')=='AgentTimeoutError':
            trajectory=result_path.parent/'agent/trajectory.json'
            if trajectory.exists(): timeout_sessions.add(read(trajectory).get('session_id'))
    for path in sorted(folder.rglob('audit/*.json')):
        event=read(path)
        if event.get('status')!='complete' or event.get('http_status')!=200:
            session=event.get('identity',{}).get('x-terminal-trial')
            if session in timeout_sessions and event.get('http_status') in [None,200]:
                model_diagnostics.append(dict(source=str(path),kind='official_agent_timeout_cancelled_inflight_request',
                    token_ids_available=False))
            else:
                failures.append(dict(source=str(path),error='Incomplete/non-200 API call'))
            continue
        q,r=event['request'],event['response']
        expected=dict(config['sampling'],seed=q.get('seed'))
        errors=[]
        identity=event['identity'].get('x-terminal-identity','')
        repetition=re.search(r'/r([0-2])-thinking-([01])/gpu([0-4])$',identity)
        if repetition:
            if q.get('seed')!=config['seeds'][int(repetition[1])]: errors.append('Wrong trial repeat seed')
            if event['gpu']!=repetition[3]: errors.append('Wrong trial GPU endpoint')
            if identity.split('/')[1]!=event['model']: errors.append('Wrong trial model')
        elif not identity.startswith('synthetic/'):
            errors.append('Unknown trial identity')
        if q.get('seed') not in config['seeds']: errors.append('Unexpected seed')
        if event.get('engine_sampling')!=expected: errors.append('Engine sampling mismatch')
        if q.get('reasoning_effort')!='xhigh': errors.append('Reasoning effort not passed to API')
        if q.get('chat_template_kwargs')!=config['chat_template_kwargs']: errors.append('Template kwargs mismatch')
        ids=render_ids(tokenizer,q)
        if ids!=r.get('prompt_token_ids'): errors.append('Actual prompt IDs differ from saved messages')
        choice=r['choices'][0]
        actual_ids=choice.get('token_ids')
        if not actual_ids: errors.append('Missing actual generated IDs')
        if choice['finish_reason']=='length':
            model_diagnostics.append(dict(source=str(path),kind='output_truncated'))
        elif choice['finish_reason']!='stop': errors.append('Abnormal API finish')
        message=choice['message']
        reasoning=message.get('reasoning') or message.get('reasoning_content') or ''
        content=message.get('content') or ''
        if not reasoning or not content:
            model_diagnostics.append(dict(source=str(path),kind='missing_reasoning_or_final'))
        if actual_ids:
            raw=tokenizer.decode(actual_ids,skip_special_tokens=False)
            # Serving removes only stop/special EOS tokens from the displayed generation.
            for eos in json.loads((BASE/'generation_config.json').read_text())['eos_token_id']:
                raw=raw.removesuffix(tokenizer.decode([eos],skip_special_tokens=False))
            parser=ReasoningParserManager.get_reasoning_parser('qwen3')(
                tokenizer,chat_template_kwargs=q['chat_template_kwargs'])
            parser.adjust_initial_state_from_prompt(ids)
            parsed_reasoning,parsed_content=parser.extract_reasoning(raw,ChatCompletionRequest(**copy.deepcopy(q)))
            if (parsed_reasoning or '')!=reasoning or (parsed_content or '')!=content:
                errors.append('Qwen reasoning/final replay mismatch')
        preserved=tokenizer.apply_chat_template(q['messages'],tokenize=False,add_generation_prompt=True,
                                               **config['chat_template_kwargs'])
        ablated=tokenizer.apply_chat_template(q['messages'],tokenize=False,add_generation_prompt=True,
            **dict(config['chat_template_kwargs'],preserve_thinking=False))
        history=[m for m in q['messages'] if m['role']=='assistant']
        mode=re.search(r'thinking-([01])',identity)
        for message in history:
            if mode and mode[1]=='0' and message.get('reasoning_content'):
                errors.append('Default interleaved protocol unexpectedly includes historical reasoning')
        histories=[m.get('reasoning_content','').strip() for m in history]
        row=dict(source=str(path),sha256=sha(path),gpu=event['gpu'],identity=event['identity'],
            prompt_tokens=len(ids),output_tokens=len(actual_ids or []),
            reasoning_tokens=(r['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens'),
            history_assistants=len(history),history_reasoning_characters=sum(map(len,histories)),
            preserve_changes_prompt=preserved!=ablated,
            preserve_added_tokens=len(ids)-len(tokenizer.encode(ablated,add_special_tokens=False)),
            prompt_has_xhigh='Reasoning effort is set to xhigh.' in preserved,
            prompt_ends_think=preserved.endswith('<think>\n'),errors=errors)
        if not row['prompt_has_xhigh'] or not row['prompt_ends_think']: errors.append('Thinking prompt malformed')
        details.append(row)
        group=(event['identity'].get('x-terminal-identity'),event['identity'].get('x-terminal-trial'))
        groups[group].append(event)
    history_checks=[]
    for key,events in groups.items():
        events.sort(key=lambda e:e['started_at'])
        resets=0
        for prev,current in zip(events,events[1:]):
            old=prev['request']['messages']; new=current['request']['messages']
            reply=prev['response']['choices'][0]['message']
            # Native summarization has separate requests and a new shorter history.
            continuous=len(new)==len(old)+2 and new[:len(old)]==old
            if not continuous:
                resets+=1
                continue
            assistant=new[-2]
            assert assistant['role']=='assistant' and new[-1]['role']=='user'
            assert assistant['content']==(reply.get('content') or ''), 'Historical assistant final changed'
            mode=re.search(r'thinking-([01])',key[0] or '')
            expected_reasoning=(reply.get('reasoning') or reply.get('reasoning_content') or '')
            if mode and mode[1]=='1':
                assert assistant.get('reasoning_content','')==expected_reasoning, 'Historical reasoning lost or changed'
            elif assistant.get('reasoning_content'):
                raise AssertionError('Unexpected reasoning in the official default protocol')
        history_checks.append(dict(identity=key,requests=len(events),history_resets=resets))
    failures.extend(dict(source=row['source'],error=error) for row in details for error in row['errors'])
    return dict(passed=bool(details) and not failures,requests=len(details),details=details,
        failures=failures,model_diagnostics=model_diagnostics,history_checks=history_checks,
        note='History resets must be reconciled against native ATIF summarization sub-trajectories for official runs.')


def readiness(runs,stage,api):
    plan=read(runs/'plan.json')
    config=plan['config']
    expected={(c['model'],c['repeat'],c['interleaved_thinking'],task)
              for c in plan['cells'][stage] for task in c['tasks']}
    observed={}
    errors=[]
    for cell in plan['cells'][stage]:
        key=f'{stage}/{cell["model"]}/r{cell["repeat"]}-thinking-{int(cell["interleaved_thinking"])}'
        for path in (runs/key/'jobs').glob('*/*/result.json'):
            r=read(path)
            task=r['task_name'].split('/')[-1]
            k=(cell['model'],cell['repeat'],cell['interleaved_thinking'],task)
            assert k not in observed,'Duplicate task/repeat outcome'
            observed[k]=str(path)
            exc=(r.get('exception_info') or {}).get('exception_type')
            if exc and exc not in {'AgentTimeoutError','ContextLengthExceededError',
                                  'OutputLengthExceededError','AgentSafetyRefusalError'}:
                errors.append(dict(source=str(path),error=exc))
            if not r.get('verifier_result') or not r['verifier_result'].get('rewards'):
                errors.append(dict(source=str(path),error='Official Verifier reward missing'))
            cfg=r['config']
            multipliers=['timeout_multiplier','agent_timeout_multiplier','verifier_timeout_multiplier',
                'agent_setup_timeout_multiplier','environment_build_timeout_multiplier']
            modified=any(cfg.get(k) not in [None,1.0] for k in multipliers)
            modified|=any(cfg[phase].get(k) is not None for phase,keys in [
                ('agent',['override_timeout_sec','override_setup_timeout_sec','max_timeout_sec']),
                ('verifier',['override_timeout_sec']),
                ('environment',['override_cpus','override_memory_mb','override_storage_mb','override_gpus'])]
                for k in keys)
            if modified: errors.append(dict(source=str(path),error='Official timeout/resource modified'))
    missing=sorted(expected-set(observed))
    if set(observed)-expected: errors.append(dict(error='Unexpected official task result'))
    if any(x['history_resets'] for x in api['history_checks']):
        errors.append(dict(error='History reset/summarization requires ATIF review before gate approval'))
    result=dict(passed=not missing and not errors and api['passed'],stage=stage,
        expected=len(expected),observed=len(observed),missing=missing,errors=errors,
        api_requests=api['requests'],source_manifest_sha256=plan['manifest_sha256'])
    save(runs/('SMOKE_READY.json' if stage=='smoke' else 'GATE_READY.json'),result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['smoke','gate'])
    a=p.parse_args()
    audit=audit_api(a.runs)
    if a.stage: audit['readiness']=readiness(a.runs,a.stage,audit)
    save(a.output,audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['details','history_checks']},ensure_ascii=False))
    sys.exit(0 if audit['passed'] and (not a.stage or audit['readiness']['passed']) else 1)
