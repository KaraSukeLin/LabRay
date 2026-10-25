"""CPU audit of smoke wire records and native trial artifacts; no inference."""
import argparse
import copy
from collections import Counter,defaultdict
import json
from pathlib import Path
import sqlite3

from terminal_bench import BASE,read,save,sha
from terminal_low import CONFIG,OUTPUT
from agentic_fidelity import render_ids


def main(partial=False):
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning import ReasoningParserManager
    cfg=read(CONFIG)
    connection=sqlite3.connect('file:'+str(OUTPUT/'smoke.sqlite')+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row
    rows={row['id']:dict(row) for row in connection.execute('SELECT * FROM trials')}
    assert len(rows)==30
    if partial:
        rows={key:row for key,row in rows.items() if row['status'] not in {'PENDING','RUNNING'}}
    else:
        assert all(row['status'] not in {'PENDING','RUNNING'} for row in rows.values())
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    events=defaultdict(list)
    failures=[]
    incomplete=[]
    details=[]
    for path in sorted((OUTPUT/'services').rglob('audit/*.json')):
        event=read(path)
        identity=event.get('identity',{}).get('x-terminal-identity','')
        assert identity.startswith('smoke/'),path
        key=identity.rsplit('/',1)[-1]
        if partial and key not in rows: continue
        row=rows[key]
        request=event['request']
        assert event['model']==request['model']==row['model']
        assert request['reasoning_effort']=='low'
        assert request['chat_template_kwargs']==cfg['chat_template_kwargs']
        assert request['seed']==row['seed']
        if event.get('engine_sampling'):
            assert event['engine_sampling']==dict(cfg['sampling'],seed=row['seed'])
        response=event.get('response')
        if not isinstance(response,dict) or not response.get('choices') or event['status']!='complete':
            incomplete.append(dict(trial=key,path=str(path),started_at=event['started_at'],status=event['status'],http_status=event.get('http_status')))
            if row['status']!='TIMEOUT': failures.append(dict(path=str(path),reason='Incomplete API without an official timeout'))
            continue
        assert event['http_status']==200
        assert event['engine_sampling']==dict(cfg['sampling'],seed=row['seed'])
        ids=render_ids(tokenizer,request)
        assert ids==response['prompt_token_ids'],path
        choice=response['choices'][0]
        output=choice['token_ids']
        assert output
        text=tokenizer.apply_chat_template(request['messages'],tokenize=False,add_generation_prompt=True,**cfg['chat_template_kwargs'])
        assert 'Reasoning effort is set to low.' in text and text.endswith('<think>\n')
        raw=tokenizer.decode(output,skip_special_tokens=False)
        for eos in read(BASE/'generation_config.json')['eos_token_id']:
            raw=raw.removesuffix(tokenizer.decode([eos],skip_special_tokens=False))
        parser=ReasoningParserManager.get_reasoning_parser('qwen3')(tokenizer,chat_template_kwargs=cfg['chat_template_kwargs'])
        parser.adjust_initial_state_from_prompt(ids)
        reasoning,content=parser.extract_reasoning(raw,ChatCompletionRequest(**copy.deepcopy(request)))
        message=choice['message']
        assert (reasoning or '')==(message.get('reasoning') or message.get('reasoning_content') or '')
        assert (content or '')==(message.get('content') or '')
        history=[m for m in request['messages'] if m['role']=='assistant']
        if not row['mode']: assert all(not m.get('reasoning_content') for m in history)
        detail=dict(trial=key,path=str(path),sha256=sha(path),prompt_tokens=len(ids),output_tokens=len(output),
            reasoning_tokens=(response.get('usage',{}).get('completion_tokens_details') or {}).get('reasoning_tokens'),
            finish_reason=choice['finish_reason'],prompt_ids_match=True,sampling_match=True,parser_match=True,
            history_assistants=len(history),seconds=event['elapsed_seconds'],started_at=event['started_at'])
        details.append(detail)
        events[key].append(event)
    trials=[]
    coverage=read(OUTPUT/'coverage.json')['tasks']
    for key,row in rows.items():
        folder=OUTPUT/'smoke'/key
        native=read(folder/'trial/result.json')
        assert native.get('finished_at')
        counts=read(folder/'trial/verifier/ctrf.json')['results']['summary']
        oracle=Path(coverage[row['task']]['controls']['oracle']['result']).parent/'verifier/ctrf.json'
        expected=read(oracle)['results']['summary']['tests']
        assert counts['tests']==expected>0,(key,counts,expected)
        assert not read(folder/'trial/proot/cleanup.json')['after']
        trajectory=read(folder/'trial/agent/trajectory.json')
        tool_calls=sum(len(step.get('tool_calls') or []) for step in trajectory.get('steps',[]))
        ordered=sorted(events[key],key=lambda event:event['started_at'])
        # Harbor uses the same local trial name in separate directories; join by
        # the globally unique X-Terminal-Identity as well as the native session.
        assert all(event['identity'].get('x-terminal-trial')==trajectory['session_id'] for event in ordered)
        resets=0
        for before,after in zip(ordered,ordered[1:]):
            old,new=before['request']['messages'],after['request']['messages']
            if len(new)!=len(old)+2 or new[:len(old)]!=old:
                resets+=1
                continue
            reply=before['response']['choices'][0]['message']
            assert new[-2]['role']=='assistant' and new[-1]['role']=='user'
            assert new[-2]['content']==(reply.get('content') or '')
            expected=(reply.get('reasoning') or reply.get('reasoning_content') or '') if row['mode'] else ''
            assert (new[-2].get('reasoning_content') or '')==expected
        if resets:failures.append(dict(trial=key,reason='History reset needs native summarization reconciliation',resets=resets))
        trials.append(dict(id=key,model=row['model'],task=row['task'],interleaved=bool(row['mode']),status=row['status'],
            reward=(native.get('verifier_result') or {}).get('rewards',{}).get('reward'),
            complete_api_responses=len(ordered),tool_calls=tool_calls,history_resets=resets,cleanup_clean=True,
            session_id=trajectory.get('session_id'),verifier_tests=counts['tests'],verifier_failed=counts.get('failed',0),verifier_test_count_matches_oracle=True))
    first_turn_pairs=[]
    cells=defaultdict(dict)
    for key,row in rows.items():cells[(row['model'],row['task'])][row['mode']]=key
    for (model,task),modes in cells.items():
        if set(modes)!={0,1}:continue
        pair=[sorted(events[modes[mode]],key=lambda event:event['started_at']) for mode in [0,1]]
        available=all(pair) and all(not any(item['trial']==modes[mode] and item['started_at']<=pair[mode][0]['started_at'] for item in incomplete) for mode in [0,1])
        item=dict(model=model,task=task,first_calls_available=available)
        if available:
            left,right=pair[0][0],pair[1][0]
            same=left['response']['prompt_token_ids']==right['response']['prompt_token_ids']
            item.update(first_prompt_ids_equal=same,engine_sampling_equal=left['engine_sampling']==right['engine_sampling'],
                first_output_ids_equal=(left['response']['choices'][0]['token_ids']==right['response']['choices'][0]['token_ids']) if same else None,
                gpus=[left['gpu'],right['gpu']])
        first_turn_pairs.append(item)
    result=dict(passed=not failures,trials=trials,requests=details,incomplete_api=incomplete,failures=failures,first_turn_pairs=first_turn_pairs,
        status_counts=dict(Counter(row['status'] for row in rows.values())),
        scope='PRoot internal smoke; target logits/full-vocabulary Fidelity not run in this audit.')
    save(OUTPUT/('PROGRESS_AUDIT.json' if partial else 'SMOKE_AUDIT.json'),result)
    assert not failures,failures
    print('PASS',len(trials),'native trials;',len(details),'complete API replies;',len(incomplete),'timeout API records')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true')
    main(parser.parse_args().partial)
