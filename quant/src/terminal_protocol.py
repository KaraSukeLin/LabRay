#!/usr/bin/env python3
"""Small, explicitly non-benchmark GPU checks of the unmodified Terminus2 stack."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import time

from terminal_bench import CONFIG, ROOT, agent_config, read, save, services, stop_process


async def worker(args):
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.llms.chat import Chat
    config=read(CONFIG)
    folder=args.output/f'gpu{args.gpu}'
    folder.mkdir(parents=True,exist_ok=True)
    modes=[False,True] if args.multiturn else [False]
    for mode in modes:
        dest=folder/f'interleaved-{int(mode)}'
        dest.mkdir(exist_ok=False)
        identity=f'synthetic/gpu{args.gpu}/thinking-{int(mode)}'
        ac=agent_config(config,config['models'][0],args.gpu,config['seeds'][0],mode,identity)
        agent=Terminus2(logs_dir=dest,model_name=ac['model_name'],**ac['kwargs'])
        agent._reset_per_run_state()
        chat=Chat(agent._llm,interleaved_thinking=mode)
        commands=["printf 'ROUND1_OK\\n'\n","printf 'ROUND2_OK\\n'\n"]
        instruction=('This is a synthetic protocol check, not a benchmark task. '
            'Execute exactly one command per turn. On the first turn execute exactly '+repr(commands[0])+
            '. Wait for its output. On the second turn execute exactly '+repr(commands[1])+
            '. Wait for its output. Only after receiving both outputs, finish with commands=[] '
            'and task_complete=true. Use the required JSON format.')
        prompt=agent._prompt_template.format(instruction=instruction,terminal_state='$ ')
        turns=[]
        for turn in range(3 if args.multiturn else 1):
            started=time.time()
            response=await agent._query_llm(chat,prompt)
            parsed=agent._parser.parse_response(response.content)
            assert not parsed.error, parsed.error
            assert response.reasoning_content and response.content, 'Reasoning/final missing'
            assert response.prompt_token_ids and response.completion_token_ids, 'Actual token IDs missing'
            row=dict(turn=turn,prompt=prompt,reasoning=response.reasoning_content,final=response.content,
                prompt_token_ids=response.prompt_token_ids,token_ids=response.completion_token_ids,
                elapsed_seconds=time.time()-started,parse_warning=parsed.warning,
                messages=list(chat.messages),summarizations=agent._summarization_count)
            turns.append(row)
            save(dest/'turns.json',turns)
            if turn<2:
                expected=commands[turn]
                assert len(parsed.commands)==1 and parsed.commands[0].keystrokes==expected, parsed
                # The host executes only these two exact allowlisted printf commands.
                result=subprocess.run(['/bin/sh','-c',expected],check=True,capture_output=True,text=True)
                row['tool_output']=result.stdout
                prompt='New Terminal Output:\n'+result.stdout+'\n$ '
            else:
                assert parsed.is_task_complete and not parsed.commands
            save(dest/'turns.json',turns)
        save(dest/'result.json',dict(passed=True,benchmark=False,turns=len(turns),
            interleaved_thinking=mode,summarizations=agent._summarization_count,
            actual_context=agent._llm.get_model_context_limit(),
            actual_output_limit=agent._llm.get_model_output_limit(),seed=config['seeds'][0]))
        print(json.dumps(dict(gpu=args.gpu,interleaved=mode,turns=len(turns),passed=True)),flush=True)


def launch(args):
    config=read(CONFIG)
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'scope.json',dict(benchmark=False,official_trials=0,gpus=args.gpus,
        config=config,note='Synthetic allowlisted shell feedback; no Docker/Verifier claim.'))
    with services(config,config['models'][0],args.gpus,args.output/'services'):
        children=[]
        logs=[]
        try:
            for gpu in args.gpus:
                argv=[str(ROOT/'.harbor-venv/bin/python'),str(Path(__file__).resolve()),'worker',
                      '--gpu',str(gpu),'--output',str(args.output)]
                if gpu==args.gpus[0] and not args.single_turn: argv+=['--multiturn']
                env=dict(os.environ,LITELLM_LOCAL_MODEL_COST_MAP='True',OPENAI_API_KEY='local-only')
                log=(args.output/f'worker-gpu{gpu}.log').open('w')
                logs.append(log)
                children.append(subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,
                                                  start_new_session=True))
            while any(p.poll() is None for p in children):
                assert not any(p.poll() not in [None,0] for p in children), 'Protocol worker failed; inspect its log'
                time.sleep(1)
            assert all(p.returncode==0 for p in children), 'Protocol worker failed; inspect its log'
        finally:
            for child in children: stop_process(child)
            for log in logs: log.close()
    save(args.output/'STATE.json',dict(state='PAUSED',services_stopped=True,official_trials=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','worker'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=[1,3,4])
    p.add_argument('--gpu',type=int)
    p.add_argument('--multiturn',action='store_true')
    p.add_argument('--single-turn',action='store_true')
    a=p.parse_args()
    def terminate(signum,frame): raise KeyboardInterrupt('Protocol interrupted')
    signal.signal(signal.SIGTERM,terminate)
    if a.command=='worker': asyncio.run(worker(a))
    else: launch(a)
