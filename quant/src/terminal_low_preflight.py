"""Bounded, non-scoring Low/MTP3 API checks through real PRoot terminal tools."""
import argparse
import asyncio
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time
from unittest.mock import patch

import httpx
from terminal_bench import agent_config, read, save, services, server_command
from terminal_low import CONFIG, OUTPUT, SOURCE
from terminal_proot_images import prepare_image


def metrics(gpu):
    response=httpx.get(f'http://127.0.0.1:{read(CONFIG)["port_base"]+gpu}/metrics',timeout=5)
    response.raise_for_status()
    return {line.rsplit(' ',1)[0]:float(line.rsplit(' ',1)[1])
        for line in response.text.splitlines() if line.startswith('vllm:spec_decode_')
        and '_created' not in line and '_bucket' not in line}


async def worker(cfg,model,gpu,folder):
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial
    from harbor.llms.chat import Chat
    from harbor.agents.terminus_2.terminus_2 import Command
    await asyncio.to_thread(prepare_image,'regex-log')
    for mode in [False,True]:
        dest=folder/f'interleaved-{int(mode)}'
        dest.mkdir()
        ac=agent_config(cfg,model,gpu,cfg['seeds'][0],mode,f'synthetic/{model}/{mode}')
        ac.pop('n_concurrent')
        ac['kwargs']['llm_call_kwargs']['timeout']=120
        save(dest/'agent.json',ac)
        tc=TrialConfig.model_validate(dict(task=dict(path=str(SOURCE/'terminal-bench-2-1/tasks/regex-log')),
            trial_name='protocol',trials_dir=str(dest),agent=ac,
            environment=dict(type=None,import_path='terminal_proot:PRootEnvironment',delete=True)))
        trial=await Trial.create(tc)
        env,agent=trial.agent_environment,trial.agent
        before=metrics(gpu)
        began=time.time()
        turns=[]
        try:
            await env.start(False)
            await agent.setup(env)
            agent._reset_per_run_state()
            chat=Chat(agent._llm,interleaved_thinking=mode)
            command="printf 'LOW_MTP_%s\\n' OK\n"
            instruction=('This is a non-scored protocol diagnostic. First execute exactly '+repr(command)+
                '. Wait for terminal feedback containing LOW_MTP_OK. Only then finish with commands=[] and task_complete=true. '
                'Keep analysis and plan to one brief sentence each. Use the required JSON format.')
            prompt=agent._prompt_template.format(instruction=instruction,terminal_state='$ ')
            async with asyncio.timeout(120):
                for turn in range(2):
                    start=time.time()
                    response=await agent._query_llm(chat,prompt)
                    parsed=agent._parser.parse_response(response.content)
                    row=dict(turn=turn,seconds=time.time()-start,messages=copy.deepcopy(chat.messages),
                        response={**asdict(response),'usage':response.usage.model_dump(mode='json') if response.usage else None},
                        parser_error=parsed.error,parser_warning=parsed.warning)
                    turns.append(row)
                    save(dest/'turns.json',turns)
                    assert not parsed.error,parsed.error
                    assert response.prompt_token_ids and response.completion_token_ids
                    if turn==0:
                        assert len(parsed.commands)==1 and parsed.commands[0].keystrokes==command
                        timed,output=await agent._execute_commands([Command(keystrokes=command,duration_sec=0.3)],agent._session)
                        assert not timed and 'LOW_MTP_OK' in output
                        row['terminal_output']=output
                        prompt='New Terminal Output:\n'+output
                    else:
                        assert parsed.is_task_complete and not parsed.commands
                    save(dest/'turns.json',turns)
            after=metrics(gpu)
            delta={key:value-before.get(key,0) for key,value in after.items()}
            accepted=sum(value for key,value in delta.items() if key.startswith('vllm:spec_decode_num_accepted_tokens_total'))
            drafted=sum(value for key,value in delta.items() if key.startswith('vllm:spec_decode_num_draft_tokens_total'))
            assert drafted>0 and accepted>0,'No actual MTP acceptance'
            save(dest/'result.json',dict(passed=True,benchmark=False,seconds=time.time()-began,
                metrics_delta=delta,accepted=accepted,drafted=drafted,acceptance=accepted/drafted,
                summarizations=agent._summarization_count))
        finally:
            if agent._session: await agent._session.stop()
            await env.stop(True)
            trial._close_logger_handler()
        assert not read(dest/'protocol/proot/cleanup.json')['after']
        print(model,mode,'PASS',flush=True)


def launch(model,gpu):
    import terminal_bench
    cfg=read(CONFIG)
    if model not in cfg['models']:
        raise ValueError('Model identity must be confirmed in the new configuration first')
    folder=OUTPUT/'preflight'/model
    folder.mkdir(parents=True,exist_ok=False)
    cfg=copy.deepcopy(cfg)
    cfg['sampling']['max_tokens']=1024
    save(folder/'scope.json',dict(config=cfg,scored=False,diagnostic_output_cap=1024,
        diagnostic_deadline=120,num_speculative_tokens=3))
    def command(config,checkpoint,index):
        return server_command(config,checkpoint,index)+['--speculative-config',json.dumps(dict(method='mtp',num_speculative_tokens=3))]
    with patch.object(terminal_bench,'server_command',command):
        with services(cfg,model,[gpu],folder/'services'):
            asyncio.run(worker(cfg,model,gpu,folder))
    save(folder/'execution.json',dict(finished=True,audit_required=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    launch(args.model,args.gpu)
