"""Summarize native smoke evidence and sampled resource costs without inference."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import json
import statistics
import sqlite3

from terminal_bench import read,save
from terminal_low import CONFIG,OUTPUT


def seconds(interval):
    if not interval or not interval.get('finished_at'): return None
    parse=lambda value:datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
    return parse(interval['finished_at'])-parse(interval['started_at'])


def main(partial=False):
    cfg=read(CONFIG)
    connection=sqlite3.connect('file:'+str(OUTPUT/'smoke.sqlite')+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row
    rows=[dict(row) for row in connection.execute('SELECT * FROM trials ORDER BY model,task,mode')]
    if not partial:
        assert len(rows)==30 and all(row['status'] not in {'PENDING','RUNNING'} for row in rows)
        assert read(OUTPUT/'SMOKE_AUDIT.json')['passed']
    wire=defaultdict(lambda:dict(prompt_tokens=0,output_tokens=0,reasoning_tokens=0,replies=0,incomplete=0,api_seconds=0.))
    for path in (OUTPUT/'services').rglob('audit/*.json'):
        event=read(path)
        key=event['identity']['x-terminal-identity'].rsplit('/',1)[-1]
        response=event.get('response')
        if not isinstance(response,dict) or not response.get('choices'):
            wire[key]['incomplete']+=1
            continue
        usage=response.get('usage') or {}
        wire[key]['prompt_tokens']+=usage.get('prompt_tokens',0)
        wire[key]['output_tokens']+=usage.get('completion_tokens',0)
        wire[key]['reasoning_tokens']+=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens',0)
        wire[key]['replies']+=1
        wire[key]['api_seconds']+=event['elapsed_seconds']
    details=[]
    for row in rows:
        folder=OUTPUT/'smoke'/row['id']
        if row['status'] in {'PENDING','RUNNING'}: continue
        native=read(folder/'trial/result.json')
        trajectory=read(folder/'trial/agent/trajectory.json')
        usage=trajectory.get('final_metrics') or {}
        mtp=read(folder/'mtp.json')['delta'] if (folder/'mtp.json').exists() else {}
        accepted=sum(value for key,value in mtp.items() if key.startswith('vllm:spec_decode_num_accepted_tokens_total'))
        drafted=sum(value for key,value in mtp.items() if key.startswith('vllm:spec_decode_num_draft_tokens_total'))
        details.append(dict(id=row['id'],model=row['model'],task=row['task'],interleaved=bool(row['mode']),status=row['status'],
            reward=(native.get('verifier_result') or {}).get('rewards',{}).get('reward'),
            prompt_tokens=wire[row['id']]['prompt_tokens'],output_tokens=wire[row['id']]['output_tokens'],
            reasoning_tokens=wire[row['id']]['reasoning_tokens'],complete_api_seconds=wire[row['id']]['api_seconds'],complete_api_replies=wire[row['id']]['replies'],incomplete_api_replies=wire[row['id']]['incomplete'],
            native_prompt_tokens=usage.get('total_prompt_tokens'),native_output_tokens=usage.get('total_completion_tokens'),
            tool_calls=sum(len(step.get('tool_calls') or []) for step in trajectory.get('steps',[])),
            agent_seconds=seconds(native.get('agent_execution')),wall_seconds=seconds(native),
            mtp_accepted=accepted,mtp_drafted=drafted,mtp_acceptance=accepted/drafted if drafted else None,
            gpu=row['gpu'],result=str(folder/'trial/result.json')))
    services=[]
    for path in sorted((OUTPUT/'services').rglob('services.json')):
        for service in read(path):
            services.append(dict(gpu=service['gpu'],model=service['model'],started_at=service['started_at'],
                stopped_at=service.get('stopped_at'),load_seconds=service.get('ready_at',service['started_at'])-service['started_at'],
                gpu_hours=service.get('gpu_hours'),source=str(path)))
    samples=[json.loads(line) for line in (OUTPUT/'resources.jsonl').read_text().splitlines()]
    samples.sort(key=lambda row:row['time'])
    util={gpu:0. for gpu in range(5)}
    no_service={gpu:0. for gpu in range(5)}
    phases=defaultdict(float)
    observed=0.
    for before,after in zip(samples,samples[1:]):
        elapsed=after['time']-before['time']
        observed+=elapsed
        phases[before['stage']]+=elapsed
        for gpu in range(5):
            item=before['gpus'][str(gpu)]
            util[gpu]+=elapsed*item['utilization']
            no_service[gpu]+=elapsed*(item['used_mb']<500)
    gpu_summary=[dict(gpu=gpu,sampled_seconds=observed,mean_utilization_percent=util[gpu]/observed if observed else None,
        no_service_seconds=no_service[gpu],peak_vram_mib=max(sample['gpus'][str(gpu)]['used_mb'] for sample in samples),service_gpu_hours=sum(s['gpu_hours'] or 0 for s in services if s['gpu']==gpu),
        service_loads=sum(s['gpu']==gpu for s in services)) for gpu in range(5)]
    result=dict(final=not partial,smoke_planned=30,completed=len(details),formal_trials=0,
        status_counts=dict(Counter(row['status'] for row in rows)),trials=details,services=services,gpus=gpu_summary,
        phase_sample_seconds=dict(phases),stop_after_smoke=True,
        note='PRoot nonstandard internal smoke. Sampled utilization is not a pure decode benchmark; no capability stability estimate from single attempts.')
    save(OUTPUT/('PROGRESS_SUMMARY.json' if partial else 'SMOKE_SUMMARY.json'),result)
    if partial:
        print(result['status_counts']);return
    assert all(s['stopped_at'] for s in services),'Model services must stop before final smoke handoff'
    text='# Terminal-Bench 2.1 Low／MTP3 Smoke 報告\n\n'
    text+='**30／30 Smoke 已完成，正式 0／267；依使用者要求暫停等待確認。** 本次為 PRoot 非標準協定下的內部比較，不是官方 Docker Leaderboard 成績。\n\n'
    text+='固定 BF16、NaviQuant=eNeural Fast、Aqua=PrismaAQUA 5.5-bit；三模型五題各跑 False／True 一次。Low、MTP3、seed=20261020、131072 共用 output budget、262144 context、BF16 KV、原官方 Agent／Verifier timeout。False 為主要模式，True 為獨立診斷。Oracle／NOP／預檢不混入 30 個 Smoke。\n\n'
    text+='| 模型 | interleaved | reward=1／5 | 分類 | Agent 秒中位數 | 工具呼叫 | MTP 接受／draft |\n|---|---|---:|---|---:|---:|---:|\n'
    for model in cfg['models']:
        for mode in [False,True]:
            cell=[row for row in details if row['model']==model and row['interleaved']==mode]
            assert len(cell)==5
            med=statistics.median(row['agent_seconds'] for row in cell if row['agent_seconds'] is not None)
            text+=f'| {model} | {mode} | {sum(row["reward"]==1 for row in cell)}/5 | {dict(Counter(row["status"] for row in cell))} | {med:.1f} | {sum(row["tool_calls"] for row in cell)} | {sum(row["mtp_accepted"] for row in cell):.0f}/{sum(row["mtp_drafted"] for row in cell):.0f} |\n'
    text+='\n| 模型 | Smoke service GPU-hours | 完整 API output tokens | reasoning tokens | 完整 API tokens/s | 模型載入次數 |\n|---|---:|---:|---:|---:|---:|\n'
    for model in cfg['models']:
        cell=[row for row in details if row['model']==model]
        loaded=[row for row in services if row['model']==model]
        rate=sum(row['output_tokens'] for row in cell)/sum(row['complete_api_seconds'] for row in cell)
        text+=f'| {model} | {sum(row["gpu_hours"] or 0 for row in loaded):.4f} | {sum(row["output_tokens"] for row in cell)} | {sum(row["reasoning_tokens"] for row in cell)} | {rate:.2f} | {len(loaded)} |\n'
    loads=[row['load_seconds'] for row in services]
    switches=0
    for gpu in range(5):
        sequence=sorted([row for row in services if row['gpu']==gpu],key=lambda row:row['started_at'])
        switches+=sum(before['model']!=after['model'] for before,after in zip(sequence,sequence[1:]))
    text+=f'\nSmoke 模型服務合計 {sum(row["gpu_hours"] or 0 for row in services):.4f} GPU-hours；實際跨模型切換 {switches} 次。服務啟動到 health-ready 中位數 {statistics.median(loads):.1f} 秒，範圍 {min(loads):.1f}–{max(loads):.1f} 秒；包含初始化與權重載入，並非純磁碟讀取。\n'
    text+='\n| GPU | 取樣平均 utilization | 取樣峰值 VRAM MiB | 模型服務 GPU-hours | 載入次數 | 無服務取樣秒數 |\n|---|---:|---:|---:|---:|---:|\n'
    for row in gpu_summary:
        text+=f'| {row["gpu"]} | {row["mean_utilization_percent"]:.1f}% | {row["peak_vram_mib"]} | {row["service_gpu_hours"]:.4f} | {row["service_loads"]} | {row["no_service_seconds"]:.1f} |\n'
    text+='\nToken 成本以全部完整 wire API usage 累計，包含模型格式錯誤後的原生修正回合；另保留 native trajectory totals。完整 API tokens/s 為完整回覆 output tokens 除以這些 API 的總耗時，含 prefill，不是純 decode 速度。中斷 API 無法回收的 token 不捏造，另列 incomplete 計數。\n'
    text+='\n逐題原始結果、token／工具／timeout／MTP 與載入成本在 SMOKE_SUMMARY.json；完整 prompt IDs、parser、history、trajectory、Verifier、cleanup audit 在 SMOKE_AUDIT.json。單次 timeout 的未完成 API 會另外列出，沒有捏造其 output token IDs。\n\n'
    text+='1→2→5 Gate 的實際 Agent 重疊證據在 concurrency-1.json、concurrency-2.json、concurrency-5.json。前段為必要 Gate 而保留 GPU；之後以五卡工作佇列執行，優先接續同模型，佇列清空才換模型。取樣 utilization 包含載入、Agent 工具、Verifier 等待及尾段工作量不足，不能以低於 100% 判定無效排程，也未用無意義重跑填滿 GPU。CPU／RAM／I/O 壓力保留在 resources.jsonl，這些是共享環境量測，不是每題隔離的 cgroup 成本。\n\n'
    if (OUTPUT/'TERMINUS_PARSE_AUDIT.json').exists():
        text+='原始 Terminus parser 的逐回覆 error／warning 與各模型統計另存 TERMINUS_PARSE_AUDIT.json；這些是模型格式診斷，不覆寫原 Trial reward 或任意修補回覆。\n\n'
    if (OUTPUT/'concurrency-5-observed.json').exists():
        observed=read(OUTPUT/'concurrency-5-observed.json')
        text+=f'五卡 Gate 已由原始時間證據驗證至少 {observed["conservative_overlap_seconds"]:.2f} 秒五 Agent 重疊。原先整批等待曾讓已完成卡閒置；初批四個 Trial 完成並清理後，續跑監督暫停原協調器，保留較慢 Trial／服務繼續執行，讓空閒卡以原 worker 接續工作；所有工作完成後恢復原協調器。詳見 concurrency-5-observed.json、continuation-controller.json 與 continuation.log。沒有重跑原 Trial 或改推論／評分設定。\n\n'
    text+='PRoot 原始環境控制覆蓋為 5／89；其餘 84 題尚未驗證，不以它們不相容作結論。原 Oracle=1、NOP=0、原 Verifier 不改。H100 上 FP4 層採 Marlin weight-only 路徑，非原生 FP4 計算；原量化 checkpoint 與 native MTP3 已由三模型 runtime audit 核對。\n\n'
    text+='本輪每個 model/task/mode 只有一次，不能估計 seed 穩定性，亦不足以隔離多模型並行對模型成績的因果影響。五卡 overlap 與原始 infrastructure 分類是工程證據，不能當作「並行完全不影響結果」的證明。Smoke 五題分數不是 89 題正式能力保留率。Full-vocabulary target-model Fidelity 與正式比較尚待後續階段；沒有以 draft logits 代替 target logits。\n'
    (OUTPUT/'REPORT.md').write_text(text)
    print('FINAL SMOKE REPORT',len(details),'trials; formal=0')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true')
    main(parser.parse_args().partial)
