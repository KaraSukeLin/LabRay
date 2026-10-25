"""Read-only evidence audit and progress report for the Low/MTP3 smoke work."""
from collections import Counter
from pathlib import Path
import time

from terminal_bench import read, save, gpu_inventory, prepare
from terminal_low import CONFIG, OUTPUT


def main():
    cfg=read(CONFIG)
    prepare(cfg)
    coverage=read(OUTPUT/'coverage.json')
    rows=[]
    for task in cfg['smoke_tasks']:
        item=coverage['tasks'][task]
        assert item['status']=='CONTROLS_PASS',task
        controls=item['controls']
        for agent,expected in [('oracle',1),('nop',0)]:
            row=controls[agent]
            assert row['reward']==expected and row['cleanup_clean']
            native=read(row['result'])
            assert native['finished_at'] and not native.get('exception_info')
            verifier=Path(row['result']).parent/'verifier/ctrf.json'
            ctrf=read(verifier)['results']['summary']
            assert ctrf['tests']>0
            assert ctrf.get('failed',0)==0 if expected else ctrf.get('failed',0)>0
            assert not read(Path(row['result']).parent/'proot/cleanup.json')['after']
        rows.append(f'| {task} | 1 | 0 | PASS |')
    preflight=OUTPUT/'preflight/Qwen-Qwen3.8-27B'
    assert read(preflight/'AUDIT.json')['passed']
    ledger=read(preflight/'services/services.json')
    assert all(row.get('stopped_at') for row in ledger)
    model_rows=read(OUTPUT/'model-audit.json')
    assert len(model_rows)==3 and all(row['structural_integrity_passed'] for row in model_rows)
    text='''# Terminal-Bench 2.1 Low／MTP3 準備進度

2026-10-21。**五題 PRoot 控制與 BF16 兩模式 API 預檢通過。Smoke 0／30，正式 0／267。**

本次是 PRoot 非標準執行協定下的內部比較，不能當作官方 Docker Leaderboard 成績。使用者追加要求：Smoke 完成後暫停等候確認，禁止自動啟動正式批次。當前尚未完成 Smoke，因此不宣稱目標已完成。

| 官方任務 | Oracle reward | NOP reward | Verifier／cleanup |
|---|---:|---:|---|
'''+ '\n'.join(rows)+'''

控制皆使用固定官方 commit、映像 digest、原答案與 Verifier，各題 timeout 沿用官方值。完整 89 題清單已建立；實測控制覆蓋率為 **5／89**，其餘 **84 題 UNVALIDATED**，不是不相容。五題控制皆在觀察本輪任何模型 Smoke 結果前固定。

初次 modernize 在解壓 OCI 根目錄 `.` 項目時被 pilot 的路徑檢查拒絕。已用最小測試重現並修正新 backend；原 attempt 保留。attempt 1 的五題十個控制全部完成，並非重跑模型失敗直到成功。

BF16 Low／MTP3 預檢使用真實 Harbor／Terminus／PRoot／tmux，每模式兩次模型 API 回覆、一次工具往返。實際 prompt IDs 與 CPU 重建逐 token 相同；reasoning/final parser、sampling、seed、Low 指示、False 省略歷史 reasoning／True 保留歷史 reasoning，以及程序清理均通過。預檢 cap=1024、API deadline=120 秒，屬非計分診斷，不套用於 Smoke。

'''
    for mode in [0,1]:
        r=read(preflight/f'interleaved-{mode}/result.json')
        text+=f'- interleaved={bool(mode)}：MTP 接受 {r["accepted"]:.0f}／draft {r["drafted"]:.0f} tokens（{r["acceptance"]:.1%}）。\n'
    text+=f'\nBF16 service 含載入／終端準備的實測成本 {sum(row["gpu_hours"] for row in ledger):.6f} GPU-hours。未執行五 GPU 並行，因此沒有五卡利用率或穩定性結論。\n'
    text+='''
三個本機 checkpoint 已核對 shard index、tensor bounds、總大小、config SHA256 與全部權重 SHA256。BF16／eNeural 使用歷史已驗證 fingerprint；PrismaAQUA 使用本機固定 revision 的 Hugging Face LFS SHA256 metadata。BF16／eNeural 的舊 CRC 清單與目前 chat_template、generation_config、tokenizer_config 不同，差異完整記錄；實際 serving 使用的共同 BF16 tokenizer/template 與已固定 Terminal manifest 完全相同，沒有在本輪更換。

**模型身分仍待使用者確認**：本機 `eNeural-Qwen3.8-27B-fast` 舊報告稱 NaviQuant Fast；`rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm` card 稱 PrismaQuant AQUA-AURA。但附件明確要求不直接把 NaviQuant／Aqua 當成 eNeural／Prisma，因此兩候選尚未載入或排入 Smoke。不能將結構完整性當作量化 runtime／MTP 通過。

已新增 durable SQLite queue 與 Smoke-only scheduler 草稿：唯一 Trial ID、原結果不可覆寫、每 GPU 一個 Trial、同模型優先、模型切換、1→2→5 真實 Agent overlap Gate、GPU／cgroup／CPU/RAM/I/O telemetry、獨立 logs/results、基礎設施失敗停止與 Smoke 後暫停。**尚未執行 scheduler 整合驗證**，不能當作已通過並行 Gate；模型身分與候選 runtime Gate 未通過時拒絕啟動。中斷的 claimed Trial 保留，不自動重試。正式批次入口尚未啟用。

針對性 CPU tests：15 passed，見 targeted-tests.log。未跑完整 test suite。Low trajectory full-vocabulary GPU Fidelity、正式統計與能力／效率比較皆尚未執行，需在後續對應階段完成；不從 synthetic 預檢推論模型能力。

證據：coverage.json、controls/、images/、model-audit.json、preflight/Qwen-Qwen3.8-27B/AUDIT.json、preflight-bf16-audit.log、targeted-tests.log。舊 Xhigh 配置／pilot 結果／官方原檔保持不變。
'''
    text+='\n續跑補強：worker 若在 Harbor 已保存完整結果後中止，scheduler 會核對已死亡 owner、原 Trial ID、模型、任務、mode、seed 與乾淨的 cleanup receipt，再將同一原始結果回填 SQLite；不建立新 attempt，不重跑完成 Trial。仍活動或清理未完成的 Trial 不恢復計分。9 個相關 CPU tests 通過，見 resume-targeted-tests.log。尚無 GPU scheduler 整合驗證，Smoke 仍為 0／30。\n'
    (OUTPUT/'REPORT.md').write_text(text)
    state=dict(state='WAITING_FOR_MODEL_IDENTITY',time=time.time(),environment_inventory=89,
        controls_pass=5,controls_denominator=89,smoke_completed=0,smoke_planned=30,
        formal_trials_started=0,stop_after_smoke=True,formal_authorized=False,
        bf16_preflight_audit_passed=True,candidate_runtime_verified=False,
        scheduler_integration_verified=False,gpus=gpu_inventory())
    save(OUTPUT/'STATE.json',state)
    save(OUTPUT/'preparation-audit.json',dict(passed=True,scope='Five official controls and BF16 bounded protocol preparation only',
        controls=10,bf16_api_requests=4,smoke=0,formal=0,goal_complete=False))
    print(state)


if __name__=='__main__': main()
