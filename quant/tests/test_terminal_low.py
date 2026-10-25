from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from terminal_bench import read, CONFIG as OLD_CONFIG
from terminal_low import CONFIG, classify


def test_low_protocol_preserves_sampling_and_requires_smoke_pause():
    old,new=read(OLD_CONFIG),read(CONFIG)
    for key in ['sampling','max_model_len','kv_cache_dtype','smoke_tasks','harbor_max_input_tokens']:
        assert new[key]==old[key]
    assert new['reasoning_effort']=='low'
    assert new['chat_template_kwargs']==dict(enable_thinking=True,preserve_thinking=True,reasoning_effort='low')
    assert new['mtp'] and new['num_speculative_tokens']==3
    assert new['stop_after_smoke'] and new['formal_requires_new_user_confirmation']
    assert new['candidate_models']=={'NaviQuant':'eNeural-Qwen3.8-27B-fast','Aqua':'rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm'}
    assert new['model_mapping_confirmed_by_user']
    assert len(set(new['models']))==3
    assert old['reasoning_effort']=='xhigh' and old['mtp'] is False


def test_infrastructure_failure_is_not_model_failure(tmp_path):
    for kind,expected in [('VerifierTimeoutError','VERIFIER_FAIL'),('AgentTimeoutError','TIMEOUT'),
                          ('APIConnectionError','API_FAILURE'),('EnvironmentStartError','ENVIRONMENT_FAIL')]:
        assert classify(dict(exception_info=dict(exception_type=kind)),tmp_path)['status']==expected
    data=dict(verifier_result=dict(rewards=dict(reward=0)))
    assert classify(data,tmp_path)['status']=='VERIFIER_FAIL'
    (tmp_path/'verifier').mkdir()
    (tmp_path/'verifier/ctrf.json').write_text('{}')
    assert classify(data,tmp_path)['status']=='MODEL_FAIL'
    data['verifier_result']['rewards']['reward']=1
    assert classify(data,tmp_path)['status']=='PASS'


def test_oci_root_directory_entry(tmp_path):
    import hashlib
    import tarfile
    from terminal_proot_images import unpack
    layer=tmp_path/'layer.tar'
    with tarfile.open(layer,'w') as archive:
        entry=tarfile.TarInfo('.')
        entry.type=tarfile.DIRTYPE
        entry.mode=0o755
        archive.addfile(entry)
    info=dict(layers=[dict(path=str(layer),digest='sha256:'+hashlib.sha256(layer.read_bytes()).hexdigest())])
    root=tmp_path/'root'
    root.mkdir()
    assert unpack(root,info)==[]
