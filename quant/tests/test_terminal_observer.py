import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from terminal_observer import CURRENT, AuditMiddleware, install_sampling_observer


def test_sampling_observer_calls_original_once_and_returns_same_object(monkeypatch):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    original=ChatCompletionRequest.to_sampling_params
    seen=[]
    def spy(self,*args,**kwargs):
        value=original(self,*args,**kwargs)
        seen.append(value)
        return value
    monkeypatch.setattr(ChatCompletionRequest,'to_sampling_params',spy)
    install_sampling_observer()
    event={};token=CURRENT.set(event)
    try:
        request=ChatCompletionRequest(model='local',messages=[dict(role='user',content='x')],
            temperature=1.0,top_p=.95,top_k=20,min_p=0,presence_penalty=0,
            repetition_penalty=1,seed=20261020)
        value=request.to_sampling_params(131072,{})
    finally: CURRENT.reset(token)
    assert len(seen)==1 and value is seen[0]
    assert event['engine_sampling']==dict(temperature=1.0,top_p=.95,top_k=20,min_p=0,
        presence_penalty=0,repetition_penalty=1,seed=20261020,max_tokens=131072)


def test_asgi_observer_preserves_exact_wire_bytes_and_omits_auth(monkeypatch,tmp_path):
    monkeypatch.setenv('TERMINAL_AUDIT_DIR',str(tmp_path))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    monkeypatch.setenv('TERMINAL_MODEL','bf16')
    payload=b'{"messages":[{"role":"user","content":"x"}]}'
    frames=[dict(type='http.response.start',status=200,headers=[]),
            dict(type='http.response.body',body=b'{"ok":true}',more_body=False)]
    async def app(scope,receive,send):
        assert (await receive())['body']==payload
        for frame in frames: await send(frame)
    async def receive(): return dict(type='http.request',body=payload,more_body=False)
    sent=[]
    async def send(frame): sent.append(frame)
    middleware=AuditMiddleware(app)
    asyncio.run(middleware(dict(type='http',path='/v1/chat/completions',headers=[
        (b'authorization',b'Bearer private'),(b'x-terminal-trial',b'trial-3')]),receive,send))
    assert sent==frames
    row=json.loads(next(tmp_path.glob('*.json')).read_text())
    assert row['status']=='complete' and row['gpu']=='3'
    assert row['identity']=={'x-terminal-trial':'trial-3'}
    assert 'private' not in json.dumps(row)
    assert row['request']==json.loads(payload)
