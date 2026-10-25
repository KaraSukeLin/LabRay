"""Passive vLLM middleware: record wire messages, engine sampling and actual IDs.

The wrapper calls the original SamplingParams conversion exactly once and does
not change its arguments or result. Non-streaming is Harbor's existing path.
"""
import contextvars
import json
import os
from pathlib import Path
import time
import uuid

CURRENT = contextvars.ContextVar('terminal_request', default=None)
KEYS = ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty',
        'repetition_penalty', 'seed', 'max_tokens')


def install_sampling_observer():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    original = ChatCompletionRequest.to_sampling_params
    if getattr(original, '_terminal_observer', False):
        return

    def observed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        event = CURRENT.get()
        if event is not None:
            event['engine_sampling'] = {key: getattr(result, key) for key in KEYS}
        return result

    observed._terminal_observer = True
    ChatCompletionRequest.to_sampling_params = observed


class AuditMiddleware:
    def __init__(self, app):
        self.app = app
        self.folder = Path(os.environ['TERMINAL_AUDIT_DIR'])
        self.folder.mkdir(parents=True, exist_ok=True)
        install_sampling_observer()

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/v1/chat/completions':
            return await self.app(scope, receive, send)
        key = uuid.uuid4().hex
        event = dict(id=key, started_at=time.time(), gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                     model=os.environ['TERMINAL_MODEL'], status='incomplete',
                     identity={k.decode(): v.decode() for k, v in scope['headers']
                               if k.lower().startswith(b'x-terminal-')})
        incoming, outgoing = bytearray(), bytearray()
        token = CURRENT.set(event)

        async def read():
            message = await receive()
            if message['type'] == 'http.request':
                incoming.extend(message.get('body', b''))
            return message

        async def write(message):
            if message['type'] == 'http.response.start':
                event['http_status'] = message['status']
            elif message['type'] == 'http.response.body':
                outgoing.extend(message.get('body', b''))
                if not message.get('more_body', False):
                    event['status'] = 'complete'
            await send(message)

        try:
            await self.app(scope, read, write)
        except BaseException as exc:
            event['error'] = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            CURRENT.reset(token)
            event.update(finished_at=time.time(), elapsed_seconds=time.time()-event['started_at'])
            for name, value in [('request', incoming), ('response', outgoing)]:
                try:
                    event[name] = json.loads(value)
                except (ValueError, UnicodeError):
                    event[name+'_raw'] = value.decode(errors='replace')
            path = self.folder / (key+'.json')
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(event, ensure_ascii=False)+'\n')
            temp.replace(path)
