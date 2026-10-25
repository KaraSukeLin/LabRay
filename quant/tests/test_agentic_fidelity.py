import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from agentic_fidelity import (LOW_CONFIG, completed_payloads, task_from_identity,
                              validate_source_request)


def test_source_trajectory_must_match_low_mtp3_template():
    low = json.loads(LOW_CONFIG.read_text())['chat_template_kwargs']
    validate_source_request({'chat_template_kwargs': low})
    with pytest.raises(ValueError, match='pinned Low/MTP3'):
        validate_source_request({'chat_template_kwargs': {'reasoning_effort': 'xhigh'}})


def test_incomplete_audit_with_null_response_is_skipped():
    assert completed_payloads({'status': 'incomplete', 'response': None}) is None
    assert completed_payloads({'status': 'complete', 'http_status': 200,
                               'response': None}) is None


def test_formal_task_comes_from_unique_trial_identity():
    sessions = {'trial__agent': {'task-a', 'task-b'}}
    assert task_from_identity('formal/Qwen/task-b/unique-trial',
                              'trial__agent', sessions) == 'task-b'


def test_formal_identity_task_must_exist_in_source_sessions():
    sessions = {'trial__agent': {'task-a'}}
    with pytest.raises(ValueError, match='absent from session'):
        task_from_identity('formal/Qwen/task-b/unique-trial',
                           'trial__agent', sessions)
