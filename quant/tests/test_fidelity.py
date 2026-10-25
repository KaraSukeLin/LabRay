from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fidelity import aggregate, compare_arrays, full_logprobs, sanity_passed


def test_prompt_alignment_and_duplicate_target():
    # Position 1 predicts token 2; position 2 predicts token 1.
    flat = SimpleNamespace(start_indices=[0, 0, 4], end_indices=[0, 4, 8],
                           token_ids=[2, 0, 1, 2, 1, 2, 0, 1],
                           logprobs=np.log([.6, .1, .3, .6, .7, .2, .1, .7]))
    logp = full_logprobs(flat, [0, 2, 1], 3)
    np.testing.assert_allclose(np.exp(logp), [[.1, .3, .6], [.1, .7, .2]], rtol=1e-6)
    metrics = aggregate([compare_arrays(logp, logp, [0, 2, 1])])
    assert metrics['ppl'] == pytest.approx(np.exp(-np.log([.6, .7]).mean()))
    assert sanity_passed(metrics)


def test_full_kl_direction_and_flip():
    p = np.log(np.array([[.6, .3, .1], [.1, .7, .2]], dtype=np.float32))
    q = np.log(np.array([[.2, .7, .1], [.2, .6, .2]], dtype=np.float32))
    metrics = aggregate([compare_arrays(p, q, [0, 1, 2])])
    expected = np.sum(np.exp(p) * (p - q), axis=1).mean()
    assert metrics['kl_bf16_to_candidate'] == pytest.approx(expected, abs=1e-7)
    assert metrics['top1_flip_rate'] == .5
    assert metrics['ppl'] == pytest.approx(np.exp(-np.log([.7, .2]).mean()))
    assert not sanity_passed(metrics)


def test_reject_topk_as_full_vocab():
    flat = SimpleNamespace(start_indices=[0, 0], end_indices=[0, 2],
                           token_ids=[0, 2], logprobs=[-.5, -1.2])
    with pytest.raises(ValueError, match='incomplete vocabulary'):
        full_logprobs(flat, [0, 2], 3)


def test_masked_vocab_and_token_weighted_aggregation():
    p = np.array([[-.2, np.log1p(-np.exp(-.2)), -np.inf]], dtype=np.float32)
    a = compare_arrays(p, p, [0, 1])
    b = compare_arrays(np.repeat(p, 2, axis=0), np.repeat(p, 2, axis=0), [0, 0, 0])
    metrics = aggregate([a, b])
    assert metrics['tokens'] == 3
    assert metrics['kl_bf16_to_candidate'] == 0
    assert metrics['ppl'] == pytest.approx(np.exp((a['nll'] + b['nll']) / 3))
