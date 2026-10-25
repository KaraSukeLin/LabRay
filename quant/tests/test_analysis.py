import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from analyze import cluster_interval, statistic
from fidelity import compare_arrays


def test_paired_document_bootstrap_preserves_correlated_fields():
    # Within-document values vary greatly; the paired NLL effect is constant.
    parts = [dict(document_id=str(i), domain='a' if i < 8 else 'b', context=128,
                  nll=float(100 + i * i) + 3, ref_nll=float(100 + i * i),
                  kl_sum=.4, flips=2, tokens=100) for i in range(16)]
    point, ci = cluster_interval(parts, pair=True, resamples=999)
    assert point[0] == pytest.approx(.03)
    np.testing.assert_allclose([ci.low[0], ci.high[0]], [.03, .03], atol=1e-14)
    assert point[1] == pytest.approx(np.exp(.03))
    assert point[3] == pytest.approx(.004)
    assert point[4] == pytest.approx(.02)


def test_score_start_excludes_context_without_shift_error():
    ids = [0, 2, 1, 0, 2]
    logp = np.log(np.array([[.1, .2, .7], [.2, .6, .2], [.8, .1, .1], [.2, .1, .7]], dtype=np.float32))
    # score_start=3 means score input tokens 3 and 4 using causal rows 2 and 3.
    result = compare_arrays(logp[2:], logp[2:], ids[2:])
    assert result['tokens'] == 2
    assert result['nll'] == pytest.approx(-np.log([.8, .7]).sum(), abs=1e-6)


def test_weighted_statistic_and_duplicate_cluster_rejection():
    a = np.array([[2, 3], [1, 2], [.1, .2], [0, 1], [1, 2]])
    b = np.array([[5], [4], [.3], [1], [3]])
    result = statistic(a, b)
    assert result[0] == pytest.approx(np.exp(10 / 6))
    assert result[2] == pytest.approx(3 / 6)
    assert result[4] == pytest.approx(.1)
    p = dict(document_id='same', domain='a', context=128)
    with pytest.raises(ValueError, match='joint clustering'):
        cluster_interval([p, p], resamples=99)
