from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from benchmark_budget import TASK_COUNTS
from summarize_benchmark_pilot import subset_cost


def test_task_subset_charges_one_whole_model_initialization():
    rows=[dict(task=t,typical_gpu_hours=10+TASK_COUNTS[t]/4001,
               optimistic_gpu_hours=5+TASK_COUNTS[t]/4001,
               conservative_gpu_hours=(15+TASK_COUNTS[t]/4001)*1.25)
          for t in TASK_COUNTS]
    assert subset_cost(rows,3600,list(TASK_COUNTS))==31.
    np.testing.assert_allclose(subset_cost(rows,3600,['ifeval']),11.)
