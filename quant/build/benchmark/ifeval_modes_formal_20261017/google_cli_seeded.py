"""Run Google's unchanged IFEval CLI with the runner's deterministic RNG policy."""
import hashlib
from pathlib import Path
import random
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'build/autoquant/tools/google-research'))

import langdetect
from instruction_following_eval import evaluation_lib

langdetect.DetectorFactory.seed = 0
strict = evaluation_lib.test_instruction_following_strict
loose = evaluation_lib.test_instruction_following_loose


def seed_question(example):
    random.seed(int(hashlib.sha256(f'ifeval/{example.key}'.encode()).hexdigest()[:16], 16))


def strict_seeded(example, responses):
    seed_question(example)
    return strict(example, responses)


def loose_seeded(example, responses):
    seed_question(example)
    strict(example, responses)  # Match the strict-then-loose RNG consumption in the runner.
    return loose(example, responses)


evaluation_lib.test_instruction_following_strict = strict_seeded
evaluation_lib.test_instruction_following_loose = loose_seeded
runpy.run_module('instruction_following_eval.evaluation_main', run_name='__main__')
