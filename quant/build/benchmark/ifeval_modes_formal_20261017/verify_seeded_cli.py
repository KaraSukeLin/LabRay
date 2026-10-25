"""Reuse the verified IFEval audit with its saved per-question RNG policy."""
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

import ifeval_modes

OUT = Path(__file__).resolve().parent
original_run = ifeval_modes.subprocess.run


def run_seeded(command, **kwargs):
    if command[1:3] == ['-m', 'instruction_following_eval.evaluation_main']:
        command = [command[0], str(OUT / 'google_cli_seeded.py'), *command[3:]]
    return original_run(command, **kwargs)


ifeval_modes.subprocess.run = run_seeded
ifeval_modes.verify(SimpleNamespace(
    runs=OUT,
    output=OUT / 'verification_seeded_20261020',
))
