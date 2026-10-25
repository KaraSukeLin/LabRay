"""Rebuild all saved Phase 4 reports without weights, inference, or package installs."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit('Use Python 3.12, as in the recorded experiment; confidence ties can change on Python 3.11.')
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = PROJECT/'experiments/orsharc'
sys.path.insert(0, str(EXPERIMENT/'src'))
from b_orsharc.analysis import report

def compare(actual, expected, path=''):
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            compare(actual[key], expected[key], path+'/'+key)
    elif isinstance(actual, list) and isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, path+'/'+str(index))
    elif isinstance(actual, float) or isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True, help='New directory for regenerated reports')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
runs = EXPERIMENT/'artifacts/runs/phase4'
partitions = ['dev_select', 'dev_calibrate', 'test']
checked = 0
for condition in sorted(runs.iterdir()):
    if not condition.is_dir():
        continue
    target = args.output/'runs'/condition.name
    target.mkdir(parents=True)
    (target/'calibration.json').symlink_to(condition/'calibration.json')
    for part in partitions:
        (target/part).symlink_to(condition/part, target_is_directory=True)
        result = report(condition/part, EXPERIMENT/'artifacts/prepared', target/'reports'/part, condition/'calibration.json')
        saved = json.loads((condition/'reports'/part/'metrics.json').read_text())
        compare(result, saved, condition.name+'/'+part)
        checked += 1
spec = importlib.util.spec_from_file_location('phase4_summary', EXPERIMENT/'artifacts/reports/phase4/build_summary.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
summary.RUNS = args.output/'runs'
summary.OUT = args.output/'summary'
summary.main()
regenerated = json.loads((summary.OUT/'summary.json').read_text())
original = json.loads((EXPERIMENT/'artifacts/reports/phase4/summary.json').read_text())
for value in [regenerated, original]:
    value.pop('generated_utc')
compare(regenerated, original, 'Phase4 aggregate')
receipt = dict(conditions=checked//3, partitions=checked, full_metrics_equal_within_1e12_tolerance=True, aggregate_equal_except_generation_timestamp=True, inference=False)
(args.output/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
