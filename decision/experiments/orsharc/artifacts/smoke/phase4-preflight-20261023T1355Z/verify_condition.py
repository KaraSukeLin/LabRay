#!/usr/bin/env python3
"""Verify a complete Experiment B condition and write a JSON receipt."""

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED = {"dev_select": 456, "dev_calibrate": 649, "test": 2373}
REQUESTS_AND_QUESTIONS = {
    "single_choice": (1, 1),
    "hierarchical": (2, 2),
    "parallel_noul": (1, 3),
    "noul": (1, 1),
    "choice2": (1, 1),
    "score5": (1, 1),
}


def load(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--task", choices=("benchmark", "sufficiency"), required=True)
    ap.add_argument("--strategy", choices=REQUESTS_AND_QUESTIONS, required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    root = args.run
    cal_path = root / "calibration.json"
    cal = load(cal_path)
    receipt = {"condition": root.name, "partitions": {}, "reports_valid": True}
    expected_labels = {"Yes", "No", "Inquire"} if args.task == "benchmark" else {"insufficient", "sufficient"}
    specs = {}

    for partition, n in EXPECTED.items():
        run = root / partition
        manifest, complete = load(run / "manifest.json"), load(run / "complete.json")
        spec = manifest["spec"]
        raw = (run / "predictions.jsonl").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.splitlines()]
        ids = [row["example_id"] for row in rows]

        assert spec["task"] == args.task and spec["strategy"] == args.strategy
        assert spec["partition"] == partition and spec["limit"] == 0
        assert len(rows) == n == complete["n"] == len(spec["ids"])
        assert ids == spec["ids"] and len(ids) == len(set(ids))
        assert manifest["identity"] == complete["identity"]
        assert complete["predictions_sha256"] == digest
        assert all(row["run_identity"] == manifest["identity"] for row in rows)
        assert all(not any(key in row for key in ("gold", "label", "target", "benchmark_gold")) for row in rows)
        for row in rows:
            requests, questions = REQUESTS_AND_QUESTIONS[args.strategy]
            assert row["request_count"] == requests
            assert row["question_count"] == questions
            probs = row["probabilities"]
            assert set(probs) == expected_labels
            assert all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in probs.values())
            assert math.isclose(sum(map(float, probs.values())), 1.0, rel_tol=0, abs_tol=1e-5)

        metrics = load(root / "reports" / partition / "metrics.json")
        assert metrics["run_identity"] == manifest["identity"]
        assert metrics["condition"] == cal["condition"]
        assert metrics["calibration_sha256"] == hashlib.sha256(cal_path.read_bytes()).hexdigest()
        assert not metrics["smoke_only"]
        group_counts = {"All": n}
        group_counts.update({split: sum(row["rule_split"] == split for row in rows) for split in ("Seen", "Unseen")})
        for version in ("raw", "calibrated"):
            for split, count in group_counts.items():
                assert metrics["metrics"][version][split]["n"] == count

        specs[partition] = (manifest, complete)
        receipt["partitions"][partition] = {
            "n": n,
            "seen": group_counts["Seen"],
            "unseen": group_counts["Unseen"],
            "prediction_sha256": digest,
            "finite_normalized_probabilities": True,
        }
        if partition == "test":
            receipt["test_metrics"] = {
                version: {
                    split: {key: metrics["metrics"][version][split][key] for key in ("accuracy", "macro_f1", "ece", "brier", "nll", "aurc")}
                    for split in ("All", "Seen", "Unseen")
                }
                for version in ("raw", "calibrated")
            }
            receipt["test_cost"] = metrics["metrics"]["raw"]["All"].get("cost")

    select_manifest, select_complete = specs["dev_select"]
    calibrate_manifest, calibrate_complete = specs["dev_calibrate"]
    assert cal["selection_run"] == select_manifest["identity"]
    assert cal["calibration_run"] == calibrate_manifest["identity"]
    assert cal["selection_predictions_sha256"] == select_complete["predictions_sha256"]
    assert cal["calibration_predictions_sha256"] == calibrate_complete["predictions_sha256"]
    assert select_manifest["spec"]["code_sha256"] == calibrate_manifest["spec"]["code_sha256"]
    assert 0.05 <= cal["temperature"] <= 20.0
    if args.task == "sufficiency":
        sweep = cal["threshold_sweep_dev_select"]
        assert len(sweep) == 99
        chosen = max(sweep, key=lambda row: (row["accuracy"], -abs(row["threshold"] - 0.5), -row["threshold"]))
        assert cal["threshold"] == chosen["threshold"]
    else:
        assert cal["threshold"] == 0.5 and cal["threshold_sweep_dev_select"] == []

    receipt["calibration"] = {
        "threshold": cal["threshold"],
        "temperature": cal["temperature"],
        "threshold_sweep_points": len(cal["threshold_sweep_dev_select"]),
        "selection_run": cal["selection_run"],
        "calibration_run": cal["calibration_run"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
