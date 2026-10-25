"""A few deterministic Phase 1 fixtures, not a model evaluation suite."""

import math
import tempfile
from pathlib import Path

from .analysis import fit_temperature, metrics, report, temperature
from .common import read_json, rows, write_json, write_rows
from .data import validate_prepared
from . import backends
from .runner import infer
from .tasks import aggregate, keys, questions, state, training_example


def check(prepared=None):
    example = {
        "oracle_rule": "You qualify if you are over 18.",
        "question": "Do I qualify?",
        "scenario": "I am 25.",
        "history": [],
        "example_id": "fixture",
        "evidence": "EVIDENCE_SECRET",
        "gold": "GOLD_SECRET",
    }
    assert "SECRET" not in state(example)
    for task, strategies in (
        ("benchmark", ["single_choice", "hierarchical", "parallel_noul"]),
        ("sufficiency", ["noul", "choice2", "score5"]),
    ):
        for strategy in strategies:
            qs = questions(task, strategy)
            answers = {name: {k: 1 / len(keys(q)) for k in keys(q)} for name, q in qs.items()}
            a = aggregate(task, strategy, answers)
            assert abs(sum(a["probabilities"].values()) - 1) < 1e-10
    parallel = aggregate(
        "benchmark", "parallel_noul", {k: {"false": 0.2, "true": 0.8} for k in ["Yes", "No", "Inquire"]}
    )
    assert parallel["prediction"] == "Inquire" and parallel["composition"]["conflict"]
    no_positive = aggregate(
        "benchmark", "parallel_noul", {k: {"false": 0.8, "true": 0.2} for k in ["Yes", "No", "Inquire"]}
    )
    assert no_positive["composition"]["no_positive"]
    hierarchy = aggregate(
        "benchmark", "hierarchical", {"sufficient": {"false": 0.8, "true": 0.2}, "terminal": {"Yes": 0.9, "No": 0.1}}
    )
    assert hierarchy["prediction"] == "Inquire" and abs(hierarchy["probabilities"]["Yes"] - 0.18) < 1e-10
    train = training_example(example, "Inquire", "benchmark", "hierarchical")
    assert "terminal" not in train["gold"] and "terminal" not in train["questions"]
    try:
        training_example(example, "Yes", "sufficiency", "score5")
        raise AssertionError("Score training must fail")
    except ValueError:
        pass
    fixtures = [
        {
            "example_id": "a",
            "gold": "sufficient",
            "prediction": "sufficient",
            "probabilities": {"sufficient": 0.8, "insufficient": 0.2},
        },
        {
            "example_id": "b",
            "gold": "sufficient",
            "prediction": "insufficient",
            "probabilities": {"sufficient": 0.4, "insufficient": 0.6},
        },
    ]
    scored = metrics(fixtures, ["insufficient", "sufficient"])
    assert scored["accuracy"] == 0.5 and abs(scored["brier"] - 0.4) < 1e-10 and abs(scored["ece"] - 0.4) < 1e-10
    assert abs(scored["nll"] - (-math.log(0.8) - math.log(0.4)) / 2) < 1e-10
    fitted = fit_temperature(fixtures)
    nll = sum(-math.log(temperature(r["probabilities"], fitted)[r["gold"]]) for r in fixtures) / 2
    assert nll <= scored["nll"] + 1e-10
    result = {
        "information_allowlist": "passed",
        "six_task_strategies": "passed",
        "composition_conflicts": "passed",
        "no_invented_score_labels": "passed",
        "metric_arithmetic": "passed",
        "temperature_nll": "passed",
    }
    if prepared:
        validate_prepared(prepared)

        # Fixed fake probabilities exercise the real persistence path on eight
        # train inputs. No weights, GPU, or evaluation partition is used.
        class FixtureBackend:
            provenance = {"fixture_only": True}

            def predict(self, text, qs):
                return {
                    "answers": {"decision": {"Yes": 0.6, "No": 0.3, "Inquire": 0.1}},
                    "native": {"fixture_only": True},
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                }

        with tempfile.TemporaryDirectory(prefix="orsharc-fixture-") as tmp:
            root = Path(tmp)
            write_rows(root / "option_order.jsonl", [training_example(example, "Yes", "benchmark", "single_choice")])
            assert list(rows(root / "option_order.jsonl")[0]["questions"]["decision"]["criteria"]) == [
                "Yes",
                "No",
                "Inquire",
            ]
            write_json(root / "checkpoint/config.json", {"fixture_only": True})
            config = {
                "backend": "qwen",
                "checkpoint": str(root / "checkpoint"),
                "model_revision": "fixture",
                "quantization": "bf16",
                "fixture_only": True,
            }
            original_loader, original_open = backends.load, Path.open

            def guarded_open(path, *args, **kwargs):
                if "sealed" in path.parts:
                    raise AssertionError("inference opened sealed gold")
                return original_open(path, *args, **kwargs)

            try:
                backends.load = lambda *args: FixtureBackend()
                Path.open = guarded_open
                assert infer(config, prepared, root / "run", "train_256", limit=8)["n"] == 8
                assert infer(config, prepared, root / "run", "train_256", limit=8)["status"] == "skipped"
                (root / "run/complete.json").unlink()
                pred = root / "run/predictions.jsonl"
                pred.write_bytes(pred.read_bytes()[:-30])
                assert infer(config, prepared, root / "run", "train_256", limit=8)["n"] == 8
                try:
                    infer({**config, "model_revision": "changed"}, prepared, root / "run", "train_256", limit=8)
                    raise AssertionError("changed run identity must fail")
                except ValueError:
                    pass
            finally:
                backends.load, Path.open = original_loader, original_open
            report(root / "run", prepared, root / "report")
            assert read_json(root / "report/metrics.json")["metrics"]["raw"]["All"]["n"] == 8
            assert "Minimum confidence" in (root / "report/dashboard.html").read_text()
            assert len(list((root / "report").glob("*.svg"))) == 3
        result["saved_result_report_dashboard"] = "passed; eight deterministic train fixtures; not model results"
        result["resume_skip_partial_write_identity"] = "passed"
        result["inference_does_not_open_gold"] = "passed; file-open guard"
        result["native_option_order_roundtrip"] = "passed"
    return result
