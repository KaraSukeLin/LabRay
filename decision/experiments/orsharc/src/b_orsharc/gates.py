"""Offline checks of Phase 2 smoke artifacts. Does not claim unmeasured parity."""

import math
from pathlib import Path

from .analysis import completed
from .common import checkpoint_hash, code_hash, digest, read_json, rows, signature, write_json


def environment_verify(output):
    """Check the tested override explicitly; generic pip check will flag Torch's pin."""
    import importlib.metadata as metadata
    import platform
    import re
    import torch
    import triton
    from .runner import ROOT, model_config, runtime_sources

    lock = ROOT / "elephant-runtime.lock"
    mismatches = {}
    for line in lock.read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            name, expected = line.split("==")
            actual = metadata.version(name)
            if actual != expected:
                mismatches[name] = {"expected": expected, "actual": actual}
    # Check all requirements, allowing only the measured Torch/Triton override.
    from packaging.requirements import Requirement

    conflicts = []
    for dist in metadata.distributions():
        for value in dist.requires or []:
            req = Requirement(value)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            if re.sub(r"[-_.]+", "-", req.name).lower() == "triton" and dist.metadata["Name"] == "torch":
                continue
            try:
                actual = metadata.version(req.name)
            except metadata.PackageNotFoundError:
                conflicts.append(f"{dist.metadata['Name']}: missing {req}")
                continue
            if req.specifier and actual not in req.specifier:
                conflicts.append(f"{dist.metadata['Name']}: {req}, installed {actual}")
    result = {
        "status": "PASS"
        if not mismatches
        and not conflicts
        and platform.python_version() == "3.12.14"
        and torch.version.cuda == "12.8"
        and triton.__version__ == "3.7.1"
        else "FAIL",
        "lock_sha256": digest(lock),
        "mismatches": mismatches,
        "other_dependency_conflicts": conflicts,
        "python": platform.python_version(),
        "torch_runtime": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "runtime_source": runtime_sources(model_config("elephant")),
        "accepted_override": "torch==2.8.0 metadata requires triton==3.4.0; H100 FLA backward requires the tested 3.7.1 override; install exact lock with --no-deps",
    }
    write_json(output, result)
    if result["status"] != "PASS":
        raise ValueError(f"runtime differs from tested lock: {result}")
    return result


def smoke_verify(original, reload, output, training=None, adapted=None, admission_evidence=None):
    om, op = completed(original)
    rm, rp = completed(reload)
    if om["spec"]["partition"] != "train_256" or om["spec"]["limit"] != 8:
        raise ValueError("smoke must use eight train examples")
    if om["identity"] != rm["identity"]:
        raise ValueError("reload smoke differs from original condition")
    difference = max(
        abs(a["probabilities"][k] - b["probabilities"][k]) for a, b in zip(op, rp) for k in a["probabilities"]
    )
    if difference > 0.005 or any(a["prediction"] != b["prediction"] for a, b in zip(op, rp)):
        raise ValueError(f"reload parity failed: max probability difference {difference}")
    result = {
        "inference_reload_parity": "passed",
        "max_probability_difference": difference,
        "tolerance": 0.005,
        "original_run": om["identity"],
        "formal_experiment_admission": False,
        "remaining_gates": [
            "production longest-input memory/throughput profile",
            "trainable/frozen tensor scope audit",
            "save/merge parity on the adapted model",
            "Qwen exact-resume parity if resume will be used",
        ],
    }
    if training:
        tm = read_json(Path(training) / "manifest.json")
        tc = read_json(Path(training) / "complete.json")
        if not tc["smoke_only"] or tc["identity"] != tm["identity"]:
            raise ValueError("training smoke completion mismatch")
        am, ap = completed(adapted)
        if am["spec"]["ids"] != om["spec"]["ids"] or am["spec"]["checkpoint"]["sha256"] != tc["checkpoint"]["sha256"]:
            raise ValueError("adapted inference did not reload the completed smoke checkpoint")
        loss_paths = [Path(training) / "native/log.jsonl", Path(training) / "training.jsonl"]
        losses = [float(r["loss"]) for p in loss_paths if p.exists() for r in rows(p) if "loss" in r]
        if not losses or any(not math.isfinite(v) for v in losses):
            raise ValueError("no finite training-loss evidence")
        if tc["checkpoint"]["sha256"] == tm["spec"]["checkpoint"]["sha256"]:
            raise ValueError("smoke checkpoint bytes did not change")
        result.update(
            {
                "training_artifact_gate": "passed",
                "finite_loss_records": len(losses),
                "training_completion_sha256": digest(Path(training) / "complete.json"),
                "adapted_run": am["identity"],
            }
        )
    if admission_evidence:
        result.update(admission_verify(admission_evidence))
    write_json(output, result)
    return result


def admission_verify(index):
    """Extend the existing smoke gate with measured Phase 3 artifacts."""
    from .runner import ROOT, model_config, runtime_sources

    paths = read_json(index)
    assert paths["verified_code_sha256"] == code_hash(), (
        "code changed after verification; new admission evidence required"
    )
    configurations = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "configs").glob("*.json"))}
    assert paths["verified_configuration_sha256"] == configurations, "configuration changed after verification"
    measurement = Path(paths["measurement"])
    env = read_json(paths["environment"])
    metadata = read_json(paths["metadata"])
    manifest = read_json(measurement / "manifest.json")
    assert runtime_sources(model_config("elephant")) == manifest["runtime"]
    assert read_json(ROOT / "configs/training.json")["onejev"] == manifest["hp"]
    integrity = read_json(measurement / "integrity.json")
    optimizer = read_json(measurement / "optimizer.json")
    profile = read_json(measurement / "profile.json")
    serving = read_json(measurement / "serving.json")
    lengths = read_json(measurement / "lengths.json")
    assert (
        env["status"] == metadata["status"] == integrity["status"] == profile["status"] == serving["status"] == "PASS"
    )
    assert (
        env["lock_sha256"] == digest(ROOT / "elephant-runtime.lock")
        and not env["mismatches"]
        and not env["other_dependency_conflicts"]
    )
    assert env["runtime_source"] == manifest["runtime"] and manifest["data"] == metadata["data_manifest"]
    assert not integrity["frozen_changed"] and integrity["saved_adapter_exact"]
    assert optimizer["exact_membership"] and optimizer["frozen_before"] == integrity["frozen_after"]
    assert (
        set(optimizer["parameters"]) == set(integrity["max_abs_updates"]) == set(integrity["gradient_maxima_per_step"])
    )
    assert all("lora_" in n and v > 0 for n, v in integrity["max_abs_updates"].items())
    assert all(max(v) > 0 and all(math.isfinite(x) for x in v) for v in integrity["gradient_maxima_per_step"].values())
    assert checkpoint_hash(profile["checkpoint"]["path"]) == profile["checkpoint"]
    losses = [r["loss"] for r in rows(measurement / "native/log.jsonl") if "loss" in r]
    assert len(losses) == 3 and all(math.isfinite(v) for v in losses)
    assert profile["dropped_records"] == profile["truncated_records"] == serving["truncated_records"] == 0
    assert serving["over_limit_rejected"] and max(serving["peak_gpu_bytes"].values()) < 70 * 2**30
    assert serving["max_abs_probability_difference"] <= 0.05 and not serving["label_changes"]
    assert min(r["examples_per_s"] for r in profile["training_steps"][1:]) >= 0.5
    assert all(0 < r["seconds"] < 2 for rs in serving["results"].values() for r in rs)
    adapter = read_json(measurement / "native/adapter/adapter_config.json")
    assert adapter["r"] == manifest["hp"]["rank"] and adapter["lora_alpha"] == manifest["hp"]["alpha"]
    assert read_json(measurement / "native/adapter/readout.json") == read_json(
        measurement / "native/merged/readout.json"
    )
    for split in ("train", "dev"):
        assert not lengths["distribution"][split]["over_limit"]
        assert {p["percentile"] for p in lengths["panel"] if p["split"] == split} == {0.5, 0.9, 0.95, 0.99, 1.0}
    active = rows(measurement / "active.jsonl")
    for name, tolerance in (("reloaded", 1e-6), ("merged", 0.02)):
        compared = rows(measurement / f"{name}.jsonl")
        assert len(active) == len(compared) == len(lengths["panel"])
        for a, b in zip(active, compared):
            assert (
                a["example_id"] == b["example_id"] and a["labels"] == b["labels"] and a["prediction"] == b["prediction"]
            )
            for r in (a, b):
                p = r["probabilities"]
                assert (
                    set(p) == set(r["labels"])
                    and all(math.isfinite(v) and 0 <= v <= 1 for v in p.values())
                    and abs(sum(p.values()) - 1) < 1e-5
                )
            assert max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["labels"]) <= tolerance
    cm, predictions = completed(paths["cli_inference"])
    assert cm["spec"]["partition"] in ("train_256", "dev_select", "dev_calibrate")
    assert all(set(r["ft_exposed"]) == {"256", "1024", "4096"} for r in predictions)
    assert read_json(paths["isolation"])["status"] == read_json(paths["plans"])["status"] == "PASS"
    files = [Path(index)] + [Path(paths[k]) for k in ("environment", "metadata", "isolation", "plans", "capabilities")]
    files += [Path(paths["cli_inference"]) / "complete.json"]
    files += [p for p in measurement.glob("*.json*") if p.is_file()]
    files += [
        measurement / "native/log.jsonl",
        measurement / "native/args.json",
        measurement / "native/adapter/adapter_config.json",
        measurement / "native/adapter/adapter_model.safetensors",
    ]
    artifacts = {str(p.resolve()): digest(p) for p in files}
    return {
        "formal_experiment_admission": True,
        "admission_scope": "Elephant BF16 B1–B4 only; full-dev calibration and final evaluation remain Phase 4 work",
        "critical_checks": {
            k: "PASS"
            for k in (
                "environment",
                "parameter_integrity",
                "adapter_reload_merge_parity",
                "long_inputs",
                "labels_and_metadata",
                "commands_and_logging",
            )
        },
        "capabilities": read_json(paths["capabilities"]),
        "remaining_gates": [],
        "admission_code_sha256": code_hash(),
        "configuration_sha256": configurations,
        "evidence": artifacts,
        "evidence_sha256": signature(artifacts),
        "measurement_code_sha256": manifest["code_sha256"],
    }
