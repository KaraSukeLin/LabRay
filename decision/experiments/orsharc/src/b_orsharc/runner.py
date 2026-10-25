"""Validated, incremental inference; inference never opens evaluation gold."""

import json
import random
import sys
import time
from pathlib import Path

from . import backends
from .common import (
    append,
    checkpoint_hash,
    code_hash,
    digest,
    environment,
    owner,
    read_json,
    rows,
    signature,
    utc,
    write_json,
)
from .data import load_gold, load_inputs, validate_prepared
from .tasks import aggregate, keys, questions, state, training_example

ROOT = Path(__file__).parents[2]


def model_config(name, checkpoint=None, quantization=None):
    config = read_json(ROOT / "configs/models.json")[name].copy()
    if checkpoint:
        config["checkpoint"] = str(Path(checkpoint).resolve())
        config["model_revision"] = "local:" + str(Path(checkpoint).resolve())
    if quantization:
        config["quantization"] = quantization
    return config


def validate_config(config):
    for k in ("backend", "checkpoint", "model_revision", "quantization"):
        if not config.get(k):
            raise ValueError(f"missing model field: {k}")
    if config["backend"] not in ("qev", "kev", "startlux", "qwen"):
        raise ValueError("unrecognized backend")
    supported = {"qev": ["bf16"], "kev": ["bf16"], "startlux": ["bf16", "fp8"], "qwen": ["bf16", "bnb8", "bnb4"]}
    if config["quantization"] not in supported[config["backend"]]:
        raise ValueError(f"unsupported quantization path: {config['backend']}/{config['quantization']}")
    if config["backend"] == "startlux" and config["quantization"] == "fp8" and config.get("engine") != "vllm":
        raise ValueError("StartLux FP8 requires engine=vllm, with a matched BF16 vLLM reference")
    return config


def runtime_sources(config):
    if config["backend"] == "qwen":
        return None
    name = {"qev": "onejev", "kev": "kev", "startlux": "startlux"}[config["backend"]]
    path = (ROOT / "artifacts/upstream" / name).resolve()
    receipt = read_json(path.parent / f"{name}.receipt.json")
    actual = {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*.py")) if ".git" not in p.parts}
    if signature(actual) != receipt["python_sha256"]:
        raise ValueError(f"upstream runtime changed: {path}; bootstrap and gate the new source")
    sys.path.insert(0, str(path))
    return receipt


def recover_tail(path):
    """Only discard an interrupted final write; any complete corrupt row fails."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        last = data.rfind(b"\n") + 1
        # A complete JSON record without a newline is recoverable by adding it.
        try:
            json.loads(data[last:])
            path.write_bytes(data + b"\n")
        except json.JSONDecodeError:
            path.write_bytes(data[:last])


def infer(
    config, prepared, output, partition="dev_select", task="benchmark", strategy="single_choice", shots=0, limit=0
):
    validate_config(config)
    if shots not in (0, 4) or shots and config["backend"] != "qwen":
        raise ValueError("four-shot is a Qwen baseline; other paths use zero demonstrations")
    if partition not in ("train_256", "dev_select", "dev_calibrate", "test"):
        raise ValueError("inference uses train_256 for smoke, or a frozen evaluation partition")
    if partition == "test" and limit:
        raise ValueError("test runs must cover the complete official split")
    qs = questions(task, strategy)
    m = validate_prepared(prepared, verify_gold=False)
    examples = load_inputs(prepared, partition)
    if limit:
        examples = examples[:limit]
    demos = []
    if shots:
        if strategy != "single_choice" or task != "benchmark":
            raise ValueError("the registered four-shot baseline uses the main three-way benchmark")
        gold = load_gold(prepared, "train")
        demos = [
            training_example(r, gold[r["example_id"]]["gold"], task, strategy) for r in load_inputs(prepared, "demos_4")
        ]
    sources = runtime_sources(config)
    checkpoint = checkpoint_hash(config["checkpoint"])
    config = {**config, "checkpoint_sha256": checkpoint["sha256"]}
    if config["backend"] == "startlux" and config.get("engine") == "vllm":
        config = {**config, "runtime_log_dir": str(Path(output).resolve() / "runtime_logs")}
    spec = {
        "model": config,
        "checkpoint": checkpoint,
        "runtime_source": sources,
        "data_manifest": signature(m),
        "code_sha256": code_hash(),
        "environment": environment(),
        "partition": partition,
        "task": task,
        "strategy": strategy,
        "shots": shots,
        "demo_ids": [r["example_id"] for r in demos],
        "ids": [r["example_id"] for r in examples],
        "seed": 42,
        "limit": limit,
        "questions": qs,
        "option_order": {k: keys(v) for k, v in qs.items()},
        "latency_definition": "synchronized request wall time; excludes load and one warm-up; no cross-sample cache",
        "probability_source": "raw T=1; upstream served calibration disabled",
        "generalization_metadata": {
            "snippet_seen": "rule present in complete official training split",
            "ft_exposed": "per-budget rule membership in the nested 256/1024/4096 training subsets; full one-epoch checkpoints only, not partial smoke checkpoints",
        },
    }
    identity = signature(spec)
    output = Path(output)
    with owner(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path)["identity"] != identity:
                raise ValueError("run identity changed; use a new output directory")
        else:
            write_json(manifest_path, {"identity": identity, "created": utc(), "spec": spec})
        pred_path = output / "predictions.jsonl"
        recover_tail(pred_path)
        saved = rows(pred_path) if pred_path.exists() else []
        done = [r["example_id"] for r in saved]
        expected = spec["ids"]
        if done != expected[: len(done)] or any(r["run_identity"] != identity for r in saved):
            raise ValueError("existing predictions do not match this ordered run")
        if (output / "complete.json").exists():
            complete = read_json(output / "complete.json")
            if len(done) != len(examples) or complete["predictions_sha256"] != digest(pred_path):
                raise ValueError("completed run is corrupt")
            return {"status": "skipped", "n": len(done)}
        random.seed(42)
        start = time.perf_counter()
        backend = backends.load(config, demos)
        append(
            output / "events.jsonl",
            {"event": "loaded", "utc": utc(), "seconds": time.perf_counter() - start, "provenance": backend.provenance},
        )
        # Warm up with a fixed label-free train input, never an evaluation answer.
        warm = load_inputs(prepared, "train_256")[0]
        warm_started = time.perf_counter()
        backend.predict(state(warm), qs)
        append(
            output / "events.jsonl",
            {
                "event": "warmup",
                "example_id": warm["example_id"],
                "seconds": time.perf_counter() - warm_started,
                "utc": utc(),
            },
        )
        for row in examples[len(done) :]:
            rendered = state(row)
            begin = time.perf_counter()
            calls, answers, native, usage = [], {}, {}, []
            groups = [{k: v} for k, v in qs.items()] if strategy == "hierarchical" else [qs]
            for group in groups:
                t0 = time.perf_counter()
                response = backend.predict(rendered, group)
                calls.append(
                    {
                        "question_ids": list(group),
                        "latency_seconds": time.perf_counter() - t0,
                        "runtime_stats": response.get("stats"),
                        "native_timing_ms": response.get("timing_ms"),
                    }
                )
                for name, q in group.items():
                    answers[name] = backends.normalized(response["answers"][name], keys(q))
                native.update(response["native"])
                usage.append(response.get("usage"))
            latency = time.perf_counter() - begin
            result = aggregate(task, strategy, answers)
            record = {
                **{
                    k: row[k]
                    for k in (
                        "example_id",
                        "official_split",
                        "rule_id",
                        "tree_id",
                        "snippet_seen",
                        "rule_split",
                        "ft_exposed",
                    )
                },
                "run_identity": identity,
                "task": task,
                "strategy": strategy,
                "format": {k: v["type"] for k, v in qs.items()},
                "option_order": {k: keys(v) for k, v in qs.items()},
                "questions": qs,
                "input_sha256": signature(
                    {"state": rendered, "questions": qs, "option_order": {k: keys(v) for k, v in qs.items()}}
                ),
                "answers": answers,
                "native": native,
                "usage": usage,
                "calls": calls,
                "request_count": len(calls),
                "question_count": len(qs),
                "latency_seconds": latency,
                **result,
            }
            append(pred_path, record)
        write_json(
            output / "complete.json",
            {"identity": identity, "n": len(examples), "utc": utc(), "predictions_sha256": digest(pred_path)},
        )
        if hasattr(backend, "close"):
            backend.close()
    return {"status": "complete", "n": len(examples)}
