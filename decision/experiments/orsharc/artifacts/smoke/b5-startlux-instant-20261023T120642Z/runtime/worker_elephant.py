#!/usr/bin/env python3
"""Native Original Elephant BF16 control on the frozen StartLux smoke sample."""
import datetime
import json
import math
import pathlib
import statistics
import sys
import time
from collections import Counter

import torch

HERE = pathlib.Path(__file__).resolve().parent
ATTEMPT = HERE.parent
REPO = ATTEMPT.parents[4]
ORSHARC = REPO / "experiments" / "orsharc"
sys.path.insert(0, str(ORSHARC / "src"))
sys.path.insert(0, str(ORSHARC / "artifacts" / "upstream" / "onejev"))
from b_orsharc.backends import QevBackend
from b_orsharc.tasks import questions as task_questions


def percentile(values, p):
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


samples = [json.loads(line) for line in (ATTEMPT / "samples.jsonl").read_text().splitlines() if line.strip()]
assert len(samples) == 24 and len({x["sample_id"] for x in samples}) == 24
expected_question = task_questions("benchmark", "single_choice")["decision"]
assert all(x["question"] == expected_question for x in samples)

config_path = ORSHARC / "configs" / "models.json"
config_all = json.loads(config_path.read_text())
config = config_all["elephant"]
assert config["backend"] == "qev"
assert config["checkpoint"] == "/root/model/orsharc-elephant"
assert config["quantization"] == "bf16"
assert config["model_revision"] == "b3f8a93462fb1051a9de031e94a7ff380e254072"
assert torch.cuda.is_available() and torch.cuda.device_count() == 1

device = torch.device("cuda:0")
torch.cuda.synchronize(device)
torch.cuda.reset_peak_memory_stats(device)
load_start = time.perf_counter()
backend = QevBackend(config, demos=[])
torch.cuda.synchronize(device)
load_seconds = time.perf_counter() - load_start
load_memory = {
    "allocated_bytes": torch.cuda.memory_allocated(device),
    "reserved_bytes": torch.cuda.memory_reserved(device),
}

def request(sample):
    return backend.predict(sample["state"], {"decision": sample["question"]})

warmup_start = time.perf_counter()
warmup = request(samples[0])
warmup_seconds = time.perf_counter() - warmup_start
(HERE / "elephant-warmup.json").write_text(json.dumps({
    "sample_id": samples[0]["sample_id"],
    "wall_seconds": warmup_seconds,
    "answers": warmup["answers"],
    "native": warmup["native"],
    "usage": warmup["usage"],
}, indent=2, ensure_ascii=False) + "\n")

rows = []
latencies = []
for i, sample in enumerate(samples, 1):
    started = time.perf_counter()
    response = request(sample)
    elapsed = time.perf_counter() - started
    probs = {k: float(v) for k, v in response["answers"]["decision"].items()}
    expected = list(sample["question"]["criteria"])
    if set(probs) != set(expected):
        raise RuntimeError(f"probability keys changed for {sample['sample_id']}: {list(probs)}")
    if any(not math.isfinite(v) or v < 0 or v > 1 for v in probs.values()) or abs(sum(probs.values()) - 1.0) > 1e-5:
        raise RuntimeError(f"invalid Elephant probability distribution for {sample['sample_id']}: {probs}")
    prediction = max(probs, key=probs.__getitem__)
    rows.append({
        "sample_id": sample["sample_id"],
        "rule_split": sample["rule_split"],
        "snippet_seen": sample["snippet_seen"],
        "length_quantile": sample["length_quantile"],
        "word_count": sample["word_count"],
        "gold": sample["gold"],
        "prediction": prediction,
        "probabilities": probs,
        "native_answer": response["native"]["decision"],
        "usage": response["usage"],
        "client_wall_seconds": elapsed,
    })
    latencies.append(elapsed)
    if i % 4 == 0:
        print(json.dumps({"completed": i, "last_seconds": round(elapsed, 4), "prediction": prediction}), flush=True)

(HERE / "elephant-results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
torch.cuda.synchronize(device)
summary = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "backend": "QevBackend",
    "model_repository": config["model_repository"],
    "model_revision": config["model_revision"],
    "checkpoint": config["checkpoint"],
    "quantization": config["quantization"],
    "temperature": 1.0,
    "identity_calibration": True,
    "device": config["device"],
    "device_name": torch.cuda.get_device_name(0),
    "visible_cuda_devices": torch.cuda.device_count(),
    "max_length": config["max_length"],
    "load_seconds": load_seconds,
    "warmup_seconds": warmup_seconds,
    "load_memory_bytes": load_memory,
    "torch_peak_memory_bytes": {
        "allocated": torch.cuda.max_memory_allocated(device),
        "reserved": torch.cuda.max_memory_reserved(device),
    },
    "n_unique_cases": len(rows),
    "successful_cases": len(rows),
    "correct": sum(row["prediction"] == row["gold"] for row in rows),
    "accuracy": sum(row["prediction"] == row["gold"] for row in rows) / len(rows),
    "prediction_counts": dict(Counter(row["prediction"] for row in rows)),
    "gold_counts": dict(Counter(row["gold"] for row in rows)),
    "client_latency_seconds": {
        "p50": percentile(latencies, 0.50),
        "p95": percentile(latencies, 0.95),
        "mean": statistics.mean(latencies),
        "sum": sum(latencies),
        "sequential_requests_per_second": len(latencies) / sum(latencies),
    },
    "model_storage_bytes": sum(p.stat().st_size for p in pathlib.Path(config["checkpoint"]).rglob("*") if p.is_file()),
    "runtime_provenance": backend.provenance,
}
(HERE / "elephant-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
