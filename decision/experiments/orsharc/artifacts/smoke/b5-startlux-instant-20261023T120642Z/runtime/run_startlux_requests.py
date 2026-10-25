#!/usr/bin/env python3
"""Run the frozen 24-case StartLux smoke and one typed-readout probe."""
import datetime
import json
import math
import pathlib
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime"
BASE = "http://127.0.0.1:8090/v1/systemone"
SUFFICIENCY = "Is the available information sufficient to answer the user's initial question Yes or No without a follow-up question?"
LEVELS = [
    "Definitely insufficient",
    "Probably insufficient",
    "Uncertain whether sufficient",
    "Probably sufficient",
    "Definitely sufficient",
]


def percentile(values, p):
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def post(body, request_id):
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE,
        data=payload,
        headers={"Content-Type": "application/json", "X-Request-ID": request_id},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - started, exc.code, {"error": exc.read().decode("utf-8", "replace")}
    return time.perf_counter() - started, status, json.loads(raw)


samples = [json.loads(line) for line in (ROOT / "samples.jsonl").read_text().splitlines() if line.strip()]
assert len(samples) == 24, f"expected 24 fixed samples, got {len(samples)}"
assert len({x["sample_id"] for x in samples}) == 24
assert Counter((x["gold"], x["rule_split"]) for x in samples) == Counter(
    {(label, split): 4 for label in ("Yes", "No", "Inquire") for split in ("Seen", "Unseen")}
)

results = []
latencies = []
for i, sample in enumerate(samples, 1):
    body = {
        "state": sample["state"],
        "questions": {"decision": sample["question"]},
        "allow_none": False,
        "none_check": False,
        "think_mode": "never",
        "return_details": True,
    }
    elapsed, status, response = post(body, f"smoke-{i:02d}-{sample['sample_id'][:12]}")
    record = {
        "sample_id": sample["sample_id"],
        "rule_split": sample["rule_split"],
        "snippet_seen": sample["snippet_seen"],
        "length_quantile": sample["length_quantile"],
        "word_count": sample["word_count"],
        "gold": sample["gold"],
        "http_status": status,
        "client_wall_seconds": elapsed,
        "response": response,
    }
    if status != 200:
        record["error"] = response
        results.append(record)
        (RUNTIME / "startlux-results.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results))
        raise SystemExit(f"request {i} failed with HTTP {status}: {response}")
    answer = response["answers"]["decision"]
    probs = {k: float(v) for k, v in answer["probabilities"].items()}
    expected = list(sample["question"]["criteria"])
    if set(probs) != set(expected):
        raise SystemExit(f"probability keys changed for {sample['sample_id']}: {list(probs)}")
    if any(not math.isfinite(v) or v < 0 or v > 1 for v in probs.values()) or abs(sum(probs.values()) - 1.0) > 1e-5:
        raise SystemExit(f"invalid native probability distribution for {sample['sample_id']}: {probs}")
    record.update({
        "prediction": answer["choice"],
        "probabilities": probs,
        "confidence": answer.get("confidence"),
        "confidence_semantics": "normalized margin above an even split",
        "confidence_expected_from_probabilities": max(0.0, min(1.0, (len(probs) * max(probs.values()) - 1.0) / (len(probs) - 1.0))),
        "confidence_matches_package_formula": abs(
            float(answer.get("confidence", float("nan")))
            - max(0.0, min(1.0, (len(probs) * max(probs.values()) - 1.0) / (len(probs) - 1.0)))
        ) <= 1e-6,
        "input_tokens": response.get("usage", {}).get("input_tokens"),
        "output_tokens": response.get("usage", {}).get("output_tokens"),
        "server_wall_ms": response.get("timing_ms", {}).get("wall"),
    })
    results.append(record)
    latencies.append(elapsed)
    if i % 4 == 0:
        print(json.dumps({"completed": i, "last_seconds": round(elapsed, 4), "prediction": answer["choice"]}), flush=True)

(RUNTIME / "startlux-results.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results))

# Probe all three native readout types on one already-selected state. It adds no sample.
probe = {
    "state": samples[0]["state"],
    "questions": {
        "choice_probe": samples[0]["question"],
        "noul_probe": {"type": "noul", "instructions": SUFFICIENCY},
        "score_probe": {"type": "score", "instructions": SUFFICIENCY, "criteria": LEVELS},
    },
    "allow_none": False,
    "none_check": False,
    "think_mode": "never",
    "return_details": True,
}
probe_elapsed, probe_status, probe_response = post(probe, "smoke-type-probe-01")
probe_record = {
    "sample_id": samples[0]["sample_id"],
    "http_status": probe_status,
    "client_wall_seconds": probe_elapsed,
    "response": probe_response,
}
(RUNTIME / "startlux-type-probe.json").write_text(json.dumps(probe_record, indent=2, ensure_ascii=False) + "\n")
if probe_status != 200:
    raise SystemExit(f"typed-readout probe failed with HTTP {probe_status}: {probe_response}")

summary = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n_unique_cases": len(results),
    "successful_cases": sum(x["http_status"] == 200 for x in results),
    "correct": sum(x.get("prediction") == x["gold"] for x in results),
    "accuracy": sum(x.get("prediction") == x["gold"] for x in results) / len(results),
    "prediction_counts": dict(Counter(x["prediction"] for x in results)),
    "gold_counts": dict(Counter(x["gold"] for x in results)),
    "client_latency_seconds": {
        "p50": percentile(latencies, 0.50),
        "p95": percentile(latencies, 0.95),
        "mean": statistics.mean(latencies),
        "sum": sum(latencies),
        "sequential_requests_per_second": len(latencies) / sum(latencies),
    },
    "confidence_semantics": "Choice confidence is normalized margin above an even split, not the selected option probability.",
    "confidence_formula_match_count": sum(x["confidence_matches_package_formula"] for x in results),
    "type_probe_status": probe_status,
    "type_probe_seconds": probe_elapsed,
    "type_probe_answers": probe_response.get("answers"),
}
(RUNTIME / "startlux-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, ensure_ascii=False))
