"""Offline selection, calibration, metrics and a dependency-free HTML dashboard."""

import html
import math
import statistics
from pathlib import Path

from .common import digest, read_json, rows, signature, write_json, write_rows
from .data import LABELS, load_gold, validate_prepared
from .tasks import aggregate, target


def temperature(probabilities, value):
    logits = {k: math.log(max(p, 1e-15)) / value for k, p in probabilities.items()}
    peak = max(logits.values())
    exps = {k: math.exp(v - peak) for k, v in logits.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def fit_temperature(examples):
    def loss(log_t):
        t = math.exp(log_t)
        return sum(-math.log(max(temperature(r["probabilities"], t)[r["gold"]], 1e-15)) for r in examples) / len(
            examples
        )

    a, b = math.log(0.05), math.log(20)
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(70):
        c, d = b - ratio * (b - a), a + ratio * (b - a)
        if loss(c) <= loss(d):
            b = d
        else:
            a = c
    candidates = [0.05, 1.0, 20.0, math.exp((a + b) / 2)]
    return min(candidates, key=lambda t: (loss(math.log(t)), abs(math.log(t))))


def metrics(examples, labels):
    if not examples:
        return {"n": 0}
    n = len(examples)
    confusion = {g: {p: 0 for p in labels} for g in labels}
    brier = nll = 0.0
    bins = [{"lower": i / 10, "upper": (i + 1) / 10, "n": 0, "correct": 0, "confidence_sum": 0.0} for i in range(10)]
    for row in examples:
        p, y, pred = row["probabilities"], row["gold"], row["prediction"]
        confusion[y][pred] += 1
        brier += sum((p[k] - int(k == y)) ** 2 for k in labels)
        nll -= math.log(max(p[y], 1e-15))
        confidence = p[pred]
        b = bins[min(9, int(confidence * 10))]
        b["n"] += 1
        b["correct"] += int(pred == y)
        b["confidence_sum"] += confidence
    recalls, f1s = [], []
    for label in labels:
        tp = confusion[label][label]
        actual = sum(confusion[label].values())
        predicted = sum(confusion[g][label] for g in labels)
        recalls.append(tp / actual if actual else 0.0)
        f1s.append(2 * tp / (actual + predicted) if actual + predicted else 0.0)
    ece = 0.0
    for b in bins:
        b["accuracy"] = b["correct"] / b["n"] if b["n"] else None
        b["confidence"] = b["confidence_sum"] / b["n"] if b["n"] else None
        if b["n"]:
            ece += b["n"] / n * abs(b["accuracy"] - b["confidence"])
    # Group equal confidences so a coverage point never splits a tie arbitrarily.
    ordered = sorted(examples, key=lambda r: (-r["probabilities"][r["prediction"]], r["example_id"]))
    curve, errors, i = [], 0, 0
    while i < n:
        confidence = ordered[i]["probabilities"][ordered[i]["prediction"]]
        j = i
        while j < n and ordered[j]["probabilities"][ordered[j]["prediction"]] == confidence:
            errors += ordered[j]["prediction"] != ordered[j]["gold"]
            j += 1
        curve.append({"threshold": confidence, "coverage": j / n, "risk": errors / j, "n": j})
        i = j
    aurc, previous = 0.0, 0.0
    for point in curve:
        aurc += (point["coverage"] - previous) * point["risk"]
        previous = point["coverage"]
    result = {
        "n": n,
        "accuracy": sum(confusion[k][k] for k in labels) / n,
        "macro_f1": statistics.mean(f1s),
        "macro_accuracy": statistics.mean(recalls),
        "ece": ece,
        "brier": brier / n,
        "nll": nll / n,
        "confusion": confusion,
        "reliability": bins,
        "risk_coverage": curve,
        "aurc": aurc,
        "brier_definition": "sum over classes, averaged over samples",
        "ece_definition": "10 equal-width bins; confidence of selected policy label",
    }
    latencies = [r["latency_seconds"] for r in examples if "latency_seconds" in r]
    if latencies:
        result["cost"] = {
            "total_seconds": sum(latencies),
            "mean_seconds": statistics.mean(latencies),
            "median_seconds": statistics.median(latencies),
            "p95_seconds": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1],
            "request_count": sum(r["request_count"] for r in examples),
            "question_count": sum(r["question_count"] for r in examples),
            "input_tokens": sum((u or {}).get("input_tokens", 0) for r in examples for u in r.get("usage", [])),
            "token_measurement_available": all(
                u is not None and "input_tokens" in u for r in examples for u in r.get("usage", [])
            ),
        }
    return result


def completed(run):
    run = Path(run)
    m, c = read_json(run / "manifest.json"), read_json(run / "complete.json")
    if c["identity"] != m["identity"] or digest(run / "predictions.jsonl") != c["predictions_sha256"]:
        raise ValueError("inference completion hash mismatch")
    predictions = rows(run / "predictions.jsonl")
    if [r["example_id"] for r in predictions] != m["spec"]["ids"] or len(predictions) != c["n"]:
        raise ValueError("predictions do not cover the declared sample IDs")
    if any(r["run_identity"] != m["identity"] for r in predictions):
        raise ValueError("prediction run identity mismatch")
    return m, predictions


def condition(spec):
    value = {
        k: spec[k]
        for k in (
            "model",
            "checkpoint",
            "data_manifest",
            "code_sha256",
            "task",
            "strategy",
            "shots",
            "demo_ids",
            "questions",
            "option_order",
        )
    }
    value["model"] = {k: v for k, v in value["model"].items() if k != "runtime_log_dir"}
    return signature(value)


def validate_membership(prepared, spec):
    from .data import load_inputs

    expected = [r["example_id"] for r in load_inputs(prepared, spec["partition"])]
    if spec["limit"]:
        if spec["partition"] == "test":
            raise ValueError("a partial test run is not a valid benchmark")
        expected = expected[: spec["limit"]]
    if spec["ids"] != expected:
        raise ValueError("run membership differs from the prepared partition")


def join_gold(predictions, prepared, spec, threshold=0.5):
    split = (
        "dev" if spec["partition"].startswith("dev") else "train" if spec["partition"].startswith("train") else "test"
    )
    gold = load_gold(prepared, split)
    result = []
    for row in predictions:
        result.append(
            {
                **row,
                **aggregate(spec["task"], spec["strategy"], row["answers"], threshold),
                "gold": target(gold[row["example_id"]]["gold"], spec["task"]),
                "benchmark_gold": gold[row["example_id"]]["gold"],
            }
        )
    return result


def calibrate(select_run, calibration_run, prepared, output):
    dm = validate_prepared(prepared)
    sm, sp = completed(select_run)
    cm, cp = completed(calibration_run)
    validate_membership(prepared, sm["spec"])
    validate_membership(prepared, cm["spec"])
    if sm["spec"]["partition"] != "dev_select" or cm["spec"]["partition"] != "dev_calibrate":
        raise ValueError("threshold selection requires dev_select, temperature fitting requires dev_calibrate")
    if sm["spec"]["limit"] or cm["spec"]["limit"]:
        raise ValueError("calibration must cover both complete development partitions")
    if condition(sm["spec"]) != condition(cm["spec"]) or signature(dm) != sm["spec"]["data_manifest"]:
        raise ValueError("selection/calibration condition or data differs")
    threshold, sweep = 0.5, []
    if sm["spec"]["task"] == "sufficiency":
        for value in [i / 100 for i in range(1, 100)]:
            scored = join_gold(sp, prepared, sm["spec"], value)
            accuracy = sum(r["prediction"] == r["gold"] for r in scored) / len(scored)
            sweep.append({"threshold": value, "accuracy": accuracy})
        threshold = max(sweep, key=lambda r: (r["accuracy"], -abs(r["threshold"] - 0.5), -r["threshold"]))["threshold"]
    selected = join_gold(cp, prepared, cm["spec"], threshold)
    t = fit_temperature(selected)
    result = {
        "condition": condition(cm["spec"]),
        "temperature": t,
        "threshold": threshold,
        "selection_run": sm["identity"],
        "calibration_run": cm["identity"],
        "data_manifest": signature(dm),
        "selection_predictions_sha256": digest(Path(select_run) / "predictions.jsonl"),
        "calibration_predictions_sha256": digest(Path(calibration_run) / "predictions.jsonl"),
        "threshold_sweep_dev_select": sweep,
        "temperature_fit": "bounded NLL minimization, T in [0.05,20], dev_calibrate only",
        "probability_stage": "after deterministic task projection/aggregation; binary Score uses expected level / 4",
        "prediction_policy": "frozen raw threshold/routing labels retained after temperature scaling",
        "temperature_at_bound": t in (0.05, 20.0),
    }
    write_json(output, result)
    return result


def composition_metrics(examples, strategy):
    if strategy == "parallel_noul":
        return {
            "conflicts": sum(r["composition"]["conflict"] for r in examples),
            "no_positive": sum(r["composition"]["no_positive"] for r in examples),
            "component_errors": {
                k: sum((r["answers"][k]["true"] >= 0.5) != (r["gold"] == k) for r in examples) for k in LABELS
            },
            "fallback_errors": sum(
                r["prediction"] != r["gold"] and (r["composition"]["conflict"] or r["composition"]["no_positive"])
                for r in examples
            ),
        }
    if strategy == "hierarchical":
        gate_errors = sum(r["composition"]["sufficient_prediction"] != (r["gold"] != "Inquire") for r in examples)
        return {
            "gate_errors": gate_errors,
            "terminal_errors_given_correct_gate": sum(
                r["gold"] != "Inquire"
                and r["composition"]["sufficient_prediction"]
                and r["composition"]["terminal_prediction"] != r["gold"]
                for r in examples
            ),
            "final_errors_from_gate": sum(
                r["prediction"] != r["gold"] and r["composition"]["sufficient_prediction"] != (r["gold"] != "Inquire")
                for r in examples
            ),
            "all_final_errors": sum(r["prediction"] != r["gold"] for r in examples),
        }
    return {}


def report(run, prepared, output, calibration=None):
    dm = validate_prepared(prepared)
    manifest, predictions = completed(run)
    spec = manifest["spec"]
    validate_membership(prepared, spec)
    if signature(dm) != spec["data_manifest"]:
        raise ValueError("report data differs from inference data")
    fitted = read_json(calibration) if calibration else None
    if fitted and (fitted["condition"] != condition(spec) or fitted["data_manifest"] != signature(dm)):
        raise ValueError("calibration belongs to another model/task/format/precision/data condition")
    threshold = fitted["threshold"] if fitted else 0.5
    raw = join_gold(predictions, prepared, spec, threshold)
    versions = {"raw": raw}
    if fitted:
        versions["calibrated"] = [
            {
                **r,
                "probabilities": temperature(r["probabilities"], fitted["temperature"]),
                "confidence": temperature(r["probabilities"], fitted["temperature"])[r["prediction"]],
            }
            for r in raw
        ]
    labels = LABELS if spec["task"] == "benchmark" else ["insufficient", "sufficient"]
    result = {
        "run_identity": manifest["identity"],
        "condition": condition(spec),
        "partition": spec["partition"],
        "task": spec["task"],
        "strategy": spec["strategy"],
        "threshold": threshold,
        "labels": labels,
        "calibration_sha256": digest(calibration) if calibration else None,
        "smoke_only": bool(spec["limit"]),
        "composition": composition_metrics(raw, spec["strategy"]),
        "metrics": {},
        "fixture_only": spec["model"].get("fixture_only", False),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for version, examples in versions.items():
        result["metrics"][version] = {
            split: metrics([r for r in examples if split == "All" or r["rule_split"] == split], labels)
            for split in ("All", "Seen", "Unseen")
        }
        write_rows(output / f"scored_{version}.jsonl", examples)
        for split in ("All", "Seen", "Unseen"):
            rel = result["metrics"][version][split].get("reliability", [])
            (output / f"reliability_{version}_{split}.svg").write_text(reliability_svg(rel, f"{version} / {split}"))
    write_json(output / "metrics.json", result)
    write_dashboard(output / "dashboard.html", versions, labels, result)
    return result


def reliability_svg(bins, title):
    bars = []
    for i, b in enumerate(bins):
        if b["n"]:
            y = 220 - 180 * b["accuracy"]
            bars.append(
                f'<rect x="{40 + i * 18}" y="{y}" width="16" height="{220 - y}" fill="#2563eb"><title>n={b["n"]}, accuracy={b["accuracy"]:.3f}, confidence={b["confidence"]:.3f}</title></rect>'
            )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="270" viewBox="0 0 300 270">'
        '<rect width="300" height="270" fill="white"/>'
        f'<text x="20" y="22">{html.escape(title)}</text><path d="M40 40V220H220" fill="none" stroke="black"/>'
        '<path d="M40 220L220 40" stroke="#999" stroke-dasharray="4"/>'
        + "".join(bars)
        + '<text x="70" y="250">Confidence (0–1)</text><text x="2" y="35">Accuracy</text></svg>'
    )


def write_dashboard(path, versions, labels, metadata):
    from .common import canonical

    values = {
        name: [
            {
                "id": r["example_id"],
                "split": r["rule_split"],
                "gold": r["gold"],
                "prediction": r["prediction"],
                "confidence": r["confidence"],
                "latency": r.get("latency_seconds", 0),
            }
            for r in examples
        ]
        for name, examples in versions.items()
    }
    encoded = canonical(values).replace("<", "\\u003c")
    page = """<!doctype html><meta charset="utf-8"><title>OR-ShARC confidence dashboard</title>
<style>body{font:16px system-ui;max-width:1000px;margin:32px auto;padding:12px;background:#f8fafc;color:#172554}select,input{margin:8px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #cbd5e1}svg{background:white;margin:12px 0}#cards{font-size:20px;margin:20px 0}</style>
<h1>OR-ShARC confidence dashboard</h1><p>Exploratory review of saved predictions. Test slider changes do not update frozen selection or calibration parameters. Zero accepted samples have undefined risk.</p>
<label>Predictions <select id="version"></select></label><label>Rule split <select id="split"><option>All</option><option>Seen</option><option>Unseen</option></select></label>
<label>Minimum confidence <input id="threshold" type="range" min="0" max="1" step="0.01" value="0.5"><output id="value"></output></label>
<div id="cards"></div><svg id="curve" viewBox="0 0 600 260" width="600" aria-label="Risk coverage curve"></svg>
<table><thead><tr><th>Gold</th><th>Prediction</th><th>Count accepted</th></tr></thead><tbody id="confusion"></tbody></table>
<script>const data=DATA, labels=LABELS;
for(const k of Object.keys(data)){const o=document.createElement('option');o.textContent=k;document.querySelector('#version').append(o)}
function update(){const mode=document.querySelector('#version').value,s=document.querySelector('#split').value,t=+document.querySelector('#threshold').value;
const all=data[mode].filter(r=>s==='All'||r.split===s), kept=all.filter(r=>r.confidence>=t), correct=kept.filter(r=>r.gold===r.prediction).length;
document.querySelector('#value').textContent=t.toFixed(2);const n=kept.length;
document.querySelector('#cards').textContent=`Accepted ${n}/${all.length} | Coverage ${all.length?(100*n/all.length).toFixed(1):'—'}% | Accuracy ${n?(100*correct/n).toFixed(1)+'%':'undefined'} | Risk ${n?(1-correct/n).toFixed(3):'undefined'}`;
const body=document.querySelector('#confusion');body.replaceChildren();for(const g of labels)for(const p of labels){const tr=document.createElement('tr');for(const v of [g,p,kept.filter(r=>r.gold===g&&r.prediction===p).length]){const td=document.createElement('td');td.textContent=v;tr.append(td)}body.append(tr)}
const sorted=[...all].sort((a,b)=>b.confidence-a.confidence);let errors=0,points=[];for(let i=0;i<sorted.length;i++){errors+=sorted[i].gold!==sorted[i].prediction;if(i===sorted.length-1||sorted[i].confidence!==sorted[i+1].confidence)points.push(`${50+500*(i+1)/sorted.length},${220-180*errors/(i+1)}`)}
document.querySelector('#curve').innerHTML='<path d="M50 40V220H550" fill="none" stroke="black"/><text x="240" y="250">Coverage (0–1)</text><text x="5" y="25">Risk (0–1)</text><polyline fill="none" stroke="#2563eb" stroke-width="2" points="'+points.join(' ')+'"/>';
}for(const id of ['version','split','threshold'])document.querySelector('#'+id).addEventListener('input',update);update();</script>"""
    page = page.replace("DATA", encoded).replace("LABELS", canonical(labels))
    if metadata.get("fixture_only"):
        page = page.replace(
            "<h1>OR-ShARC confidence dashboard</h1>",
            "<h1>Fixture only: confidence dashboard</h1><p>Artificial probabilities for a CPU interface check. These are not model or benchmark results.</p>",
        )
    Path(path).write_text(page)


def compare_quantization(runs, output):
    loaded = [completed(run) for run in runs]
    if len(loaded) < 2:
        raise ValueError("need BF16 and at least one quantized run")
    if len({m["spec"]["model"]["quantization"] for m, _ in loaded}) < 2:
        raise ValueError("provide at least two distinct precision conditions")
    reference = next((v for v in loaded if v[0]["spec"]["model"]["quantization"] == "bf16"), None)
    if reference is None:
        raise ValueError("BF16 reference is required")
    rm, rp = reference
    comparisons = []
    for m, p in loaded:
        s, r = m["spec"], rm["spec"]
        for key in (
            "data_manifest",
            "ids",
            "task",
            "strategy",
            "shots",
            "demo_ids",
            "questions",
            "partition",
            "code_sha256",
            "environment",
            "runtime_source",
        ):
            if s[key] != r[key]:
                raise ValueError(f"quantization comparison changed {key}")
        if s["checkpoint"]["sha256"] != r["checkpoint"]["sha256"]:
            raise ValueError("quantization comparison must load identical source checkpoint bytes")
        for key in ("backend", "engine"):
            if s["model"].get(key) != r["model"].get(key):
                raise ValueError(f"quantization comparison changed runtime {key}")
        if any(a["input_sha256"] != b["input_sha256"] or a["option_order"] != b["option_order"] for a, b in zip(p, rp)):
            raise ValueError("quantization input or option-order mismatch")
        differences = [
            abs(a["probabilities"][k] - b["probabilities"][k]) for a, b in zip(p, rp) for k in a["probabilities"]
        ]
        comparisons.append(
            {
                "run": m["identity"],
                "quantization": s["model"]["quantization"],
                "n": len(p),
                "prediction_agreement": sum(a["prediction"] == b["prediction"] for a, b in zip(p, rp)) / len(p),
                "max_probability_difference": max(differences),
                "mean_latency_seconds": statistics.mean(a["latency_seconds"] for a in p),
            }
        )
    result = {
        "reference": rm["identity"],
        "checkpoint_sha256": rm["spec"]["checkpoint"]["sha256"],
        "comparisons": comparisons,
    }
    write_json(output, result)
    return result
