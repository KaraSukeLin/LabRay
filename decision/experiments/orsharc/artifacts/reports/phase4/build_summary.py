#!/usr/bin/env python3
"""Build compact Phase 4 summaries and an offline selective-risk dashboard."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
RUNS = ROOT / "experiments/orsharc/artifacts/runs/phase4"
OUT = ROOT / "experiments/orsharc/artifacts/reports/phase4"
TELEMETRY = ROOT / "experiments/orsharc/artifacts/resource/phase4-gpu-telemetry.csv"

CONDITIONS = [
    ("elephant-original-single_choice-bf16-0shot", "Elephant Original · Choice · BF16 · 0-shot"),
    ("elephant-ft_256-single_choice-bf16-0shot", "Elephant FT-256 · Choice · BF16 · 0-shot"),
    ("elephant-ft_1024-single_choice-bf16-0shot", "Elephant FT-1,024 · Choice · BF16 · 0-shot"),
    ("elephant-ft_4096-single_choice-bf16-0shot", "Elephant FT-4,096 · Choice · BF16 · 0-shot"),
    ("elephant-original-noul-bf16-0shot", "Elephant Original · Noul · BF16 · 0-shot"),
    ("elephant-original-choice2-bf16-0shot", "Elephant Original · Choice2 · BF16 · 0-shot"),
    ("elephant-original-score5-bf16-0shot", "Elephant Original · Score5 · BF16 · 0-shot"),
    ("elephant-original-hierarchical-bf16-0shot", "Elephant Original · Hierarchical · BF16 · 0-shot"),
    ("elephant-original-parallel_noul-bf16-0shot", "Elephant Original · Parallel Noul · BF16 · 0-shot"),
    ("elephant-provided-2c62fb8392-hierarchical-bf16-0shot", "Elephant FT-4,096 · Hierarchical · BF16 · 0-shot"),
    ("elephant-provided-2c62fb8392-parallel_noul-bf16-0shot", "Elephant FT-4,096 · Parallel Noul · BF16 · 0-shot"),
    ("qwen-original-single_choice-bf16-0shot", "Qwen3.5-9B · Choice · BF16 · 0-shot"),
    ("qwen-original-single_choice-bf16-4shot", "Qwen3.5-9B · Choice · BF16 · 4-shot"),
    ("startlux27b-original-single_choice-q4_k_m-0shot-llamacpp", "StartLux 27B Instant · Choice · Q4_K_M · 0-shot"),
]

PARTITIONS = ("dev_select", "dev_calibrate", "test")
GROUPS = ("All", "Seen", "Unseen")
PARTITION_COUNTS = {"dev_select": 456, "dev_calibrate": 649, "test": 2373}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact_metric(value: dict) -> dict:
    keep = (
        "n",
        "accuracy",
        "macro_f1",
        "macro_accuracy",
        "ece",
        "brier",
        "nll",
        "aurc",
        "cost",
    )
    result = {key: value[key] for key in keep if key in value}
    if "cost" in result:
        cost = result["cost"]
        result["cost"] = {
            key: result["cost"][key]
            for key in (
                "total_seconds",
                "mean_seconds",
                "median_seconds",
                "p95_seconds",
                "request_count",
                "question_count",
                "input_tokens",
                "token_measurement_available",
            )
            if key in cost
        }
        elapsed = cost.get("total_seconds", 0.0)
        if elapsed > 0:
            result["cost"]["examples_per_second"] = value.get("n", 0) / elapsed
            result["cost"]["requests_per_second"] = cost.get("request_count", 0) / elapsed
            result["cost"]["questions_per_second"] = cost.get("question_count", 0) / elapsed
    return result


def read_scores(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc).replace(tzinfo=None)


def read_gpu0_samples() -> list[tuple[dt.datetime, int, int]]:
    samples = []
    with TELEMETRY.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) < 4:
                continue
            try:
                if int(row[1]) != 0:
                    continue
                samples.append((dt.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"), int(row[2]), int(row[3])))
            except (ValueError, IndexError):
                continue
    return samples


def gpu_window(start: dt.datetime, end: dt.datetime, samples: list[tuple[dt.datetime, int, int]]) -> dict:
    window = [sample for sample in samples if start <= sample[0] <= end]
    values = [(memory, utilization) for _, memory, utilization in window]
    if not values:
        return {"sample_count": 0, "peak_mib": None, "mean_mib": None, "p95_mib": None, "utilization_weighted_hours": 0.0}
    memories = sorted(memory for memory, _ in values)
    index = min(len(memories) - 1, int(0.95 * (len(memories) - 1)))
    return {
        "sample_count": len(values),
        "peak_mib": memories[-1],
        "mean_mib": sum(memories) / len(memories),
        "p95_mib": memories[index],
        "utilization_weighted_hours": sum(
            (window[index + 1][0] - sample[0]).total_seconds() * sample[2] / 100 / 3600
            for index, sample in enumerate(window[:-1])
        ),
    }


def load_condition(
    run_name: str,
    label: str,
    gpu_samples: list[tuple[dt.datetime, int, int]],
    canonical_ids: dict[str, list[str]],
) -> tuple[dict, dict]:
    run = RUNS / run_name
    calibration = read_json(run / "calibration.json")
    manifest = read_json(run / "test/manifest.json")
    spec = manifest["spec"]
    model = spec["model"]
    condition = {
        "id": run_name,
        "label": label,
        "task": spec["task"],
        "strategy": spec["strategy"],
        "shots": spec["shots"],
        "backend": model["backend"],
        "quantization": model["quantization"],
        "model_repository": model.get("model_repository"),
        "checkpoint_sha256": model.get("checkpoint_sha256"),
        "calibration": {
            "temperature": calibration["temperature"],
            "threshold": calibration["threshold"],
            "temperature_at_bound": calibration["temperature_at_bound"],
            "threshold_sweep_points": len(calibration["threshold_sweep_dev_select"]),
            "probability_stage": calibration["probability_stage"],
            "prediction_policy": calibration["prediction_policy"],
        },
        "partitions": {},
    }

    for partition in PARTITIONS:
        run_partition = run / partition
        manifest = read_json(run_partition / "manifest.json")
        complete = read_json(run_partition / "complete.json")
        spec = manifest["spec"]
        ids = spec["ids"]
        ids_digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        if partition in canonical_ids and ids != canonical_ids[partition]:
            raise ValueError(f"{run_name}: {partition} example IDs differ from the other conditions")
        canonical_ids.setdefault(partition, ids)
        if len(ids) != PARTITION_COUNTS[partition] or complete["n"] != PARTITION_COUNTS[partition]:
            raise ValueError(f"{run_name}: incomplete manifest or completion file for {partition}")
        if manifest["identity"] != complete["identity"]:
            raise ValueError(f"{run_name}: manifest/completion identity differs for {partition}")
        gpu = gpu_window(parse_utc(manifest["created"]), parse_utc(complete["utc"]), gpu_samples)
        base = run / "reports" / partition
        metrics = read_json(base / "metrics.json")
        if metrics["partition"] != partition:
            raise ValueError(f"{run_name}: wrong partition in {base / 'metrics.json'}")
        if metrics["metrics"]["raw"]["All"]["n"] != PARTITION_COUNTS[partition]:
            raise ValueError(f"{run_name}: wrong row count for {partition}")
        if metrics["run_identity"] != manifest["identity"]:
            raise ValueError(f"{run_name}: report identity differs from {partition} manifest")
        condition["partitions"][partition] = {
            "condition_identity": metrics["condition"],
            "calibration_sha256": metrics.get("calibration_sha256"),
            "example_ids_sha256": ids_digest,
            "gpu0_memory": gpu,
            "raw": {group: compact_metric(metrics["metrics"]["raw"][group]) for group in GROUPS},
            "calibrated": {
                group: compact_metric(metrics["metrics"]["calibrated"][group])
                for group in GROUPS
            },
        }

    raw_rows = read_scores(run / "reports/test/scored_raw.jsonl")
    calibrated_rows = read_scores(run / "reports/test/scored_calibrated.jsonl")
    if len(raw_rows) != len(calibrated_rows):
        raise ValueError(f"{run_name}: raw/calibrated row count differs")
    scores = {group: {"raw": [], "calibrated": []} for group in GROUPS}
    for raw, calibrated in zip(raw_rows, calibrated_rows, strict=True):
        if raw["example_id"] != calibrated["example_id"]:
            raise ValueError(f"{run_name}: raw/calibrated example order differs")
        if raw["gold"] != calibrated["gold"] or raw["prediction"] != calibrated["prediction"]:
            raise ValueError(f"{run_name}: calibrated output changed the frozen decision label")
        group = raw["rule_split"] if raw["rule_split"] in ("Seen", "Unseen") else "All"
        correct = raw["prediction"] == raw["gold"]
        for target in dict.fromkeys(("All", group)):
            scores[target]["raw"].append([raw["confidence"], correct])
            scores[target]["calibrated"].append([calibrated["confidence"], correct])
    if len(scores["All"]["raw"]) != condition["partitions"]["test"]["raw"]["All"]["n"]:
        raise ValueError(f"{run_name}: score rows do not match metrics count")
    return condition, scores


def write_csv(conditions: list[dict]) -> None:
    path = OUT / "condition_metrics.csv"
    fields = [
        "condition",
        "label",
        "task",
        "strategy",
        "shots",
        "backend",
        "quantization",
        "partition",
        "group",
        "n",
        "raw_accuracy",
        "calibrated_accuracy",
        "raw_macro_f1",
        "calibrated_macro_f1",
        "raw_ece",
        "calibrated_ece",
        "raw_brier",
        "calibrated_brier",
        "raw_nll",
        "calibrated_nll",
        "raw_aurc",
        "calibrated_aurc",
        "request_count",
        "question_count",
        "input_tokens",
        "mean_seconds",
        "p95_seconds",
        "examples_per_second",
        "requests_per_second",
        "questions_per_second",
        "gpu0_peak_mib",
        "gpu0_mean_mib",
        "gpu0_sample_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition in conditions:
            for partition, report in condition["partitions"].items():
                for group in GROUPS:
                    raw = report["raw"][group]
                    calibrated = report["calibrated"][group]
                    cost = raw.get("cost", {})
                    writer.writerow(
                        {
                            "condition": condition["id"],
                            "label": condition["label"],
                            "task": condition["task"],
                            "strategy": condition["strategy"],
                            "shots": condition["shots"],
                            "backend": condition["backend"],
                            "quantization": condition["quantization"],
                            "partition": partition,
                            "group": group,
                            "n": raw.get("n"),
                            "raw_accuracy": raw.get("accuracy"),
                            "calibrated_accuracy": calibrated.get("accuracy"),
                            "raw_macro_f1": raw.get("macro_f1"),
                            "calibrated_macro_f1": calibrated.get("macro_f1"),
                            "raw_ece": raw.get("ece"),
                            "calibrated_ece": calibrated.get("ece"),
                            "raw_brier": raw.get("brier"),
                            "calibrated_brier": calibrated.get("brier"),
                            "raw_nll": raw.get("nll"),
                            "calibrated_nll": calibrated.get("nll"),
                            "raw_aurc": raw.get("aurc"),
                            "calibrated_aurc": calibrated.get("aurc"),
                            "request_count": cost.get("request_count"),
                            "question_count": cost.get("question_count"),
                            "input_tokens": cost.get("input_tokens"),
                            "mean_seconds": cost.get("mean_seconds"),
                            "p95_seconds": cost.get("p95_seconds"),
                            "examples_per_second": cost.get("examples_per_second"),
                            "requests_per_second": cost.get("requests_per_second"),
                            "questions_per_second": cost.get("questions_per_second"),
                            "gpu0_peak_mib": report["gpu0_memory"]["peak_mib"],
                            "gpu0_mean_mib": report["gpu0_memory"]["mean_mib"],
                            "gpu0_sample_count": report["gpu0_memory"]["sample_count"],
                        }
                    )


def write_bar_chart(path: Path, conditions: list[dict], metric: str, series: list[tuple[str, str, str]], title: str, subtitle: str) -> None:
    width = 1720
    left = 460
    right = 80
    top = 110
    row_h = 43
    bottom = 70
    height = top + row_h * len(conditions) + bottom
    plot_w = width - left - right
    max_value = max(
        value
        for condition in conditions
        for _, flavor, group in series
        if (value := condition["partitions"]["test"][flavor][group].get(metric)) is not None
    )
    axis_max = 1.0 if metric == "accuracy" else max(0.1, math.ceil(max_value * 10) / 10)
    colors = {"All": "#245b8f", "Seen": "#e07a2d", "Unseen": "#33866b", "raw": "#577d9f", "calibrated": "#d28b42"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(subtitle)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="36" y="42" font-family="Arial,sans-serif" font-size="28" font-weight="700" fill="#172b3a">{html.escape(title)}</text>',
        f'<text x="36" y="70" font-family="Arial,sans-serif" font-size="15" fill="#485b68">{html.escape(subtitle)}</text>',
    ]
    for tick in range(0, 11):
        value = axis_max * tick / 10
        x = left + plot_w * value / axis_max
        parts.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-bottom+8}" stroke="#dce3e8" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+32}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#52636f">{value:.0%}</text>' if metric == "accuracy" else f'<text x="{x:.1f}" y="{height-bottom+32}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#52636f">{value:.2f}</text>')
    bar_h = 7
    gap = 2
    block_h = len(series) * (bar_h + gap) - gap
    for idx, condition in enumerate(conditions):
        row_y = top + idx * row_h
        parts.append(f'<text x="{left-16}" y="{row_y+19}" text-anchor="end" font-family="Arial,sans-serif" font-size="13" fill="#203440">{html.escape(condition["label"])}</text>')
        for s_idx, (legend, flavor, group) in enumerate(series):
            value = condition["partitions"]["test"][flavor][group].get(metric)
            y = row_y + (row_h - block_h) / 2 + s_idx * (bar_h + gap)
            if value is None:
                continue
            bar_w = max(0.5, plot_w * value / axis_max)
            parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="{bar_h}" rx="2" fill="{colors.get(legend, "#557")}"/>')
    legend_y = height - 19
    legend_x = left
    for legend, _, _ in series:
        color = colors.get(legend, "#557")
        parts.append(f'<rect x="{legend_x}" y="{legend_y-11}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+18}" y="{legend_y}" font-family="Arial,sans-serif" font-size="12" fill="#394c58">{html.escape(legend)}</text>')
        legend_x += max(90, len(legend) * 8 + 42)
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_dashboard(conditions: list[dict], test_scores: dict[str, dict]) -> None:
    conditions_json = json.dumps(conditions, ensure_ascii=False, separators=(",", ":"))
    scores_json = json.dumps(test_scores, separators=(",", ":"))
    labels = "\n".join(
        f'<option value="{html.escape(condition["id"], quote=True)}">{html.escape(condition["label"])}</option>'
        for condition in conditions
    )
    page = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OR-ShARC Experiment B · Selective-risk dashboard</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#1d2a32;background:#f8fafb}}
h1{{margin-bottom:4px}} .note{{background:#fff5d8;border-left:4px solid #d89c27;padding:12px 16px;margin:18px 0}}
.controls,.cards,.metrics{{display:flex;gap:14px;flex-wrap:wrap}} .controls{{align-items:end;background:white;border:1px solid #dce3e8;border-radius:10px;padding:18px}}
label{{display:grid;gap:6px;font-size:14px;color:#425460}} select,input{{font:inherit;padding:7px;border:1px solid #bcc9d1;border-radius:6px}}
select{{min-width:260px}} .slider{{min-width:320px;flex:1}} .cards{{margin:18px 0}} .card{{background:white;border:1px solid #dce3e8;border-radius:10px;padding:16px 18px;min-width:155px;flex:1}}
.card b{{display:block;font-size:25px;color:#174e78}} table{{border-collapse:collapse;background:white;width:100%;margin:14px 0}} td,th{{padding:9px 11px;border-bottom:1px solid #e1e7eb;text-align:left}}
th{{background:#edf2f5}} .muted{{color:#60717c;font-size:14px}} a{{color:#145a8b}} .group{{display:flex;gap:10px;align-items:center}}
</style>
<h1>OR-ShARC Experiment B</h1><div class="muted">Test-set selective-risk explorer · 14 fixed conditions · 2,373 examples per condition</div>
<div class="note"><b>Use care:</b> this slider explores test outcomes. The official thresholds were selected on <code>dev_select</code>; the temperatures were fit on <code>dev_calibrate</code>. Do not use the test slider to choose or revise the reported policy.</div>
<div class="controls">
 <label>Condition<select id="condition">{labels}</select></label>
 <label>Group<select id="group"><option>All</option><option>Seen</option><option>Unseen</option></select></label>
 <label>Confidence<select id="flavor"><option value="raw">Raw</option><option value="calibrated">Calibrated</option></select></label>
 <label class="slider">Minimum confidence <span id="thresholdLabel">0.50</span><input id="threshold" type="range" min="0" max="1" step="0.01" value="0.50"></label>
</div>
<div class="cards">
 <div class="card">Accepted examples<b id="accepted">—</b></div><div class="card">Coverage<b id="coverage">—</b></div>
 <div class="card">Selective accuracy<b id="accuracy">—</b></div><div class="card">Selective risk<b id="risk">—</b></div>
 <div class="card">Mean confidence<b id="meanConfidence">—</b></div>
</div>
<h2>Frozen condition facts</h2><table><tbody id="facts"></tbody></table>
<p class="muted">Accuracy compares each prediction with the saved gold label for its task. Risk is 1 − accuracy among accepted examples. Seen/Unseen uses official rule membership from the complete train split.</p>
<p><a href="../../../EXPERIMENT_B_REPORT.md">Open the experiment report</a> · <a href="condition_metrics.csv">Download all condition metrics</a> · <a href="test_accuracy_by_condition.svg">View accuracy figure</a> · <a href="test_aurc_before_after.svg">View risk–coverage figure</a></p>
<script>
const conditions={conditions_json}; const scores={scores_json};
const el=id=>document.getElementById(id); const pct=x=>Number.isFinite(x)?(100*x).toFixed(1)+'%':'—';
function update(){{
 const id=el('condition').value, group=el('group').value, flavor=el('flavor').value;
 const threshold=Number(el('threshold').value); el('thresholdLabel').textContent=threshold.toFixed(2);
 const rows=scores[id][group][flavor]; let n=0,correct=0,sum=0;
 for(const [confidence,isCorrect] of rows) if(confidence>=threshold){{n++;sum+=confidence;if(isCorrect)correct++;}}
 el('accepted').textContent=n.toLocaleString()+' / '+rows.length.toLocaleString();
 el('coverage').textContent=pct(n/rows.length); el('accuracy').textContent=n?pct(correct/n):'—';
 el('risk').textContent=n?pct(1-correct/n):'—'; el('meanConfidence').textContent=n?(sum/n).toFixed(3):'—';
 const c=conditions.find(x=>x.id===id), cal=c.calibration, t=c.partitions.test;
 const r=t.raw[group], k=t.calibrated[group];
 const facts=[['Task / strategy',c.task+' / '+c.strategy],['Shots / quantization',c.shots+' / '+c.quantization],['Frozen temperature',cal.temperature.toFixed(4)],['Frozen threshold',cal.threshold.toFixed(3)],['Official test n',r.n.toLocaleString()],['Raw test accuracy / ECE',pct(r.accuracy)+' / '+r.ece.toFixed(3)],['Calibrated test accuracy / ECE',pct(k.accuracy)+' / '+k.ece.toFixed(3)],['Raw avg / p95 latency',r.cost.mean_seconds.toFixed(3)+'s / '+r.cost.p95_seconds.toFixed(3)+'s'],['Raw example throughput',r.cost.examples_per_second.toFixed(2)+' examples/s'],['Sampled GPU0 peak',c.gpu0_peak_mib.toLocaleString()+' MiB']];
 el('facts').innerHTML=facts.map(([a,b])=>'<tr><th>'+a+'</th><td>'+b+'</td></tr>').join('');
}}
for(const id of ['condition','group','flavor','threshold'])el(id).addEventListener('input',update); update();
</script></html>'''
    (OUT / "test_threshold_dashboard.html").write_text(page, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conditions = []
    test_scores = {}
    gpu_samples = read_gpu0_samples()
    canonical_ids = {}
    for run_name, label in CONDITIONS:
        condition, scores = load_condition(run_name, label, gpu_samples, canonical_ids)
        condition["gpu0_peak_mib"] = max(
            condition["partitions"][partition]["gpu0_memory"]["peak_mib"] or 0
            for partition in PARTITIONS
        )
        conditions.append(condition)
        test_scores[run_name] = scores

    total_rows = sum(condition["partitions"]["test"]["raw"]["All"]["n"] for condition in conditions)
    totals = {"backend_requests": 0, "questions": 0, "input_tokens": 0, "reported_seconds": 0.0}
    for condition in conditions:
        for partition in PARTITIONS:
            cost = condition["partitions"][partition]["raw"]["All"].get("cost", {})
            totals["backend_requests"] += cost.get("request_count", 0)
            totals["questions"] += cost.get("question_count", 0)
            totals["input_tokens"] += cost.get("input_tokens", 0)
            totals["reported_seconds"] += cost.get("total_seconds", 0.0)
    resident_seconds = sum(
        (gpu_samples[index + 1][0] - sample[0]).total_seconds()
        for index, sample in enumerate(gpu_samples[:-1])
        if sample[1] > 100
    )
    weighted_seconds = sum(
        (gpu_samples[index + 1][0] - sample[0]).total_seconds() * sample[2] / 100
        for index, sample in enumerate(gpu_samples[:-1])
    )
    summary = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "condition_count": len(conditions),
        "condition_example_rows": sum(
            condition["partitions"][partition]["raw"]["All"]["n"]
            for condition in conditions
            for partition in PARTITIONS
        ),
        "test_example_rows": total_rows,
        "partitions": {partition: conditions[0]["partitions"][partition]["raw"]["All"]["n"] for partition in PARTITIONS},
        "partition_example_ids_sha256": {
            partition: hashlib.sha256(json.dumps(canonical_ids[partition], separators=(",", ":")).encode()).hexdigest()
            for partition in PARTITIONS
        },
        "cost_totals": totals,
        "gpu_telemetry": {
            "source": str(TELEMETRY.relative_to(ROOT)),
            "device_index": 0,
            "sample_count": len(gpu_samples),
            "start_utc": gpu_samples[0][0].isoformat(timespec="milliseconds") + "Z",
            "end_utc": gpu_samples[-1][0].isoformat(timespec="milliseconds") + "Z",
            "memory_resident_over_100_mib_hours": resident_seconds / 3600,
            "utilization_weighted_hours": weighted_seconds / 3600,
            "peak_memory_mib": max(sample[1] for sample in gpu_samples),
        },
        "conditions": conditions,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(conditions)
    benchmark = [condition for condition in conditions if condition["task"] == "benchmark"]
    sufficiency = [condition for condition in conditions if condition["task"] == "sufficiency"]
    write_bar_chart(
        OUT / "test_accuracy_by_condition.svg",
        benchmark,
        "accuracy",
        [(group, "raw", group) for group in GROUPS],
        "Test accuracy by condition",
        "Benchmark task only; official test split; bars show All, Seen and Unseen accuracy.",
    )
    write_bar_chart(
        OUT / "test_ece_before_after.svg",
        benchmark,
        "ece",
        [("raw", "raw", "All"), ("calibrated", "calibrated", "All")],
        "Test calibration error before and after dev calibration",
        "Lower ECE is better. Temperature was fit on dev_calibrate; test is reporting only.",
    )
    write_bar_chart(
        OUT / "test_aurc_before_after.svg",
        benchmark,
        "aurc",
        [("raw", "raw", "All"), ("calibrated", "calibrated", "All")],
        "Test risk–coverage area before and after dev calibration",
        "Lower AURC is better. Curves use all test confidences; calibration does not select labels.",
    )
    write_bar_chart(
        OUT / "sufficiency_accuracy.svg",
        sufficiency,
        "accuracy",
        [(group, "raw", group) for group in GROUPS],
        "Test accuracy for sufficiency conditions",
        "Sufficiency task only; official test split; bars show All, Seen and Unseen accuracy.",
    )
    write_bar_chart(
        OUT / "sufficiency_aurc_before_after.svg",
        sufficiency,
        "aurc",
        [("raw", "raw", "All"), ("calibrated", "calibrated", "All")],
        "Sufficiency test risk–coverage area",
        "Lower AURC is better. Thresholds were selected on dev_select.",
    )
    write_dashboard(conditions, test_scores)
    print(f"conditions={len(conditions)} test_example_rows={total_rows} condition_example_rows={summary['condition_example_rows']}")
    print(f"requests={totals['backend_requests']} questions={totals['questions']} input_tokens={totals['input_tokens']} reported_seconds={totals['reported_seconds']:.1f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
