"""Bounded admission measurements using the unchanged upstream Elephant trainer."""

import gc
import hashlib
import math
import time
from pathlib import Path

from .common import checkpoint_hash, code_hash, digest, environment, read_json, rows, signature, write_json, write_rows
from .data import answer_label, load_gold, load_inputs, validate_prepared
from .runner import ROOT, model_config, runtime_sources
from .tasks import questions, training_example


def metadata_audit(prepared, output):
    manifest = validate_prepared(prepared)
    train = load_inputs(prepared, "train")
    official_rules = {r["rule_id"] for r in train}
    budgets = {str(n): {r["rule_id"] for r in load_inputs(prepared, f"train_{n}")} for n in (256, 1024, 4096)}
    counts, errors = {}, []
    for split in ("train", "dev", "test"):
        raw = rows(ROOT / "artifacts/raw" / f"open_retrieval_sharc_{split}.json")
        gold = load_gold(prepared, split)
        inputs = load_inputs(prepared, split)
        nonstandard, label_counts = {}, {}
        for r in raw:
            label = answer_label(r["answer"])
            if gold[r["utterance_id"]]["gold"] != label:
                errors.append(r["utterance_id"])
            label_counts[label] = label_counts.get(label, 0) + 1
            if r["answer"] not in ("Yes", "No") and not r["answer"].endswith("?"):
                nonstandard[r["answer"]] = nonstandard.get(r["answer"], 0) + 1
        for r in inputs:
            if (r["rule_split"] == "Seen") != (r["rule_id"] in official_rules):
                errors.append(r["example_id"])
            if split != "train" and r["snippet_seen"] != (r["rule_id"] in official_rules):
                errors.append(r["example_id"])
            if any(r["ft_exposed"][n] != (r["rule_id"] in rules) for n, rules in budgets.items()):
                errors.append(r["example_id"])
        counts[split] = {
            "labels": label_counts,
            "nonstandard_followups": nonstandard,
            "seen_but_not_ft_exposed": {
                n: sum(r["rule_split"] == "Seen" and not r["ft_exposed"][n] for r in inputs) for n in budgets
            },
        }
    assert answer_label("yes") == "Yes" and answer_label("NO") == "No"
    for value in ("Irrelevant", " unknown ", "Maybe", ""):
        try:
            answer_label(value)
        except ValueError:
            continue
        raise AssertionError(f"unexpected classification accepted: {value!r}")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "counts": counts,
        "data_manifest": signature(manifest),
        "partitions_sha256": digest(Path(prepared) / "partitions.json"),
        "budget_rule_ids": {n: sorted(rules) for n, rules in budgets.items()},
        "definitions": {
            "snippet_seen": "rule in full official train",
            "ft_exposed": "rule in specified complete nested training budget; partial smoke is not a full-budget checkpoint",
        },
        "test_scope": "static labels/metadata audit only; no model inference or test evaluation",
    }
    write_json(output, result)
    assert not errors
    return result


def measure(prepared, output):
    config = model_config("elephant")
    source = runtime_sources(config)
    import torch
    import peft
    from safetensors.torch import load_file
    from train import common, sft
    from qev.readout import Readout, ReadoutConfig

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    hp = read_json(ROOT / "configs/training.json")["onejev"]
    write_json(
        output / "manifest.json",
        {
            "code_sha256": code_hash(),
            "runtime": source,
            "environment": environment(),
            "checkpoint": checkpoint_hash(config["checkpoint"]),
            "data": signature(validate_prepared(prepared)),
            "hp": hp,
            "scope": "3 native steps on 24 real train rows; train/dev inference only",
            "tolerances": {"adapter_reload_atol": 1e-6, "bf16_merge_atol": 0.02, "rtol": 0, "label_changes": 0},
        },
    )
    started = time.perf_counter()
    bundle = common.load_bundle(config["checkpoint"], load_model=False)
    all_rows, distribution, panel = {}, {}, []
    for split in ("train", "dev"):
        converted = []
        for r in load_inputs(prepared, split):
            ex = training_example(r, "Yes", "benchmark", "single_choice")
            native = {
                "id": r["example_id"],
                "state": ex["state"],
                "question": ex["questions"]["decision"],
                "target": {"Yes": 1.0},
                "source": "orsharc_" + split,
                "task": "benchmark",
                "split": split,
            }
            n = len(bundle.encode(common.render_example(native, bundle.template_for(native), 0, seed=None).prompt))
            converted.append((n, r["example_id"], native))
        converted.sort(key=lambda r: (r[0], r[1]))
        all_rows[split] = converted
        distribution[split] = {
            "n": len(converted),
            "tokens": sum(x[0] for x in converted),
            "over_limit": sum(x[0] > hp["max_length"] for x in converted),
        }
        for q in (0.5, 0.9, 0.95, 0.99, 1.0):
            entry = converted[math.ceil(q * len(converted)) - 1]
            panel.append(
                {"split": split, "percentile": q, "tokens": entry[0], "example_id": entry[1], "record": entry[2]}
            )
            distribution[split][str(q)] = entry[0]
    train_by_id = {i: (n, r) for n, i, r in all_rows["train"]}
    for n in (256, 1024, 4096):
        ids = read_json(Path(prepared) / "partitions.json")[f"train_{n}"]
        distribution[f"train_{n}"] = {
            "n": n,
            "tokens": sum(train_by_id[i][0] for i in ids),
            "max_tokens": max(train_by_id[i][0] for i in ids),
        }
    write_json(
        output / "lengths.json",
        {
            "distribution": distribution,
            "panel": panel,
            "max_length": hp["max_length"],
            "seconds": time.perf_counter() - started,
        },
    )
    assert not any(distribution[s]["over_limit"] for s in ("train", "dev"))
    # 24 distinct real training records spanning the upper tail, including the maximum.
    chosen = {p["example_id"] for p in panel if p["split"] == "train"}
    for j in range(24):
        chosen.add(all_rows["train"][round((0.90 + 0.10 * j / 23) * (len(all_rows["train"]) - 1))][1])
    selected_ids = sorted(chosen)
    while len(selected_ids) > 24:
        selected_ids.remove(
            next(i for i in selected_ids if i not in {p["example_id"] for p in panel if p["split"] == "train"})
        )
    gold = load_gold(prepared, "train")
    native_train = []
    for i in selected_ids:
        r = dict(train_by_id[i][1])
        r["target"] = {gold[i]["gold"]: 1.0}
        native_train.append(r)
    write_rows(output / "train.jsonl", native_train)
    write_rows(output / "panel.jsonl", panel)
    held = {}
    original_load, original_peft, original_init, original_save = (
        common.load_bundle,
        peft.get_peft_model,
        torch.optim.AdamW.__init__,
        sft.save_weights,
    )

    def tensor_hash(p):
        return hashlib.sha256(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    def frozen(model):
        return {n: tensor_hash(p) for n, p in model.named_parameters() if not p.requires_grad}

    def probabilities(b, m):
        m.eval()
        if hasattr(m, "gradient_checkpointing_disable"):
            m.gradient_checkpointing_disable()
        result = []
        with torch.inference_mode():
            for entry in panel:
                r = entry["record"]
                ex = common.render_example(r, b.template_for(r), 0, seed=None)
                t = time.perf_counter()
                logits, mask = common.batch_scores(b, m, [ex], [r], torch.device("cuda:0"))
                p = logits[0, : len(ex.labels)].float().softmax(-1).cpu().tolist()
                torch.cuda.synchronize()
                result.append(
                    {
                        "example_id": entry["example_id"],
                        "labels": ex.labels,
                        "probabilities": dict(zip(ex.labels, p)),
                        "prediction": ex.labels[max(range(len(p)), key=p.__getitem__)],
                        "tokens": entry["tokens"],
                        "seconds": time.perf_counter() - t,
                    }
                )
        return result

    def capture_bundle(*args, **kwargs):
        b = original_load(*args, **kwargs)
        held["bundle"] = b
        return b

    def capture_peft(*args, **kwargs):
        m = original_peft(*args, **kwargs)
        held["model"] = m
        return m

    def capture_optimizer(opt, *args, **kwargs):
        original_init(opt, *args, **kwargs)
        m = held["model"]
        trainable = {n: p for n, p in m.named_parameters() if p.requires_grad}
        optim_ids = [id(p) for group in opt.param_groups for p in group["params"]]
        assert set(optim_ids) == {id(p) for p in trainable.values()} and len(optim_ids) == len(trainable)
        assert trainable and all("lora_" in n for n in trainable)
        held["before"] = {n: p.detach().cpu().clone() for n, p in trainable.items()}
        held["frozen_before"] = frozen(m)
        held["gradients"] = {n: [] for n in trainable}
        write_json(
            output / "optimizer.json",
            {
                "status": "PASS",
                "parameters": {n: p.numel() for n, p in trainable.items()},
                "count": sum(p.numel() for p in trainable.values()),
                "exact_membership": True,
                "frozen_before": held["frozen_before"],
            },
        )

        def before_step(opt, args, kwargs):
            for n, p in trainable.items():
                assert p.grad is not None and bool(torch.isfinite(p.grad).all()), n
                held["gradients"][n].append(float(p.grad.detach().abs().max()))

        opt.register_step_pre_hook(before_step)

    def capture_save(m, b, directory, *args, **kwargs):
        t = time.perf_counter()
        original_save(m, b, directory, *args, **kwargs)
        write_json(output / "save.json", {"adapter_seconds": time.perf_counter() - t})
        after = frozen(m)
        changed = [n for n in after if held["frozen_before"].get(n) != after[n]]
        updates = {
            n: float((p.detach().cpu() - held["before"][n]).abs().max())
            for n, p in m.named_parameters()
            if p.requires_grad
        }
        evidence = {
            "status": "PASS"
            if not changed
            and all(v > 0 for v in updates.values())
            and all(max(v) > 0 for v in held["gradients"].values())
            else "FAIL",
            "frozen_changed": changed,
            "frozen_after": after,
            "max_abs_updates": updates,
            "gradient_maxima_per_step": held["gradients"],
        }
        saved = load_file(str(directory / "adapter_model.safetensors"))
        active = peft.get_peft_model_state_dict(m)
        evidence["saved_adapter_exact"] = set(saved) == set(active) and all(
            torch.equal(saved[k], active[k].cpu()) for k in saved
        )
        write_json(output / "integrity.json", evidence)
        assert evidence["status"] == "PASS" and evidence["saved_adapter_exact"]
        write_rows(output / "active.jsonl", probabilities(b, m))
        reload_bundle = original_load(config["checkpoint"])
        reload_bundle.readout = Readout.attach(
            reload_bundle.model,
            ReadoutConfig.load(config["checkpoint"]),
            slot_ids=reload_bundle.slot_ids,
            device=torch.device("cuda:0"),
        )
        reload_model = peft.PeftModel.from_pretrained(reload_bundle.model, str(directory))
        write_rows(output / "reloaded.jsonl", probabilities(reload_bundle, reload_model))
        del reload_model, reload_bundle
        gc.collect()
        torch.cuda.empty_cache()

    common.load_bundle, peft.get_peft_model, torch.optim.AdamW.__init__, sft.save_weights = (
        capture_bundle,
        capture_peft,
        capture_optimizer,
        capture_save,
    )
    t = time.perf_counter()
    try:
        sft.main(
            [
                "--model",
                config["checkpoint"],
                "--device",
                "cuda:0",
                "--train",
                str(output / "train.jsonl"),
                "--runs-dir",
                str(output),
                "--name",
                "native",
                "--seed",
                "42",
                "--no-augment",
                "--attention",
                "noncausal",
                "--readout",
                "slot_lm",
                "--lr",
                str(hp["learning_rate"]),
                "--epochs",
                "1",
                "--lora-r",
                str(hp["rank"]),
                "--lora-alpha",
                str(hp["alpha"]),
                "--max-len",
                str(hp["max_length"]),
                "--batch-tokens",
                str(hp["max_length"]),
                "--max-rows",
                "1",
                "--accum-rows",
                "8",
                "--max-steps",
                "3",
                "--log-every",
                "1",
                "--save-every",
                "0",
                "--merge",
            ]
        )
    finally:
        common.load_bundle, peft.get_peft_model, torch.optim.AdamW.__init__, sft.save_weights = (
            original_load,
            original_peft,
            original_init,
            original_save,
        )
    native_seconds = time.perf_counter() - t
    held.clear()
    gc.collect()
    torch.cuda.empty_cache()
    merged_dir = output / "native/merged"
    # Same metadata copy as the existing production wrapper.
    import shutil

    shutil.copy2(output / "native/adapter/readout.json", merged_dir / "readout.json")
    t = time.perf_counter()
    merged_bundle = original_load(str(merged_dir))
    merged_bundle.readout = Readout.attach(
        merged_bundle.model,
        ReadoutConfig.load(merged_dir),
        slot_ids=merged_bundle.slot_ids,
        device=torch.device("cuda:0"),
    )
    merged_load_seconds = time.perf_counter() - t
    write_rows(output / "merged.jsonl", probabilities(merged_bundle, merged_bundle.model))
    del merged_bundle
    gc.collect()
    torch.cuda.empty_cache()
    parity = {}
    active = rows(output / "active.jsonl")
    for name, tolerance in (("reloaded", 1e-6), ("merged", 0.02)):
        compared = rows(output / f"{name}.jsonl")
        difference = max(
            abs(a["probabilities"][k] - b["probabilities"][k]) for a, b in zip(active, compared) for k in a["labels"]
        )
        changes = [a["example_id"] for a, b in zip(active, compared) if a["prediction"] != b["prediction"]]
        parity[name] = {
            "status": "PASS" if difference <= tolerance and not changes else "FAIL",
            "max_abs_probability_difference": difference,
            "label_changes": changes,
            "atol": tolerance,
            "rtol": 0,
            "n": len(compared),
        }
    write_json(output / "parity.json", parity)
    logs = [r for r in rows(output / "native/log.jsonl") if "loss" in r]
    assert len(logs) == 3 and all(math.isfinite(r["loss"]) for r in logs)
    write_json(
        output / "profile.json",
        {
            "status": "PASS",
            "native_total_seconds": native_seconds,
            "merged_load_seconds": merged_load_seconds,
            "training_steps": logs,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "truncated_records": 0,
            "dropped_records": 0,
            "training_ids": selected_ids,
            "training_tokens": sum(train_by_id[i][0] for i in selected_ids),
            "checkpoint": checkpoint_hash(merged_dir),
            "convolution": "existing PyTorch reference fallback",
        },
    )
    assert all(r["status"] == "PASS" for r in parity.values()), parity
    return parity


def serving(measurement):
    """Measure the production inference adapter on the recorded real-input panel."""
    import torch
    import peft
    from .backends import QevBackend
    from qev.prompt import render_question

    output = Path(measurement)
    panel = rows(output / "panel.jsonl")
    config = model_config("elephant")
    runtime_sources(config)
    results, peaks, load_times = {}, {}, {}
    for name, checkpoint in (("adapter", config["checkpoint"]), ("merged", str(output / "native/merged"))):
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        backend = QevBackend({**config, "checkpoint": checkpoint}, [])
        if name == "adapter":
            backend.engine.model = peft.PeftModel.from_pretrained(
                backend.engine.model, str(output / "native/adapter")
            ).eval()
        load_times[name] = time.perf_counter() - begin
        qs = questions("benchmark", "single_choice")
        backend.predict(panel[0]["record"]["state"], qs)
        records = []
        for entry in panel:
            req = backend.request(state=entry["record"]["state"], questions=qs)
            prefix, suffixes = backend.engine._prefix_and_suffix_ids(
                req.state, [render_question(k, q) for k, q in req.questions.items()]
            )
            expected = len(prefix) + sum(map(len, suffixes))
            t = time.perf_counter()
            answer = backend.predict(req.state, qs)
            elapsed = time.perf_counter() - t
            assert answer["usage"]["input_tokens"] == expected == entry["tokens"]
            p = answer["answers"]["decision"]
            records.append(
                {
                    "example_id": entry["example_id"],
                    "split": entry["split"],
                    "percentile": entry["percentile"],
                    "probabilities": p,
                    "prediction": max(p, key=p.__getitem__),
                    "seconds": elapsed,
                    "input_tokens": expected,
                    "usage": answer["usage"],
                    "stats": answer["stats"],
                }
            )
        # Reject an actual over-limit request before execution, without truncating it.
        backend.engine.max_branch_tokens = entry["tokens"] - 1
        try:
            backend.predict(entry["record"]["state"], qs)
        except ValueError as exc:
            assert "limit" in str(exc)
        else:
            raise AssertionError("over-context request was not rejected")
        backend.engine.max_branch_tokens = config["max_length"]
        peaks[name] = torch.cuda.max_memory_allocated()
        results[name] = records
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    difference = max(
        abs(a["probabilities"][k] - b["probabilities"][k])
        for a, b in zip(results["adapter"], results["merged"])
        for k in a["probabilities"]
    )
    changes = [
        a["example_id"] for a, b in zip(results["adapter"], results["merged"]) if a["prediction"] != b["prediction"]
    ]
    result = {
        "status": "PASS" if difference <= 0.05 and not changes else "FAIL",
        "results": results,
        "max_abs_probability_difference": difference,
        "label_changes": changes,
        "atol": 0.05,
        "rtol": 0,
        "peak_gpu_bytes": peaks,
        "load_seconds": load_times,
        "truncated_records": 0,
        "over_limit_rejected": True,
        "head_precision": "production FP32 LM head; BF16 backbone; FP32 LoRA before merge",
        "code_sha256": code_hash(),
        "tolerance_basis": "upstream qev.engine documents BF16 cached/batched rounding and uses 0.05; native full-forward parity remains 0.02; no changed labels allowed",
    }
    write_json(output / "serving.json", result)
    assert result["status"] == "PASS", {k: result[k] for k in ("max_abs_probability_difference", "label_changes")}
    return {k: v for k, v in result.items() if k != "results"}
