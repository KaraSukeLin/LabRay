"""Export native records and invoke inspected LoRA recipes; no custom heads/losses."""

import math
import re
import subprocess
import sys
import time
from pathlib import Path

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
    write_rows,
)
from .data import load_gold, load_inputs, validate_prepared
from .runner import ROOT, runtime_sources, validate_config
from .tasks import keys, questions, training_example


def export(prepared, output, size=256, task="benchmark", strategy="single_choice", recipe="onejev"):
    validate_prepared(prepared)
    if size not in (256, 1024, 4096):
        raise ValueError("registered training sizes are 256/1024/4096")
    if strategy == "score5":
        raise ValueError("Score is evaluation only; five-level gold does not exist")
    files = {}
    for partition, split in ((f"train_{size}", "train"), ("dev_select", "dev")):
        gold = load_gold(prepared, split)
        examples = [
            training_example(r, gold[r["example_id"]]["gold"], task, strategy) for r in load_inputs(prepared, partition)
        ]
        converted = []
        for example in examples:
            if recipe == "onejev":
                for qid, q in example["questions"].items():
                    label = example["gold"][qid]
                    label = str(label).lower() if isinstance(label, bool) else str(label)
                    converted.append(
                        {
                            "id": example["example_id"] + ":" + qid,
                            "state": example["state"],
                            "question": q,
                            "target": {label: 1.0},
                            "source": "orsharc_train" if split == "train" else "orsharc_dev_select",
                            "task": task,
                            "split": split,
                        }
                    )
            elif recipe == "kev":
                converted.append(
                    {
                        "state": example["state"],
                        "questions": {k: {**q, "label": example["gold"][k]} for k, q in example["questions"].items()},
                        "_meta": {"source": "orsharc", "id": example["example_id"], "split": split},
                    }
                )
            elif recipe == "qwen":
                converted.append(example)
            else:
                raise ValueError(f"unsupported export recipe: {recipe}")
        name = "train" if split == "train" else "dev_select"
        path = Path(output) / f"{name}.jsonl"
        write_rows(path, converted)
        files[name] = {"path": str(path.resolve()), "sha256": digest(path), "records": len(converted)}
    write_json(
        Path(output) / "manifest.json",
        {
            "recipe": recipe,
            "size": size,
            "task": task,
            "strategy": strategy,
            "seed": 42,
            "files": files,
            "selection": "dev_select only; no dev_calibrate or test labels",
        },
    )
    return files


def train(
    config,
    prepared,
    output,
    size=256,
    task="benchmark",
    strategy="single_choice",
    execute=False,
    max_steps=0,
    resume=False,
):
    validate_config(config)
    if config.get("training_blocker"):
        raise ValueError(config["training_blocker"])
    questions(task, strategy)
    if strategy == "score5":
        raise ValueError("Score has no five-level gold; training is unsupported")
    if config["backend"] == "startlux":
        raise ValueError(
            "StartLux v1.1 fine-tuning is unsupported: published trainer is under v1/ and must not be assumed compatible"
        )
    if config["backend"] not in ("qev", "kev", "qwen") or config["quantization"] != "bf16":
        raise ValueError("training is bf16 LoRA only for the registered qev/kev/qwen paths")
    if resume and config["backend"] != "qwen":
        raise ValueError(
            "official OneJev/Kev LoRA trainers do not restore optimizer/scheduler/RNG; restart in a NEW attempt directory"
        )
    output = Path(output).resolve()
    recipe = {"qev": "onejev", "kev": "kev", "qwen": "qwen"}[config["backend"]]
    hp = read_json(ROOT / "configs/training.json")[recipe]
    if not execute:
        return {
            "execute": False,
            "model": config,
            "recipe": recipe,
            "size": size,
            "task": task,
            "strategy": strategy,
            "hyperparameters": hp,
            "max_steps": max_steps,
            "out": str(output),
            "next": "add --execute in Phase 2, after the train-only GPU gates",
        }
    runtime = runtime_sources(config)
    with owner(output):
        manifest_path = output / "manifest.json"
        spec = {
            "model": config,
            "checkpoint": checkpoint_hash(config["checkpoint"]),
            "runtime": runtime,
            "data": signature(validate_prepared(prepared)),
            "code_sha256": code_hash(),
            "environment": environment(),
            "size": size,
            "task": task,
            "strategy": strategy,
            "recipe": recipe,
            "hp": hp,
            "seed": 42,
            "max_steps": max_steps,
        }
        identity = signature(spec)
        if manifest_path.exists():
            if read_json(manifest_path)["identity"] != identity:
                raise ValueError("training identity changed; use a new attempt directory")
            if (output / "complete.json").exists():
                expected = read_json(output / "complete.json")["checkpoint"]
                if checkpoint_hash(expected["path"])["sha256"] != expected["sha256"]:
                    raise ValueError("completed checkpoint changed")
                return {"status": "skipped", "output": str(output)}
            if not resume:
                raise ValueError(
                    "incomplete attempt exists; preserve it and use a new directory, or explicit Qwen --resume"
                )
        else:
            write_json(manifest_path, {"identity": identity, "spec": spec, "created": utc()})
        files = export(prepared, output / "data", size, task, strategy, recipe)
        start = time.perf_counter()
        append(output / "events.jsonl", {"event": "start", "utc": utc(), "resume": resume})
        try:
            if recipe == "qwen":
                checkpoint = train_qwen(config, output, files, hp, max_steps, resume)
            else:
                upstream = ROOT / "artifacts/upstream" / recipe
                write_json(output / "length_audit.json", preflight_native(recipe, config, files, hp))
                if recipe == "onejev":
                    # The patched trainer's CLI defaults do NOT read the model's attention override.
                    # Pass the released model's explicit configuration through unchanged.
                    from qev.readout import ReadoutConfig

                    rcfg = ReadoutConfig.load(config["checkpoint"])
                    if rcfg.readout != "slot_lm":
                        raise ValueError(
                            "this recipe warm-starts slot_lm checkpoints only; never initialize a replacement head"
                        )
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "train.sft",
                        "--model",
                        config["checkpoint"],
                        "--device",
                        config.get("device", "cuda:0"),
                        "--train",
                        files["train"]["path"],
                        "--eval",
                        "dev_select=" + files["dev_select"]["path"],
                        "--runs-dir",
                        str(output),
                        "--name",
                        "native",
                        "--seed",
                        "42",
                        "--no-augment",
                        "--attention",
                        rcfg.attention,
                        "--readout",
                        rcfg.readout,
                        "--lr",
                        str(hp["learning_rate"]),
                        "--epochs",
                        str(hp["epochs"]),
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
                        "--log-every",
                        "1",
                        "--eval-at-start",
                        "--merge",
                    ]
                    if max_steps:
                        command += ["--max-steps", str(max_steps), "--eval-limit", "8"]
                    checkpoint = output / "native/merged"
                else:
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "kev.train",
                        "--device",
                        config.get("device", "cuda:0"),
                        "--data",
                        files["train"]["path"],
                        "--init_from",
                        config["checkpoint"],
                        "--base",
                        hp["base"],
                        "--base_revision",
                        hp["base_revision"],
                        "--out",
                        str(output / "native"),
                        "--dtype",
                        "bf16",
                        "--weights_dtype",
                        "bf16",
                        "--lora",
                        str(hp["rank"]),
                        "--lr",
                        str(hp["learning_rate"]),
                        "--epochs",
                        str(hp["epochs"]),
                        "--batch",
                        "1",
                        "--accum",
                        "8",
                        "--checkpointing",
                        "1",
                        "--seed",
                        "42",
                        "--max_state",
                        str(hp["max_length"]),
                        "--p_none",
                        "0",
                        "--p_none_distract",
                        "0",
                        "--p_distract",
                        "0",
                        "--perm_frac",
                        "0",
                    ]
                    if max_steps:
                        command += ["--max_steps", str(max_steps)]
                    checkpoint = output / "native"
                write_json(output / "command.json", {"argv": command, "cwd": str(upstream.resolve())})
                with (output / "console.log").open("a") as log:
                    process = subprocess.Popen(
                        command, cwd=upstream, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                    )
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        match = re.search(r"ep(\d+) step (\d+)/(\d+) loss ([\d.eE+-]+)", line)
                        if match:
                            append(
                                output / "training.jsonl",
                                {
                                    "epoch": int(match[1]),
                                    "step": int(match[2]),
                                    "steps": int(match[3]),
                                    "loss": float(match[4]),
                                    "utc": utc(),
                                },
                            )
                    status = process.wait()
                if status:
                    raise ValueError(f"official trainer exited {status}; see {output / 'console.log'}")
                if recipe == "onejev":
                    # Upstream saves readout for adapters but omits it from the final merge.
                    # Copy native metadata only; no weights or kernels are changed.
                    from shutil import copy2

                    copy2(output / "native/adapter/readout.json", checkpoint / "readout.json")
                    log_rows = rows(output / "native/log.jsonl")
                    if any(not math.isfinite(r[k]) for r in log_rows for k in ("loss", "ce", "brier") if k in r):
                        raise ValueError("native training has nonfinite loss")
                    train_rows = [r for r in log_rows if "loss" in r and "tokens_per_s" in r]
                    previous, tokens = 0, 0
                    for r in train_rows:
                        tokens += round(r["tokens_per_s"] * r["sec_per_step"] * (r["step"] - previous))
                        previous = r["step"]
                    write_json(
                        output / "training_summary.json",
                        {
                            "optimizer_steps": previous,
                            "forward_tokens": tokens,
                            "token_count_source": "native tokens_per_s × sec_per_step × steps in each log window; rounded to integer",
                            "peak_device_bytes": round(max((r["peak_mem_gb"] for r in train_rows), default=0) * 2**30),
                            "loss_and_dev_metrics": "native/log.jsonl",
                        },
                    )
                else:
                    train_metrics = read_json(output / "native/training_metrics.json")
                    if train_metrics["rejected_records"] or train_metrics["truncated_records"]:
                        raise ValueError("native trainer dropped or truncated records")
                    write_json(output / "training_summary.json", train_metrics)
            write_json(
                checkpoint / "ft_exposure.json",
                {
                    "size": size,
                    "complete_budget": not bool(max_steps),
                    "rule_ids": sorted({r["rule_id"] for r in load_inputs(prepared, f"train_{size}")})
                    if not max_steps
                    else None,
                    "partitions_sha256": digest(Path(prepared) / "partitions.json"),
                    "definition": "rules in the complete one-epoch training subset; partial smoke exposure is not inferred from budget",
                },
            )
            result = {
                "identity": identity,
                "utc": utc(),
                "elapsed_seconds": time.perf_counter() - start,
                "checkpoint": checkpoint_hash(checkpoint),
                "smoke_only": bool(max_steps),
                "dev_selection": "evaluate complete dev_select with the common inference/report commands; no test selection",
                "resume": "Trainer state for Qwen; native OneJev/Kev LoRA restart only",
            }
            write_json(output / "complete.json", result)
            return result
        except BaseException as exc:
            append(
                output / "events.jsonl",
                {
                    "event": "failed",
                    "utc": utc(),
                    "elapsed_seconds": time.perf_counter() - start,
                    "exception": str(exc),
                },
            )
            raise


def preflight_native(recipe, config, files, hp):
    if recipe == "onejev":
        from train import common

        bundle = common.load_bundle(config["checkpoint"], load_model=False)
        lengths = {}
        for split, value in files.items():
            records = rows(value["path"])
            counts = [
                len(bundle.encode(common.render_example(r, bundle.template_for(r), i, seed=None).prompt))
                for i, r in enumerate(records)
            ]
            if any(n > hp["max_length"] for n in counts):
                raise ValueError(f"{split} exceeds native context; do not drop or truncate training examples")
            lengths[split] = {"n": len(counts), "max_tokens": max(counts), "tokens": sum(counts), "over_limit": 0}
        return lengths
    from kev.data import load_records, materialize
    from kev.model import load_tokenizer, fits, training_context

    tok = load_tokenizer(hp["base"], revision=hp["base_revision"])
    results = {}
    for split, value in files.items():
        records = load_records(value["path"])
        valid = [fits(materialize(r), tok, **training_context(hp["max_length"])) for r in records]
        if not all(valid):
            raise ValueError(f"{split} has over-context records; native filtering would change the subset")
        results[split] = {"n": len(valid), "over_limit": 0}
    return results


def train_qwen(config, output, files, hp, max_steps, resume):
    # Import Unsloth before Transformers, as required by its official recipe.
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from .backends import QwenBackend

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["checkpoint"],
        max_seq_length=hp["max_length"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=hp["rank"],
        lora_alpha=hp["alpha"],
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in n for n in trainable):
        raise ValueError("general LLM fine-tuning must train LoRA parameters only")
    write_json(
        output / "trainable_scope.json", {"parameters": trainable, "count": sum(trainable.values()), "lora_only": True}
    )
    renderer = object.__new__(QwenBackend)

    def convert(path):
        converted = []
        for r in rows(path):
            for qid, question in r["questions"].items():
                label = r["gold"][qid]
                label = str(label).lower() if isinstance(label, bool) else str(label)
                prompt = tokenizer.apply_chat_template(
                    renderer.messages(r["state"], question, []),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                completion = chr(65 + keys(question).index(label)) + tokenizer.eos_token
                if len(tokenizer.encode(prompt + completion, add_special_tokens=False)) > hp["max_length"]:
                    raise ValueError(f"training input exceeds context: {r['example_id']}; do not truncate/drop")
                converted.append({"prompt": prompt, "completion": completion})
        return Dataset.from_list(converted)

    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            append(
                output / "training.jsonl",
                {
                    "step": state.global_step,
                    "tokens": state.num_input_tokens_seen,
                    "utc": utc(),
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                    **(logs or {}),
                },
            )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=convert(files["train"]["path"]),
        eval_dataset=convert(files["dev_select"]["path"]),
        callbacks=[Log()],
        args=SFTConfig(
            output_dir=str(output / "trainer"),
            max_length=hp["max_length"],
            packing=False,
            completion_only_loss=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=hp["epochs"],
            max_steps=max_steps or -1,
            learning_rate=hp["learning_rate"],
            bf16=True,
            fp16=False,
            logging_steps=1,
            save_strategy="steps",
            save_steps=25,
            eval_strategy="epoch",
            seed=42,
            data_seed=42,
            report_to="none",
            optim="adamw_torch",
            include_num_input_tokens_seen=True,
        ),
    )
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_state()
    write_json(output / "trainer_metrics.json", trainer.evaluate())
    write_json(
        output / "training_summary.json",
        {
            "optimizer_steps": trainer.state.global_step,
            "forward_tokens": trainer.state.num_input_tokens_seen,
            "peak_device_bytes": torch.cuda.max_memory_allocated(),
            "loss_and_dev_metrics": "training.jsonl",
            "lora_only": True,
        },
    )
    model.save_pretrained(str(output / "adapter"))
    tokenizer.save_pretrained(str(output / "adapter"))
    model.save_pretrained_merged(str(output / "merged"), tokenizer, save_method="merged_16bit")
    return output / "merged"
