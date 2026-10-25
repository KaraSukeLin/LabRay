"""Emit independent executable matrices, using the same inference/report CLI."""

import shlex
from pathlib import Path

from .common import read_json, signature, write_json
from .runner import ROOT, model_config, validate_config


def generate(name, model, prepared, out, checkpoint=None, include_optional=False, run_root=None):
    matrix = read_json(ROOT / "configs/matrices.json")[name]
    config = model_config(model, checkpoint)
    validate_config(config)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    prepared = str(Path(prepared).resolve())
    base = ["python", str(ROOT / "run.py")]
    commands, blocked = [], []
    can_train = config["backend"] in ("qev", "kev", "qwen") and not config.get("training_blocker")
    # Passing --checkpoint evaluates a supplied checkpoint across the matrix;
    # original-vs-adapted matrices use registered training paths by default.
    conditions = (
        ["provided-" + signature(str(Path(checkpoint).resolve()))[:10]]
        if checkpoint
        else matrix.get("conditions", ["original"])
    )
    if name in ("b1", "b4") and not checkpoint:
        for size in matrix["sizes"]:
            training_out = ROOT / "artifacts/training" / f"{model}-{size}"
            if can_train and (config["backend"] != "qwen" or include_optional):
                commands.append(
                    base
                    + [
                        "train",
                        "--model",
                        model,
                        "--prepared",
                        prepared,
                        "--out",
                        str(training_out),
                        "--size",
                        str(size),
                        "--execute",
                    ]
                )
            else:
                blocked.append(
                    {
                        "condition": f"ft_{size}",
                        "reason": config.get("training_blocker")
                        or (
                            "Qwen LoRA optional: add --include-optional"
                            if can_train
                            else "StartLux v1.1 trainer compatibility is unverified; use Elephant or Kev"
                        ),
                    }
                )
    for condition in conditions:
        override = checkpoint
        if condition.startswith("ft_"):
            if not can_train or config["backend"] == "qwen" and not include_optional:
                continue
            suffix = (
                "native" if config["backend"] == "kev" else "native/merged" if config["backend"] == "qev" else "merged"
            )
            override = str(ROOT / "artifacts/training" / f"{model}-{condition[3:]}" / suffix)
        strategies = matrix["strategies"]
        quantizations = matrix.get("quantizations", ["bf16"])
        for strategy in strategies:
            for quant in quantizations:
                if quant == "bnb4" and not include_optional:
                    continue
                try:
                    validate_config(
                        {
                            **config,
                            "quantization": quant,
                            **({"engine": "vllm"} if name == "b5" and model == "startlux" else {}),
                        }
                    )
                except ValueError as exc:
                    blocked.append({"condition": condition, "quantization": quant, "reason": str(exc)})
                    continue
                shots_options = (
                    [0, 4]
                    if config["backend"] == "qwen"
                    and (condition == "original" or condition.startswith("provided-"))
                    and name in ("b1", "b4")
                    else [0]
                )
                for shots in shots_options:
                    run_id = f"{model}-{condition}-{strategy}-{quant}-{shots}shot"
                    if name == "b5" and model == "startlux":
                        run_id += "-vllm"
                    run = (Path(run_root).resolve() if run_root else ROOT / "artifacts/runs") / run_id
                    for partition in ("dev_select", "dev_calibrate", "test"):
                        args = base + [
                            "infer",
                            "--model",
                            model,
                            "--prepared",
                            prepared,
                            "--out",
                            str(run / partition),
                            "--partition",
                            partition,
                            "--task",
                            matrix["task"],
                            "--strategy",
                            strategy,
                            "--quantization",
                            quant,
                            "--shots",
                            str(shots),
                        ]
                        if override:
                            args += ["--checkpoint", str(override)]
                        if name == "b5" and model == "startlux":
                            args += ["--engine", "vllm"]
                        # Freeze dev selection and temperature before final test inference.
                        if partition == "test":
                            commands.append(
                                base
                                + [
                                    "calibrate",
                                    "--prepared",
                                    prepared,
                                    "--select-run",
                                    str(run / "dev_select"),
                                    "--calibration-run",
                                    str(run / "dev_calibrate"),
                                    "--out",
                                    str(run / "calibration.json"),
                                ]
                            )
                        commands.append(args)
                    for partition in ("dev_select", "dev_calibrate", "test"):
                        commands.append(
                            base
                            + [
                                "report",
                                "--prepared",
                                prepared,
                                "--run",
                                str(run / partition),
                                "--calibration",
                                str(run / "calibration.json"),
                                "--out",
                                str(run / "reports" / partition),
                            ]
                        )
    script = output / f"{name}-{model}.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nB_PYTHON=${B_PYTHON:-python}\n"
        + "\n".join('"$B_PYTHON" ' + shlex.join(c[1:]) for c in commands)
        + "\n"
    )
    script.chmod(0o755)
    manifest = {
        "matrix": name,
        "model": model,
        "checkpoint_override": checkpoint,
        "optional": include_optional,
        "commands": commands,
        "blocked": blocked,
        "script": str(script),
        "phase": "2; generate does not execute",
    }
    write_json(output / f"{name}-{model}.json", manifest)
    return manifest
