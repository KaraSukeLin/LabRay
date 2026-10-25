"""Experiment B command line. Accelerator work is always an explicit command."""

import argparse
import json

from . import analysis, data, matrix, setup, training
from .common import read_json
from .runner import ROOT, infer, model_config, validate_config


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--raw", default=str(ROOT / "artifacts/raw"))
    p.add_argument("--out", default=str(ROOT / "artifacts/prepared"))
    p.add_argument("--download", action="store_true")
    p = sub.add_parser("validate")
    p.add_argument("--prepared")
    p = sub.add_parser("bootstrap", help="download pinned source only; no packages or weights")
    p.add_argument(
        "--sources", nargs="+", choices=["onejev", "kev", "startlux", "unsloth"], default=["onejev", "kev", "startlux"]
    )
    p = sub.add_parser("check", help="small deterministic CPU fixtures; no model loads")
    p.add_argument("--prepared")
    for name in ("train", "infer", "export"):
        p = sub.add_parser(name)
        p.add_argument("--prepared", default=str(ROOT / "artifacts/prepared"))
        p.add_argument("--out", required=True)
        p.add_argument("--task", choices=["benchmark", "sufficiency"], default="benchmark")
        p.add_argument(
            "--strategy",
            choices=["single_choice", "hierarchical", "parallel_noul", "noul", "choice2", "score5"],
            default="single_choice",
        )
        if name == "export":
            p.add_argument("--recipe", choices=["onejev", "kev", "qwen"], default="onejev")
            p.add_argument("--size", type=int, choices=[256, 1024, 4096], default=256)
            continue
        p.add_argument("--model", choices=["elephant", "startlux", "kev", "qwen"], default="elephant")
        p.add_argument("--model-config", help="full model JSON; takes priority over profile")
        p.add_argument("--checkpoint")
        p.add_argument("--quantization", choices=["bf16", "bnb8", "fp8", "bnb4"])
        p.add_argument(
            "--engine",
            choices=["transformers", "vllm"],
            help="StartLux native engine; B5 must use vLLM for BF16 and FP8",
        )
        if name == "train":
            p.add_argument("--size", type=int, choices=[256, 1024, 4096], default=256)
            p.add_argument("--execute", action="store_true")
            p.add_argument(
                "--max-steps", type=int, default=0, help="Phase 2 smoke limit; 0 for the registered full epoch"
            )
            p.add_argument("--resume", action="store_true")
        else:
            p.add_argument(
                "--partition", choices=["train_256", "dev_select", "dev_calibrate", "test"], default="dev_select"
            )
            p.add_argument("--shots", type=int, choices=[0, 4], default=0)
            p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("matrix", help="generate executable Phase 2 script; does not run it")
    p.add_argument("--name", choices=["b1", "b2", "b3", "b4", "b5"], required=True)
    p.add_argument("--model", choices=["elephant", "startlux", "kev", "qwen"], default="elephant")
    p.add_argument("--prepared", default=str(ROOT / "artifacts/prepared"))
    p.add_argument("--out", default=str(ROOT / "artifacts/plans"))
    p.add_argument("--checkpoint")
    p.add_argument("--include-optional", action="store_true")
    p.add_argument("--run-root", help="new inference output root when code/recipe/environment changes")
    p = sub.add_parser("calibrate")
    p.add_argument("--select-run", required=True)
    p.add_argument("--calibration-run", required=True)
    p.add_argument("--prepared", default=str(ROOT / "artifacts/prepared"))
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    p.add_argument("--prepared", default=str(ROOT / "artifacts/prepared"))
    p.add_argument("--out", required=True)
    p.add_argument("--calibration")
    p = sub.add_parser("compare-quantization")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("smoke-verify", help="offline checks of Phase 2 train-only smoke outputs")
    p.add_argument("--original", required=True)
    p.add_argument("--reload", required=True)
    p.add_argument("--training")
    p.add_argument("--adapted")
    p.add_argument("--admission-evidence", help="Phase 3 artifact index; permits measured B1–B4 admission")
    p.add_argument("--out", required=True)
    p = sub.add_parser("environment-verify", help="verify the exact Elephant H100 runtime")
    p.add_argument("--out", required=True)
    for name in ("phase3-measure", "metadata-audit"):
        p = sub.add_parser(name, help="targeted Phase 3 admission evidence; no final test inference")
        p.add_argument("--prepared", default=str(ROOT / "artifacts/prepared"))
        p.add_argument("--out", required=True)
    p = sub.add_parser("phase3-serving", help="profile production inference on the recorded train/dev long-input panel")
    p.add_argument("--measurement", required=True)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if hasattr(args, "limit") and args.limit < 0 or hasattr(args, "max_steps") and args.max_steps < 0:
        raise ValueError("limits must be nonnegative")
    if args.command == "phase3-serving":
        from .phase3 import serving

        result = serving(args.measurement)
    elif args.command in ("phase3-measure", "metadata-audit"):
        from .phase3 import measure, metadata_audit

        result = (measure if args.command == "phase3-measure" else metadata_audit)(args.prepared, args.out)
    elif args.command == "environment-verify":
        from .gates import environment_verify

        result = environment_verify(args.out)
    elif args.command == "prepare":
        result = data.prepare(args.raw, args.out, args.download)
    elif args.command == "bootstrap":
        result = setup.bootstrap(args.sources)
    elif args.command in ("validate", "check"):
        profiles = read_json(ROOT / "configs/models.json")
        for config in profiles.values():
            validate_config(config)
        result = {"models": list(profiles), "configuration": "valid"}
        if args.prepared:
            result["dataset"] = data.validate_prepared(args.prepared)
        if args.command == "check":
            from .checks import check

            result["fixtures"] = check(args.prepared)
    elif args.command == "export":
        result = training.export(args.prepared, args.out, args.size, args.task, args.strategy, args.recipe)
    elif args.command in ("train", "infer"):
        config = (
            read_json(args.model_config)
            if args.model_config
            else model_config(args.model, args.checkpoint, args.quantization)
        )
        if args.engine:
            config["engine"] = args.engine
        if args.command == "train":
            result = training.train(
                config,
                args.prepared,
                args.out,
                args.size,
                args.task,
                args.strategy,
                args.execute,
                args.max_steps,
                args.resume,
            )
        else:
            result = infer(
                config, args.prepared, args.out, args.partition, args.task, args.strategy, args.shots, args.limit
            )
    elif args.command == "matrix":
        result = matrix.generate(
            args.name, args.model, args.prepared, args.out, args.checkpoint, args.include_optional, args.run_root
        )
        result = {k: v for k, v in result.items() if k != "commands"} | {"command_count": len(result["commands"])}
    elif args.command == "calibrate":
        result = analysis.calibrate(args.select_run, args.calibration_run, args.prepared, args.out)
    elif args.command == "report":
        result = analysis.report(args.run, args.prepared, args.out, args.calibration)
        result = {"run_identity": result["run_identity"], "out": args.out, "n": result["metrics"]["raw"]["All"]["n"]}
    elif args.command == "compare-quantization":
        result = analysis.compare_quantization(args.runs, args.out)
    else:
        from .gates import smoke_verify

        result = smoke_verify(
            args.original, args.reload, args.out, args.training, args.adapted, args.admission_evidence
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
