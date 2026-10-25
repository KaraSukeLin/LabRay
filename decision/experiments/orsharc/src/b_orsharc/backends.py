"""Thin adapters around inspected upstream code. Imports occur only on load."""

import math

from .tasks import keys


def normalized(values, expected):
    if set(values) != set(expected):
        raise ValueError(f"probability keys differ: got {list(values)}, expected {expected}")
    p = {k: float(values[k]) for k in expected}
    if any(not math.isfinite(v) or v < 0 or v > 1 for v in p.values()) or abs(sum(p.values()) - 1) > 0.02:
        raise ValueError(f"invalid probability distribution: {p}")
    total = sum(p.values())
    return {k: v / total for k, v in p.items()}


def native_answers(response, questions):
    result = {}
    for name, question in questions.items():
        a = response[name]
        if question["type"] == "noul":
            p = float(a["noul"])
            values = {"false": 1 - p, "true": p}
        else:
            values = a["probabilities"]
        result[name] = normalized(values, keys(question))
    return result


class QevBackend:
    def __init__(self, config, demos):
        import dataclasses
        from qev.engine import DecisionEngine
        from qev.calibrate import Calibration
        from qev.schema import SystemOneRequest

        if config["quantization"] != "bf16":
            raise ValueError("Elephant qev quantization is unsupported by this inspected runtime")
        self.request = SystemOneRequest
        # Explicit identity calibration overrides any bundled temperature.
        self.engine = DecisionEngine(
            config["checkpoint"],
            device=config.get("device", "cuda:0"),
            calibration=Calibration(),
            max_branch_tokens=config.get("max_length", 16384),
        )
        self.provenance = {"runtime": dataclasses.asdict(self.engine.info), "temperature": 1.0}

    def predict(self, state, questions):
        import torch

        torch.cuda.synchronize(self.engine.device)
        result, stats = self.engine.decide(self.request(state=state, questions=questions), debug=True)
        torch.cuda.synchronize(self.engine.device)
        native = {k: v.model_dump() for k, v in result.answers.items()}
        usage = getattr(result, "usage", None)
        return {
            "answers": native_answers(native, questions),
            "native": native,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
            "stats": stats,
        }


class KevBackend:
    def __init__(self, config, demos):
        import torch
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.api import SystemOneRequest, to_record, to_answers

        if config["quantization"] != "bf16":
            raise ValueError("Kev native CUDA quantization is unsupported by this adapter")
        self.request, self.to_record, self.to_answers = SystemOneRequest, to_record, to_answers
        self.tok, self.model = Checkpoint(config["checkpoint"]).load(
            config.get("device", "cuda"),
            LoadOptions(dtype=torch.bfloat16, temperature=1.0, cuda_graphs=False, fused=False),
        )
        self.model.eval()
        self.max_length = config.get("max_length", 16384)
        self.provenance = {"runtime": "kev.checkpoint.Checkpoint", "temperature": 1.0, "cache": "off"}

    def predict(self, state, questions):
        import torch
        from kev.model import admit, rows_of

        record, meta = self.to_record(self.request(state=state, questions=questions))
        encoded = admit(self.model, self.tok, record, truncate=False)
        prefix, _, branches = rows_of(encoded)
        if any(len(prefix) + len(b["ids"]) > self.max_length for b in branches):
            raise ValueError("Kev input exceeds the common context limit; no truncation")
        torch.cuda.synchronize(self.model.device)
        probs = self.model.probs(encoded)
        torch.cuda.synchronize(self.model.device)
        full = {
            m["id"]: normalized({k: float(v) for k, v in zip(m["keys"], p)}, m["keys"]) for m, p in zip(meta, probs)
        }
        native = self.to_answers(probs, meta)
        native = {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in native.items()}
        return {"answers": full, "native": native, "usage": {"input_tokens": len(encoded["ids"]), "output_tokens": 0}}


class StartLuxBackend:
    def __init__(self, config, demos):
        import asyncio
        import dataclasses
        from startlux_decide.modelcfg import load_model_folder, load_tokenizer
        from startlux_decide.pipeline import Decider, default_controls
        from startlux_decide.backends.transformers import TransformersBackend
        import atexit

        self.servers = None
        self.loop = asyncio.new_event_loop()
        folder = load_model_folder(config["checkpoint"])
        if folder.kind != "instant":
            raise ValueError("Experiment B requires the v1.1 Instant checkpoint")
        folder = dataclasses.replace(folder, temperature_by_type={t: 1.0 for t in ("noul", "choice", "score")})
        tokenizer = load_tokenizer(config["checkpoint"])
        if config.get("engine", "transformers") == "vllm":
            from startlux_decide.backends.vllm import VLLMBackend
            from startlux_decide.launch import VLLMServers

            self.servers = VLLMServers(
                config["checkpoint"],
                config.get("gpus", [0]),
                base_port=config.get("port", 18001),
                log_dir=config["runtime_log_dir"],
                quantization="fp8" if config["quantization"] == "fp8" else None,
                max_model_len=config.get("max_length", 16384),
                mtp=False,
                max_images=0,
            )
            self.servers.up(config.get("startup_timeout", 1800))
            self.backend = VLLMBackend(self.servers.urls, tokenizer, waves=False, images=False)
        else:
            if config["quantization"] != "bf16":
                raise ValueError("FP8 requires the official vLLM engine for both BF16 and FP8")
            self.backend = TransformersBackend(
                config["checkpoint"], tokenizer, folder.bidirectional, config.get("device", "cuda:0"), "bfloat16"
            )
        atexit.register(self.close)
        self.loop.run_until_complete(self.backend.start())
        controls = default_controls(folder, think_mode="never", none_check=False, allow_none=False)
        self.decider = Decider(folder, self.backend, controls, context_length=config.get("max_length", 16384))
        self.provenance = {
            "runtime": self.backend.describe(),
            "temperature_by_type": dict(folder.temperature_by_type),
            "none_check": False,
            "think_mode": "never",
            "server_settings": self.servers.settings() if self.servers else None,
            "launch_plan": [
                {"argv": cmd, "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES")}
                for cmd, env in self.servers.plan()
            ]
            if self.servers
            else None,
        }

    def predict(self, state, questions):
        result = self.loop.run_until_complete(self.decider.decide(state, questions))
        return {
            "answers": native_answers(result.answers, questions),
            "native": result.answers,
            "usage": result.usage,
            "timing_ms": result.timing_ms,
        }

    def close(self):
        if not self.loop.is_closed():
            self.loop.run_until_complete(self.backend.close())
            self.loop.close()
        if self.servers:
            self.servers.stop(grace=10)


class QwenBackend:
    """Official Transformers forward pass, restricted next-letter likelihood."""

    def __init__(self, config, demos):
        import torch
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

        self.torch, self.config, self.demos = torch, config, demos
        self.tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"])
        hf_config = AutoConfig.from_pretrained(config["checkpoint"])
        kwargs = {
            "config": hf_config.get_text_config(),
            "dtype": torch.bfloat16,
            "device_map": {"": config.get("device", "cuda:0")},
            "trust_remote_code": False,
        }
        if config["quantization"] == "bnb8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif config["quantization"] == "bnb4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif config["quantization"] != "bf16":
            raise ValueError("Qwen supports bf16/bnb8/bnb4 candidates; FP8 is not implemented")
        self.model = AutoModelForCausalLM.from_pretrained(config["checkpoint"], **kwargs).eval()
        self.provenance = {
            "temperature": 1.0,
            "readout": "next-token letters, conditional on valid choices",
            "thinking": False,
            "quantization_config": kwargs["quantization_config"].to_dict() if "quantization_config" in kwargs else None,
        }
        self.last_tokens = 0

    def messages(self, state, question, examples):
        def prompt(s, q):
            options = keys(q)
            descriptions = q.get("criteria") or {"false": "No", "true": "Yes"}
            if isinstance(descriptions, list):
                descriptions = {str(i): v for i, v in enumerate(descriptions)}
            return (
                s
                + "\n\n"
                + q["instructions"]
                + "\n"
                + "\n".join(f"{chr(65 + i)}. {k}: {descriptions[k]}" for i, k in enumerate(options))
                + "\nAnswer with one option letter."
            )

        messages = [
            {
                "role": "system",
                "content": "Use only the supplied rule and user information. Select the best listed option.",
            }
        ]
        for demo, qid in examples:
            q = demo["questions"][qid]
            y = demo["gold"][qid]
            y = str(y).lower() if isinstance(y, bool) else str(y)
            messages += [
                {"role": "user", "content": prompt(demo["state"], q)},
                {"role": "assistant", "content": chr(65 + keys(q).index(y))},
            ]
        return messages + [{"role": "user", "content": prompt(state, question)}]

    def predict(self, state, questions):
        result, native, total_tokens = {}, {}, 0
        for qid, question in questions.items():
            examples = [(demo, qid) for demo in self.demos if qid in demo["gold"]]
            # Require exactly four demonstrations in each few-shot question.
            if self.demos and len(examples) != 4:
                raise ValueError(f"four usable demonstrations required for {qid}")
            text = self.tokenizer.apply_chat_template(
                self.messages(state, question, examples),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            batch = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            n = batch["input_ids"].shape[1]
            if n > self.config.get("max_length", 16384):
                raise ValueError(f"context exceeds fixed limit: {n}; no truncation allowed")
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            ids = [self.tokenizer.encode(chr(65 + i), add_special_tokens=False) for i in range(len(keys(question)))]
            if any(len(v) != 1 for v in ids) or len({v[0] for v in ids}) != len(ids):
                raise ValueError("option letters must have distinct single tokens")
            with self.torch.inference_mode():
                logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                selected = logits[[v[0] for v in ids]]
                p = selected.softmax(-1).cpu().tolist()
                mass = float((selected.logsumexp(0) - logits.logsumexp(0)).exp())
            result[qid] = dict(zip(keys(question), p))
            native[qid] = {
                "option_logits": selected.cpu().tolist(),
                "option_token_ids": [v[0] for v in ids],
                "valid_option_mass": mass,
                "input_tokens": n,
            }
            total_tokens += n
        self.torch.cuda.synchronize(self.model.device)
        return {"answers": result, "native": native, "usage": {"input_tokens": total_tokens, "output_tokens": 0}}


def load(config, demos=()):
    classes = {"qev": QevBackend, "kev": KevBackend, "startlux": StartLuxBackend, "qwen": QwenBackend}
    if config["backend"] not in classes:
        raise ValueError(f"unsupported backend: {config['backend']}")
    return classes[config["backend"]](config, demos)
