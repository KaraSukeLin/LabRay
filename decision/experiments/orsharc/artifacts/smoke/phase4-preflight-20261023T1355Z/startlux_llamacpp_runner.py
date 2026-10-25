#!/usr/bin/env python3
"""Use the frozen OR-ShARC runner with the pinned StartLux native HTTP server."""

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
ORSHARC = HERE.parents[2]
sys.path.insert(0, str(ORSHARC / "src"))

from b_orsharc import backends, runner  # noqa: E402


def model_config_path():
    try:
        return Path(sys.argv[sys.argv.index("--model-config") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise SystemExit("--model-config is required for the B5 native adapter") from exc


CONFIG = json.loads(model_config_path().read_text())
ADAPTER_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
if CONFIG.get("adapter_sha256") != ADAPTER_HASH:
    raise SystemExit("B5 adapter SHA256 does not match the model configuration")


class StartLuxLlamaCppHTTP:
    def __init__(self, config, demos):
        if demos:
            raise ValueError("B5 StartLux uses the native zero-shot decision path")
        self.endpoint = config["endpoint"]
        self.request_number = 0
        self.config = config
        self.provenance = {
            "runtime": "StartLux native /v1/systemone over pinned llama.cpp CUDA server",
            "endpoint": self.endpoint,
            "model_revision": config["model_revision"],
            "gguf_file": config["model_file"],
            "quantization": config["quantization"],
            "llama_cpp_version": config["llama_cpp_version"],
            "runtime_image_digest": config["runtime_image_digest"],
            "temperature_by_type": config["native_temperature_by_type"],
            "probability_source": config["probability_source"],
            "think_mode": "never",
            "none_check": False,
            "allow_none": False,
            "batching": "one typed systemone request per example; all parallel Noul questions in one request",
            "cache_prompt": False,
            "adapter_sha256": ADAPTER_HASH,
        }
        health = self._get(config["health_url"])
        if health.get("status") != "ok" or health.get("kind") != "instant":
            raise RuntimeError(f"StartLux health check failed: {health}")
        if health.get("model") != config["health_model_name"]:
            raise RuntimeError(f"unexpected StartLux model identity: {health.get('model')}")
        self.provenance["health"] = {
            "status": health["status"],
            "model": health["model"],
            "engine": health.get("engine"),
            "context_length": health.get("context_length"),
            "prompt_attention": health.get("prompt_attention"),
        }

    @staticmethod
    def _get(url):
        with urllib.request.urlopen(url, timeout=20) as response:
            if response.status != 200:
                raise RuntimeError(f"health returned HTTP {response.status}")
            return json.loads(response.read())

    def predict(self, state, questions):
        body = {
            "state": state,
            "questions": questions,
            "allow_none": False,
            "none_check": False,
            "think_mode": "never",
            "return_details": True,
        }
        self.request_number += 1
        request_id = f"orsharc-b5-{self.request_number:05d}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Request-ID": request_id},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = response.status
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"StartLux request {request_id} returned HTTP {exc.code}: {detail}") from exc
        elapsed = time.perf_counter() - started
        if status != 200:
            raise RuntimeError(f"StartLux request {request_id} returned HTTP {status}")
        native = result["answers"]
        return {
            "answers": backends.native_answers(native, questions),
            "native": native,
            "usage": result.get("usage"),
            "timing_ms": result.get("timing_ms"),
            "stats": {"request_id": request_id, "http_seconds": elapsed},
        }

    def close(self):
        # The StartLux and llama.cpp services are managed by the outer run script.
        return None


def load_backend(config, demos=()):
    if config.get("backend") == "startlux" and config.get("engine") == "llamacpp":
        if config.get("quantization") != "q4_k_m":
            raise ValueError("the B5 llama.cpp adapter requires the measured StartLux Q4_K_M checkpoint")
        return StartLuxLlamaCppHTTP(config, demos)
    return ORIGINAL_LOAD(config, demos)


def validate(config):
    if config.get("backend") == "startlux" and config.get("engine") == "llamacpp":
        if config.get("quantization") != "q4_k_m" or not Path(config["checkpoint"]).is_dir():
            raise ValueError("B5 needs a local StartLux Q4_K_M snapshot and its pinned llama.cpp runtime")
        if config.get("adapter_sha256") != ADAPTER_HASH:
            raise ValueError("B5 adapter hash changed after the model config was frozen")
        return config
    return ORIGINAL_VALIDATE(config)


def provenance_signature(value):
    if isinstance(value, dict) and value.get("model", {}).get("engine") == "llamacpp":
        value["probability_source"] = CONFIG["probability_source"]
    return ORIGINAL_SIGNATURE(value)


ORIGINAL_LOAD = backends.load
ORIGINAL_VALIDATE = runner.validate_config
ORIGINAL_SIGNATURE = runner.signature
backends.load = load_backend
runner.validate_config = validate
runner.signature = provenance_signature

from b_orsharc.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
