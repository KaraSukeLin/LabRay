"""Decision Index engine for this checkpoint, on OneJev's qev runtime @81ce62f + serve.patch (setup.sh installs both).

The checkpoint's readout.json (attention, readout) and optional calibration.json are read at load, so the model is
served exactly as it was trained: one shared-prefix prefill per request and one batched forward over the question
suffixes. No prompt tuning, truncation or option pruning; requests over the trained length limits are refused.

    python -m decision_index pipeline --engine engine:DecisionModel --option model=<repo or path> --out runs/<name>
"""
from decision_index.engines import Engine, Unsupported
from qev.engine import DecisionEngine
from qev.schema import SystemOneRequest

FIELDS = {"choice": {"type", "choice", "probabilities"}, "noul": {"type", "noul"},
          "score": {"type", "score", "probabilities"}}


class DecisionModel(Engine):
    name = "decision-model"
    latency = "In-process qev decide() wall time including prompt rendering; excludes model loading."

    def __init__(self, model, device="cuda:0", **options):
        super().__init__(model=model, device=device, **options)
        self.engine = DecisionEngine(model, device=device)
        info = self.engine.info
        self.provenance = {"kind": "full fine-tune", "runtime": "OneJev qev @81ce62f + serve.patch", "model": model,
                           "fork_mode": info.fork_mode, "attention": info.extra["attention"],
                           "readout": info.extra["readout"], "calibration": info.extra["calibration"]}

    def __call__(self, state, questions):
        for q in questions.values():
            if q.get("type") not in FIELDS:
                raise Unsupported(f"question type {q.get('type')}")
        request = SystemOneRequest(state=state, questions=questions)
        try:
            resp, _ = self.engine.decide(request)
        except ValueError as e:
            if "limit is" in str(e):
                raise Unsupported(str(e)) from e
            raise
        except RuntimeError as e:
            # The batched prefix fork overflows 32-bit indexing on the largest multi-question requests; the
            # sequential branches are OneJev's exact fallback for the same computation.
            if "32BitIndexMath" not in str(e):
                raise
            mode, self.engine.info.fork_mode = self.engine.info.fork_mode, "sequential"
            try:
                resp, _ = self.engine.decide(request)
            finally:
                self.engine.info.fork_mode = mode
        answers = {k: a.model_dump(include=FIELDS[a.type]) for k, a in resp.answers.items()}
        return {"model": self.options["model"], "answers": answers}, None

    def synchronize(self):
        import torch
        torch.cuda.synchronize(self.engine.device)
