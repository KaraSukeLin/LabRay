# Inspected upstream paths and capability limits

Inspected on 2026-10-23. Full revisions and data hashes are in
[`configs/sources.json`](configs/sources.json). `bootstrap` verifies the revision,
applies the released Elephant patch, and records a source-file hash receipt.
Model weights were not downloaded or loaded during Phase 1.

| Source | Inspected implementation | Decision |
|---|---|---|
| [Official OR-ShARC](https://github.com/Yifan-Gao/open_retrieval_conversational_machine_reading/tree/309c4ea3a7cc4c7a815d7f46292d91fc9a0de581) | `data/*`, `README.md`, `MUDERN/evaluator.py` | Use official oracle-rule lookup and classification mapping; omit scenario `evidence` from input |
| [Elephant release](https://huggingface.co/baselabs/elephant-1/tree/b3f8a93462fb1051a9de031e94a7ff380e254072) | `setup.sh`, `serve.patch`, `readout.json`, model card | Use released OneJev commit + patch, local checkpoint and explicit T=1 |
| [OneJev](https://github.com/OmniJev/OneJev/tree/81ce62f1597c91e46767d4d02d6ac2e18534fe94) | Patched `qev/{engine,readout}.py`, `train/{sft,common}.py` | Native LoRA trainer and CE+Brier loss; explicitly propagate attention/readout settings; copy readout metadata after merge |
| [StartLux](https://github.com/StartLuxLabs/StartLux-Decision/tree/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01) | `startlux_decide/{pipeline,modelcfg,launch}.py`, native Transformers/vLLM backends, `v1/finetune/finetune_lora.py` | Native Instant inference; in-memory raw-temperature override; official matched vLLM BF16/FP8 launch. The v1 trainer does not establish v1.1 training compatibility |
| [Kev](https://github.com/jaredpalmer/kev/tree/2074cc44256200bb37540103fa4572901faa10f3) | `kev/{checkpoint,model,api,data,train}.py` | Preserve existing adapter/head with strict warm-start checks; `LoadOptions(temperature=1)` for raw probabilities; no custom head or loss |
| [Unsloth decision guide](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth) and [source](https://github.com/unslothai/unsloth/blob/c032ada4551eb49b607cdf06ca6f6dcd1fe0b6e0/unsloth/models/decision.py) | `FastDecisionModel`, `DecisionTrainer`, `decision_from_lm.py` | Loader handles Laya/Clef and may attach a new Clef head to a plain LM. This does not prove preservation of Elephant's slot readout, StartLux's Instant masking or Kev's pointer head. Native continued training uses official model code instead |
| [Unsloth Qwen recipe](https://unsloth.ai/docs/models/qwen3.5/fine-tune) | Text-only `FastLanguageModel` LoRA example | Optional Qwen SFT uses Unsloth/TRL and LoRA only; no QLoRA training. API compatibility is pending runtime validation |

The latest Kev Hub revision from the API differs from the abbreviated v2 revision
quoted in the current model card. The pipeline pins the actual Hub revision
`db029f08b290afd9fee4aa4bbcd9ae48602d1eb0`. Native warm-start checks must pass
before adaptation; the card's version name is not used as a compatibility claim.

Unsupported paths are deliberate errors: native Elephant/Kev int8/FP8/4bit,
Unsloth replacement of a released native decision head, StartLux v1.1 Trainer
adaptation through the v1 script, Score supervised fine-tuning, and full-state
resume in native OneJev/Kev LoRA. Available fallbacks are BF16 native inference,
Elephant → StartLux for inference → Kev for adaptation, Qwen int8/NF4 PTQ, and
new attempt directories for native trainer restarts.

All distributions are conditional on the supplied options. Native confidence
formulas can differ; reports consistently use probability of the selected label.
Native answers and confidence values remain in the raw result records.
