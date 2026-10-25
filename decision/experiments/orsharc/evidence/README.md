# Phase 1 evidence

This directory stores small, portable readiness evidence. Generated data and
downloaded upstream code remain ignored under `../artifacts/`.

- `data_audit.json`: official hashes, split/label counts, Seen/Unseen counts,
  overlap and known follow-up punctuation anomalies.
- `readiness.json`: exact CPU checks and their outcomes, produced after the final
  source/configuration audit. This does not contain model results.
- [dashboard_fixture.html](dashboard_fixture.html): a four-row interface example
  with artificial probabilities and an explicit fixture label.

No weights were loaded. No GPU smoke test, optimizer step, model evaluation or
research ablation was run. The eight-row report check uses fixed artificial
probabilities on training inputs only, in a temporary directory that is removed.
