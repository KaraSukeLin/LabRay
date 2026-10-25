# LabRay

Keep each experiment under `quant/` or `decision/`. Run its commands from that project root.

Before changing saved evidence or running an experiment, read that project's `migration/REPRODUCE.md`. Keep saved research data unchanged and write new results to a new output directory. Model weights belong in `/root/model/`.

Run only targeted CPU checks for changed files. Full test suites and GPU runs require an explicit user request. When moving files or tools, update the affected project guide and its migration checksum list.
