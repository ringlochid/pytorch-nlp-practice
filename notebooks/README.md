# notebooks

This folder is optional scratch space for Jupyter / VS Code notebooks.

Good uses:
- inspect dataset examples
- inspect one batch from the DataLoader
- plot metrics from `runs/metrics.json`
- load a saved checkpoint and try a few custom texts

Bad uses:
- storing the main training loop
- drifting away from the `src/` code path
- duplicating model code that should live in `src/models/`

This repo is script-first. Train from `src/train_scratch.py` / `src/train_bert.py` first, then use notebooks for inspection and experiments.
