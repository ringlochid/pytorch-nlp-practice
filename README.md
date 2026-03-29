# PyTorch NLP Practice

CPU-first starter repo for the learning ladder.

The repo now keeps:

- scratch `RNN` / `GRU` / `LSTM` / `Transformer` paths
- a minimal Hugging Face `BERT` fine-tuning path for AG News

Designed for this VPS:

- ARM64 / aarch64
- Python 3.12
- modest RAM budget
- no GPU assumed

## Quick start

```bash
cd ~/leo/projects/pytorch-nlp-practice
source .venv/bin/activate
python src/train_scratch.py --config configs/scratch_rnn.yaml
```

BERT path:

```bash
python src/train_bert.py --config configs/bert_tiny_vps.yaml
```

## Bootstrap / reinstall the venv

```bash
cd ~/leo/projects/pytorch-nlp-practice
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Low-RAM defaults

- Scratch models: `train_subset=20000`, `test_subset=2000`, `max_length=64`
- BERT path: tiny checkpoint, `batch_size=8`, `max_length=96`, `num_workers=0`
- `num_workers=0`
- dynamic padding in the collate function
- small hidden sizes / embedding sizes

If memory gets tight, reduce in this order:

1. batch size
2. max length
3. model width
4. dataset subset size

## BERT configs

- `configs/smoke_bert.yaml`
  - fully offline smoke path for this repo/venv
  - bootstraps a tiny local BERT checkpoint under `.cache/local_bert_tiny_smoke`
  - intended to verify the Hugging Face training path, not to measure real pretrained accuracy
- `configs/bert_tiny_vps.yaml`
  - safer real fine-tuning config for this VPS
  - uses `prajjwal1/bert-tiny`
  - if the checkpoint is not cached already, the first run needs network access to download it

## Project layout

```text
pytorch-nlp-practice/
  data/
  src/
    data.py
    utils.py
    engine.py
    metrics.py
    train_scratch.py
    train_bert.py
    models/
      rnn_classifier.py
  configs/
  runs/
  README.md
```
