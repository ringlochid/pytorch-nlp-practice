# PyTorch NLP Practice

CPU-first starter repo for the learning ladder.

Right now the repo keeps only the **RNN baseline**.
You can add GRU / LSTM / Transformer / BERT yourself later.

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

## Bootstrap / reinstall the venv

```bash
cd ~/leo/projects/pytorch-nlp-practice
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Only the RNN path is wired in right now.
When you want help adding the next model, ask and we can scaffold just that one.

## Low-RAM defaults

- Scratch models: `train_subset=20000`, `test_subset=2000`, `max_length=64`
- `num_workers=0`
- dynamic padding in the collate function
- small hidden sizes / embedding sizes

If memory gets tight, reduce in this order:

1. batch size
2. max length
3. model width
4. dataset subset size

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
    models/
      rnn_classifier.py
  configs/
  runs/
  README.md
```
