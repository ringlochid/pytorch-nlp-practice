# Start Here

If you're not sure where to begin, do this exact sequence.

## 1) Activate the venv

```bash
cd ~/leo/projects/pytorch-nlp-practice
source .venv/bin/activate
```

## 2) Run the smallest safe training job

```bash
python src/train_scratch.py --config configs/smoke_rnn.yaml
```

What success looks like:
- it downloads AG News if needed
- prints `device=cpu`
- prints epoch metrics
- writes a run under `runs/`

## 3) Read the code in this order

1. `src/data.py`
2. `src/models/rnn_classifier.py`
3. `src/train_scratch.py`
4. `src/engine.py`

That order shows the real pipeline:
- text -> tokens -> ids -> padded batch -> model -> logits -> loss -> metrics

## 4) Only after the smoke run works, run the real baseline

```bash
python src/train_scratch.py --config configs/scratch_rnn.yaml
```

## 5) Then climb the ladder

No big refactor needed. Just switch configs / model family:

```bash
python src/train_scratch.py --config configs/scratch_gru.yaml
python src/train_scratch.py --config configs/scratch_lstm.yaml
python src/train_scratch.py --config configs/scratch_transformer.yaml
python src/train_bert.py --config configs/bert_tiny.yaml
```

## What the `notebooks/` folder is for

Optional scratch space only.

Use it for:
- inspecting one batch
- checking tensor shapes
- plotting metrics from `runs/metrics.json`
- loading a saved checkpoint and testing a few texts

Do **not** put your main training logic there. Keep the real code in `src/`.

## If you want to use Jupyter later

Install notebook support into this venv:

```bash
source .venv/bin/activate
pip install jupyter ipykernel
```

In a notebook, add this first so imports work cleanly with the current script-first layout:

```python
import sys
sys.path.append("src")
```

Then you can do things like:

```python
from data import make_scratch_dataloaders
from models.rnn_classifier import RNNClassifier
```

## First learning goal

Don't chase accuracy first.

First goal:
- understand one full pass from raw text to loss
- get one run to finish cleanly
- see where batch size, max length, and model size live in config

Once that feels normal, the rest gets much easier.
