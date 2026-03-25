# PyTorch NLP Practice

CPU-first starter repo for the learning ladder:

- RNN
- GRU
- LSTM
- Transformer encoder
- BERT tiny fine-tuning

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

Then swap configs:

```bash
python src/train_scratch.py --config configs/scratch_gru.yaml
python src/train_scratch.py --config configs/scratch_lstm.yaml
python src/train_scratch.py --config configs/scratch_transformer.yaml
python src/train_bert.py --config configs/bert_tiny.yaml
```

## Low-RAM defaults

- Scratch models: `train_subset=20000`, `test_subset=2000`, `max_length=64`
- `num_workers=0`
- dynamic padding in the collate function
- small hidden sizes / embedding sizes
- `bert-tiny` before DistilBERT

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
    train_bert.py
    models/
      rnn_classifier.py
      gru_classifier.py
      lstm_classifier.py
      transformer_encoder_classifier.py
      bert_classifier.py
  configs/
  runs/
  README.md
```
