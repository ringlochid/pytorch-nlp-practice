# scripts

Small inspect helpers for understanding the repo without touching the training code.

## 1) Inspect dataset schema and samples

```bash
source .venv/bin/activate
python scripts/inspect_dataset.py --config configs/smoke_rnn.yaml --split train --samples 3
```

Shows:
- column names
- dataset features/schema
- label names
- raw text samples
- tokenized preview

## 2) Inspect vocab building

```bash
python scripts/inspect_vocab.py --config configs/smoke_rnn.yaml
```

Shows:
- vocab size
- pad/unk ids
- first vocab entries
- one text -> tokens -> encoded ids example

## 3) Inspect one collated batch

```bash
python scripts/inspect_batch.py --config configs/smoke_rnn.yaml --split train
```

Shows:
- `input_ids`, `lengths`, `labels` shapes
- decoded active tokens for a few samples
- where padding begins in the batch rows

## 4) Inspect model parameter + tensor shapes

```bash
python scripts/inspect_model_shapes.py --config configs/smoke_rnn.yaml
```

Shows:
- parameter names and tensor shapes
- forward-pass shapes from `input_ids` to `logits`
- first-sample probabilities/prediction
- one batch loss value

These are inspect/debug scripts only. Keep the real training path in `src/`.
