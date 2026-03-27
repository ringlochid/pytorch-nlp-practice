#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import make_scratch_dataloaders  # type: ignore
from train_scratch import build_model  # type: ignore
from utils import count_parameters, get_device, load_yaml, seed_everything  # type: ignore


def resolve_config(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect parameter shapes and the forward-pass tensor shapes.")
    parser.add_argument("--config", default="configs/smoke_rnn.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    args = parser.parse_args()

    cfg = load_yaml(resolve_config(args.config))
    seed_everything(cfg["seed"])
    device = get_device(cfg["system"].get("device", "cpu"))
    loaders = make_scratch_dataloaders(cfg["data"], seed=cfg["seed"])
    vocab = loaders["vocab"]
    label_names = loaders["label_names"]

    model = build_model(
        cfg["model"],
        vocab_size=len(vocab),
        num_classes=len(label_names),
        pad_id=vocab.pad_id,
    ).to(device)
    model.eval()

    source_loader = loaders[f"{args.split}_loader"]
    inspect_loader = DataLoader(
        source_loader.dataset,
        batch_size=source_loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=source_loader.collate_fn,
    )
    batch = next(iter(inspect_loader))
    input_ids = batch["input_ids"].to(device)
    lengths = batch["lengths"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        embedded = model.embedding(input_ids)
        encoded, hidden = model.encoder(embedded)
        mask = (input_ids != model.pad_id).unsqueeze(-1)
        masked = encoded * mask
        pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        dropped = model.dropout(pooled)
        logits = model.classifier(dropped)
        probs = torch.softmax(logits, dim=-1)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

    print(f"config: {args.config}")
    print(f"device: {device}")
    print(f"params: {count_parameters(model):,}")
    print()
    print("parameter shapes:")
    for name, param in model.named_parameters():
        print(f"  {name:<28} {tuple(param.shape)}")
    print()
    print("forward-pass shapes:")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  lengths:   {tuple(lengths.shape)}")
    print(f"  labels:    {tuple(labels.shape)}")
    print(f"  embedded:  {tuple(embedded.shape)}")
    print(f"  encoded:   {tuple(encoded.shape)}")
    print(f"  hidden:    {tuple(hidden.shape)}")
    print(f"  mask:      {tuple(mask.shape)}")
    print(f"  pooled:    {tuple(pooled.shape)}")
    print(f"  logits:    {tuple(logits.shape)}")
    print()
    print(f"first sample probs: {probs[0].cpu().tolist()}")
    print(f"first sample pred:  {int(logits[0].argmax(dim=-1).item())}")
    print(f"first sample gold:  {int(labels[0].item())}")
    print(f"batch loss:         {loss.item():.6f}")


if __name__ == "__main__":
    main()
