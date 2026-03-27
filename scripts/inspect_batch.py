#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

REPO_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import make_scratch_dataloaders  # type: ignore
from utils import load_yaml, seed_everything  # type: ignore


def resolve_config(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one collated batch and decode a few samples.")
    parser.add_argument("--config", default="configs/smoke_rnn.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--token-preview", type=int, default=20)
    args = parser.parse_args()

    cfg = load_yaml(resolve_config(args.config))
    seed_everything(cfg["seed"])
    loaders = make_scratch_dataloaders(cfg["data"], seed=cfg["seed"])
    vocab = loaders["vocab"]
    label_names = loaders["label_names"]
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

    input_ids = batch["input_ids"]
    lengths = batch["lengths"]
    labels = batch["labels"]

    print(f"config: {args.config}")
    print(f"split: {args.split}")
    print(f"input_ids.shape: {tuple(input_ids.shape)}")
    print(f"lengths.shape: {tuple(lengths.shape)}")
    print(f"labels.shape: {tuple(labels.shape)}")
    print(f"pad_id: {vocab.pad_id}")
    print(f"label_names ({len(label_names)}): {label_names}")
    print()

    for i in range(min(args.samples, input_ids.size(0))):
        true_len = int(lengths[i].item())
        label_id = int(labels[i].item())
        ids_all = input_ids[i].tolist()
        ids_active = ids_all[:true_len]
        ids_tail = ids_all[true_len:true_len + min(8, max(0, len(ids_all) - true_len))]
        tokens_active = [vocab.itos[idx] for idx in ids_active[:args.token_preview]]

        print(f"sample {i}")
        print(f"  label_id: {label_id}")
        print(f"  label_name: {label_names[label_id]}")
        print(f"  true_length: {true_len}")
        print(f"  padded_row_length: {len(ids_all)}")
        print(f"  active_ids[:{args.token_preview}]: {ids_active[:args.token_preview]}")
        print(f"  active_tokens[:{args.token_preview}]: {tokens_active}")
        print(f"  padded_tail: {ids_tail}")
        print()


if __name__ == "__main__":
    main()
