#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import _load_ag_news_splits, build_vocab, simple_tokenize  # type: ignore
from utils import load_yaml  # type: ignore


def resolve_config(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the scratch vocabulary and one encoded example.")
    parser.add_argument("--config", default="configs/smoke_rnn.yaml")
    parser.add_argument("--preview", type=int, default=20)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(resolve_config(args.config))
    data_cfg = cfg["data"]
    train_ds, _, _, label_names = _load_ag_news_splits(
        train_subset=data_cfg["train_subset"],
        val_ratio=data_cfg["val_ratio"],
        test_subset=data_cfg.get("test_subset"),
        seed=cfg["seed"],
    )

    vocab = build_vocab(
        train_ds["text"],
        max_vocab_size=data_cfg["max_vocab_size"],
        min_freq=data_cfg["min_freq"],
        lowercase=data_cfg.get("lowercase", True),
    )

    row = train_ds[args.sample_index]
    tokens = simple_tokenize(row["text"], lowercase=data_cfg.get("lowercase", True))
    encoded = vocab.encode(tokens[: data_cfg["max_length"]])
    decoded = [vocab.itos[idx] for idx in encoded[:args.preview]]

    print(f"config: {args.config}")
    print(f"vocab_size: {len(vocab)}")
    print(f"pad_token/id: {vocab.pad_token} / {vocab.pad_id}")
    print(f"unk_token/id: {vocab.unk_token} / {vocab.unk_id}")
    print(f"label_names ({len(label_names)}): {label_names}")
    print()
    print(f"first {args.preview} vocab entries:")
    for i, token in enumerate(vocab.itos[:args.preview]):
        print(f"  {i:>4}: {token}")
    print()
    print(f"sample_index: {args.sample_index}")
    print(f"raw text: {row['text'][:240]}")
    print(f"tokens[:{args.preview}]: {tokens[:args.preview]}")
    print(f"encoded[:{args.preview}]: {encoded[:args.preview]}")
    print(f"decoded[:{args.preview}]: {decoded}")


if __name__ == "__main__":
    main()
