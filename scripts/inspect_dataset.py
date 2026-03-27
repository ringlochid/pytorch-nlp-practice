#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import _load_ag_news_splits, simple_tokenize  # type: ignore
from utils import load_yaml  # type: ignore


def resolve_config(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect AG News dataset schema and sample rows.")
    parser.add_argument("--config", default="configs/smoke_rnn.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--token-preview", type=int, default=20)
    args = parser.parse_args()

    cfg = load_yaml(resolve_config(args.config))
    data_cfg = cfg["data"]
    train_ds, val_ds, test_ds, label_names = _load_ag_news_splits(
        train_subset=data_cfg["train_subset"],
        val_ratio=data_cfg["val_ratio"],
        test_subset=data_cfg.get("test_subset"),
        seed=cfg["seed"],
    )

    split_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    ds = split_map[args.split]

    print(f"config: {args.config}")
    print(f"selected split: {args.split}")
    print(f"rows: {len(ds)}")
    print(f"column_names: {ds.column_names}")
    print(f"features: {ds.features}")
    print(f"label_names ({len(label_names)}): {label_names}")
    print()

    for i in range(min(args.samples, len(ds))):
        row = ds[i]
        label_id = int(row["label"])
        tokens = simple_tokenize(row["text"])
        print(f"sample {i}")
        print(f"  label_id: {label_id}")
        print(f"  label_name: {label_names[label_id]}")
        print(f"  text: {row['text'][:240]}")
        print(f"  tokens[:{args.token_preview}]: {tokens[:args.token_preview]}")
        print(f"  num_tokens: {len(tokens)}")
        print()


if __name__ == "__main__":
    main()
