from __future__ import annotations

import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
AG_NEWS_CACHE_DIR = ".cache/ag_news"


TOKEN_RE = re.compile(r"\b\w+\b")


def simple_tokenize(text: str, lowercase: bool = True) -> list[str]:
    text = text.lower() if lowercase else text
    return TOKEN_RE.findall(text)


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = PAD_TOKEN
    unk_token: str = UNK_TOKEN

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.stoi.get(tok, self.unk_id) for tok in tokens]


class ScratchTextDataset(Dataset):
    def __init__(
        self, hf_dataset, vocab: Vocab, max_length: int, lowercase: bool = True
    ):
        self.dataset = hf_dataset
        self.vocab = vocab
        self.max_length = max_length
        self.lowercase = lowercase

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[idx]
        tokens = simple_tokenize(row["text"], lowercase=self.lowercase)
        token_ids = self.vocab.encode(tokens[: self.max_length])
        if not token_ids:
            token_ids = [self.vocab.unk_id]
        return token_ids, int(row["label"])


class HFTextDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[idx]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = int(row["label"])
        return encoded


def build_vocab(
    texts: Iterable[str],
    max_vocab_size: int = 20000,
    min_freq: int = 2,
    lowercase: bool = True,
) -> Vocab:
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text, lowercase=lowercase))

    stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    itos = [PAD_TOKEN, UNK_TOKEN]

    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in stoi:
            continue
        stoi[token] = len(itos)
        itos.append(token)
        if len(itos) >= max_vocab_size:
            break

    return Vocab(stoi=stoi, itos=itos)


def scratch_collate_fn(batch, pad_id: int):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    input_ids = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in sequences],
        batch_first=True,
        padding_value=pad_id,
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return {"input_ids": input_ids, "lengths": lengths, "labels": labels}


def hf_collate_fn(batch, collator: DataCollatorWithPadding):
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    features = [{k: v for k, v in item.items() if k != "labels"} for item in batch]
    padded = collator(features)
    padded["labels"] = labels
    return padded


def _shuffle_and_maybe_subset(dataset, subset_size: int | None, seed: int):
    dataset = dataset.shuffle(seed=seed)
    if subset_size is not None:
        subset_size = min(subset_size, len(dataset))
        dataset = dataset.select(range(subset_size))
    return dataset


def _ag_news_cache_candidates() -> list[Path]:
    explicit_cache = os.environ.get("PYTORCH_NLP_PRACTICE_AG_NEWS_CACHE")
    candidates = []
    if explicit_cache:
        candidates.append(Path(explicit_cache))

    candidates.append(Path(AG_NEWS_CACHE_DIR))
    candidates.extend(
        Path.home().glob(
            ".cache/huggingface/datasets/ag_news/default/*/*"
        )
    )
    return candidates


def _is_ag_news_arrow_dir(path: Path) -> bool:
    return all(
        (path / name).exists()
        for name in ("ag_news-train.arrow", "ag_news-test.arrow", "dataset_info.json")
    )


def _ensure_local_ag_news_cache(cache_dir: str | None = None) -> Path | None:
    target_dir = Path(cache_dir or AG_NEWS_CACHE_DIR)
    if _is_ag_news_arrow_dir(target_dir):
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    for candidate in _ag_news_cache_candidates():
        if not candidate.exists() or not _is_ag_news_arrow_dir(candidate):
            continue
        if candidate.resolve() == target_dir.resolve():
            return target_dir
        shutil.copytree(candidate, target_dir, dirs_exist_ok=True)
        return target_dir

    return None


def _load_ag_news_from_arrow(cache_dir: str | None = None):
    arrow_dir = _ensure_local_ag_news_cache(cache_dir=cache_dir)
    if arrow_dir is None:
        return None

    train_ds = HFDataset.from_file(str(arrow_dir / "ag_news-train.arrow"))
    test_ds = HFDataset.from_file(str(arrow_dir / "ag_news-test.arrow"))
    label_names = train_ds.features["label"].names
    return train_ds, test_ds, label_names


def _load_ag_news_splits(
    train_subset: int | None,
    val_ratio: float,
    test_subset: int | None,
    seed: int,
    cache_dir: str | None = None,
):
    cached = _load_ag_news_from_arrow(cache_dir=cache_dir)
    if cached is not None:
        raw_train, raw_test, label_names = cached
    else:
        raw_train = load_dataset("ag_news", split="train")
        raw_test = load_dataset("ag_news", split="test")
        label_names = raw_train.features["label"].names

    raw_train = _shuffle_and_maybe_subset(raw_train, train_subset, seed)
    raw_test = _shuffle_and_maybe_subset(raw_test, test_subset, seed)

    try:
        split = raw_train.train_test_split(
            test_size=val_ratio,
            seed=seed,
            stratify_by_column="label",
        )
    except ValueError:
        split = raw_train.train_test_split(test_size=val_ratio, seed=seed)

    return split["train"], split["test"], raw_test, label_names


def make_scratch_dataloaders(config: dict, seed: int):
    train_ds, val_ds, test_ds, label_names = _load_ag_news_splits(
        train_subset=config["train_subset"],
        val_ratio=config["val_ratio"],
        test_subset=config.get("test_subset"),
        seed=seed,
        cache_dir=config.get("ag_news_cache_dir"),
    )

    vocab = build_vocab(
        train_ds["text"],
        max_vocab_size=config["max_vocab_size"],
        min_freq=config["min_freq"],
        lowercase=config.get("lowercase", True),
    )

    max_length = config["max_length"]
    lowercase = config.get("lowercase", True)

    train_set = ScratchTextDataset(
        train_ds, vocab=vocab, max_length=max_length, lowercase=lowercase
    )
    val_set = ScratchTextDataset(
        val_ds, vocab=vocab, max_length=max_length, lowercase=lowercase
    )
    test_set = ScratchTextDataset(
        test_ds, vocab=vocab, max_length=max_length, lowercase=lowercase
    )

    collate = partial(scratch_collate_fn, pad_id=vocab.pad_id)

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config.get("num_workers", 0),
        "pin_memory": False,
        "collate_fn": collate,
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "vocab": vocab,
        "label_names": label_names,
    }


def make_hf_dataloaders(config: dict, seed: int, tokenizer):
    train_ds, val_ds, test_ds, label_names = _load_ag_news_splits(
        train_subset=config["train_subset"],
        val_ratio=config["val_ratio"],
        test_subset=config.get("test_subset"),
        seed=seed,
        cache_dir=config.get("ag_news_cache_dir"),
    )

    max_length = config["max_length"]
    train_set = HFTextDataset(train_ds, tokenizer=tokenizer, max_length=max_length)
    val_set = HFTextDataset(val_ds, tokenizer=tokenizer, max_length=max_length)
    test_set = HFTextDataset(test_ds, tokenizer=tokenizer, max_length=max_length)

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
        pad_to_multiple_of=config.get("pad_to_multiple_of"),
    )

    collate = partial(hf_collate_fn, collator=collator)
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config.get("num_workers", 0),
        "pin_memory": False,
        "collate_fn": collate,
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "label_names": label_names,
    }
