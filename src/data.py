from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Iterable

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


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


def _shuffle_and_maybe_subset(dataset, subset_size: int | None, seed: int):
    dataset = dataset.shuffle(seed=seed)
    if subset_size is not None:
        subset_size = min(subset_size, len(dataset))
        dataset = dataset.select(range(subset_size))
    return dataset


def _load_ag_news_splits(
    train_subset: int | None, val_ratio: float, test_subset: int | None, seed: int
):
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
