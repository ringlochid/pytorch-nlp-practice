from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm

from metrics import classification_metrics


def train_epoch_scratch(
    model, loader, optimizer, criterion, device, grad_clip: float | None = None
):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    torch.autograd.set_detect_anomaly(True)

    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(input_ids=input_ids, lengths=lengths)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    metrics = classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics


@torch.no_grad()
def evaluate_scratch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, lengths=lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    metrics = classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics


def train_epoch_hf(model, loader, optimizer, device, grad_clip: float | None = None):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    total_examples = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = labels.size(0)
        total_examples += bs
        total_loss += loss.item() * bs
        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    metrics = classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics


@torch.no_grad()
def evaluate_hf(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    total_examples = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        outputs = model(**batch)
        loss = outputs.loss

        bs = labels.size(0)
        total_examples += bs
        total_loss += loss.item() * bs
        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    metrics = classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics


def save_checkpoint(model, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
