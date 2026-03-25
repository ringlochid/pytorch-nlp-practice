from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data import make_scratch_dataloaders
from engine import evaluate_scratch, save_checkpoint, train_epoch_scratch
from models.gru_classifier import GRUClassifier
from models.lstm_classifier import LSTMClassifier
from models.rnn_classifier import RNNClassifier
from models.transformer_encoder_classifier import TransformerEncoderClassifier
from utils import count_parameters, get_device, load_yaml, make_run_dir, save_json, seed_everything


def build_model(model_cfg: dict, vocab_size: int, num_classes: int, pad_id: int):
    name = model_cfg["name"]

    if name == "rnn":
        return RNNClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            pad_id=pad_id,
            emb_dim=model_cfg["emb_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            bidirectional=model_cfg.get("bidirectional", False),
        )
    if name == "gru":
        return GRUClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            pad_id=pad_id,
            emb_dim=model_cfg["emb_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            bidirectional=model_cfg.get("bidirectional", False),
        )
    if name == "lstm":
        return LSTMClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            pad_id=pad_id,
            emb_dim=model_cfg["emb_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            bidirectional=model_cfg.get("bidirectional", False),
        )
    if name == "transformer":
        return TransformerEncoderClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            pad_id=pad_id,
            max_length=model_cfg["max_length"],
            d_model=model_cfg["d_model"],
            nhead=model_cfg["nhead"],
            num_layers=model_cfg["num_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"],
        )
    raise ValueError(f"Unsupported model name: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
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

    run_dir = make_run_dir(cfg["system"]["output_root"], cfg["experiment_name"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    best_val_f1 = float("-inf")
    history = []

    print(f"device={device}")
    print(f"params={count_parameters(model):,}")
    print(f"run_dir={run_dir}")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_metrics = train_epoch_scratch(
            model,
            loaders["train_loader"],
            optimizer,
            criterion,
            device,
            grad_clip=cfg["train"].get("grad_clip"),
        )
        val_metrics = evaluate_scratch(model, loaders["val_loader"], criterion, device)

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            save_checkpoint(model, run_dir / "best.pt")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    test_metrics = evaluate_scratch(model, loaders["test_loader"], criterion, device)
    print(
        f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f} test_f1={test_metrics['macro_f1']:.4f}"
    )

    save_json(
        run_dir / "metrics.json",
        {
            "config": cfg,
            "params": count_parameters(model),
            "history": history,
            "test": test_metrics,
        },
    )
    Path(run_dir / "config_used.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
