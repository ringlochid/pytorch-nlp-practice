from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from data import make_bert_dataloaders
from engine import evaluate_hf, train_epoch_hf
from models.bert_classifier import build_bert_classifier
from utils import count_parameters, get_device, load_yaml, make_run_dir, save_json, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg["seed"])

    device = get_device(cfg["system"].get("device", "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    loaders = make_bert_dataloaders(cfg["data"], tokenizer=tokenizer, seed=cfg["seed"])
    label_names = loaders["label_names"]

    model = build_bert_classifier(cfg["model"]["model_name"], num_labels=len(label_names)).to(device)

    run_dir = make_run_dir(cfg["system"]["output_root"], cfg["experiment_name"])
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
        train_metrics = train_epoch_hf(
            model,
            loaders["train_loader"],
            optimizer,
            device,
            grad_clip=cfg["train"].get("grad_clip"),
        )
        val_metrics = evaluate_hf(model, loaders["val_loader"], device)

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
            torch.save(model.state_dict(), run_dir / "best.pt")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    test_metrics = evaluate_hf(model, loaders["test_loader"], device)
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
