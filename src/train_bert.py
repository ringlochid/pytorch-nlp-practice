from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
)

from data import PAD_TOKEN, UNK_TOKEN, _load_ag_news_splits, build_vocab, make_hf_dataloaders
from engine import evaluate_hf, train_epoch_hf
from utils import (
    count_parameters,
    get_device,
    load_yaml,
    make_run_dir,
    save_json,
    seed_everything,
)

BERT_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _build_bootstrap_vocab(train_texts, vocab_size: int, min_freq: int, lowercase: bool):
    scratch_vocab = build_vocab(
        train_texts,
        max_vocab_size=max(vocab_size, len(BERT_SPECIAL_TOKENS)) + 2,
        min_freq=min_freq,
        lowercase=lowercase,
    )

    vocab_tokens = []
    seen = set()
    for token in BERT_SPECIAL_TOKENS + scratch_vocab.itos:
        if token in seen or token in {PAD_TOKEN, UNK_TOKEN}:
            continue
        seen.add(token)
        vocab_tokens.append(token)
        if len(vocab_tokens) >= vocab_size:
            break

    return vocab_tokens


def _bootstrap_local_tiny_bert(model_cfg: dict, data_cfg: dict, seed: int, num_labels: int):
    output_dir = Path(model_cfg["pretrained_model_name_or_path"])
    required_files = (output_dir / "config.json", output_dir / "vocab.txt")
    if all(path.exists() for path in required_files):
        return

    bootstrap_cfg = model_cfg.get("bootstrap", {})
    train_ds, _, _, _ = _load_ag_news_splits(
        train_subset=bootstrap_cfg.get(
            "train_subset_for_vocab",
            data_cfg.get("train_subset"),
        ),
        val_ratio=data_cfg["val_ratio"],
        test_subset=data_cfg.get("test_subset"),
        seed=seed,
        cache_dir=data_cfg.get("ag_news_cache_dir"),
    )

    vocab_tokens = _build_bootstrap_vocab(
        train_texts=train_ds["text"],
        vocab_size=bootstrap_cfg.get("vocab_size", 4096),
        min_freq=bootstrap_cfg.get("min_freq", 1),
        lowercase=bootstrap_cfg.get("lowercase", True),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")

    tokenizer = BertTokenizer(
        vocab_file=str(vocab_path),
        do_lower_case=bootstrap_cfg.get("lowercase", True),
    )

    config = BertConfig(
        vocab_size=len(vocab_tokens),
        hidden_size=bootstrap_cfg.get("hidden_size", 128),
        num_hidden_layers=bootstrap_cfg.get("num_hidden_layers", 2),
        num_attention_heads=bootstrap_cfg.get("num_attention_heads", 2),
        intermediate_size=bootstrap_cfg.get("intermediate_size", 256),
        hidden_dropout_prob=bootstrap_cfg.get("hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob=bootstrap_cfg.get(
            "attention_probs_dropout_prob", 0.1
        ),
        max_position_embeddings=max(
            bootstrap_cfg.get("max_position_embeddings", 128),
            data_cfg["max_length"] + 2,
        ),
        num_labels=num_labels,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = BertForSequenceClassification(config)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_json(
        output_dir / "bootstrap_meta.json",
        {
            "kind": "offline_smoke_only",
            "seed": seed,
            "num_labels": num_labels,
            "vocab_size": len(vocab_tokens),
            "config": config.to_dict(),
        },
    )


def _load_tokenizer(model_cfg: dict):
    return BertTokenizer.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        local_files_only=model_cfg.get("local_files_only", False),
        do_lower_case=True,
    )


def _load_model(model_cfg: dict, num_labels: int):
    return BertForSequenceClassification.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        num_labels=num_labels,
        local_files_only=model_cfg.get("local_files_only", False),
        ignore_mismatched_sizes=True,
    )


def _save_hf_checkpoint(model, tokenizer, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg["seed"])

    device = get_device(cfg["system"].get("device", "cpu"))
    _, _, _, label_names = _load_ag_news_splits(
        train_subset=cfg["data"]["train_subset"],
        val_ratio=cfg["data"]["val_ratio"],
        test_subset=cfg["data"].get("test_subset"),
        seed=cfg["seed"],
        cache_dir=cfg["data"].get("ag_news_cache_dir"),
    )

    if cfg["model"].get("bootstrap_local_model", False):
        _bootstrap_local_tiny_bert(
            cfg["model"],
            cfg["data"],
            seed=cfg["seed"],
            num_labels=len(label_names),
        )

    tokenizer = _load_tokenizer(cfg["model"])
    loaders = make_hf_dataloaders(cfg["data"], seed=cfg["seed"], tokenizer=tokenizer)
    model = _load_model(cfg["model"], num_labels=len(loaders["label_names"])).to(device)

    run_dir = make_run_dir(cfg["system"]["output_root"], cfg["experiment_name"])
    best_dir = run_dir / "best"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    best_val_f1 = float("-inf")
    history = []

    print(f"device={device}")
    print(f"params={count_parameters(model):,}")
    print(f"checkpoint={cfg['model']['pretrained_model_name_or_path']}")
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
            _save_hf_checkpoint(model, tokenizer, best_dir)

    model = BertForSequenceClassification.from_pretrained(
        best_dir,
        local_files_only=True,
    ).to(device)
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
    Path(run_dir / "config_used.yaml").write_text(
        Path(args.config).read_text(encoding="utf-8"), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
