from __future__ import annotations

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        pad_id: int,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        encoded, _ = self.encoder(embedded)
        mask = (input_ids != self.pad_id).unsqueeze(-1)
        encoded = encoded * mask
        pooled = encoded.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(self.dropout(pooled))
