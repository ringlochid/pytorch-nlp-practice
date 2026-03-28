from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        even_dims = torch.arange(0, emb_dim, 2, dtype=torch.float32)

        angle_rates = torch.exp(-math.log(10000.0) * even_dims / emb_dim)

        pe = torch.zeros(max_len, emb_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * angle_rates)
        pe[:, 1::2] = torch.cos(positions * angle_rates[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"Expected x to have shape (B, L, E), got {tuple(x.shape)}"
            )

        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len={self.max_len}"
            )

        stored_pe = cast(torch.Tensor, self.pe)
        pe_slice = stored_pe[:, :seq_len, :].to(device=x.device, dtype=x.dtype)

        x = x + pe_slice
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, ffn_dim=None, dropout=0.1):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * emb_dim

        self.fc1 = nn.Linear(emb_dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, emb_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"Expected x to have shape (B, L, E), got {tuple(x.shape)}"
            )

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_head=8,
        ffn_dim=None,
        dropout=0.1,
    ) -> None:
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * emb_dim

        self.mha = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_head,
            batch_first=True,
            dropout=dropout,
        )
        self.ffn = FeedForward(emb_dim=emb_dim, ffn_dim=ffn_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        if x.dim() != 3:
            raise ValueError(
                f"Expected x to have shape (B, L, E), got {tuple(x.shape)}"
            )

        if pad_mask is not None:
            if pad_mask.shape != x.shape[:2]:
                raise ValueError(
                    f"Expected pad_mask to have shape {(x.shape[0], x.shape[1])}, got {tuple(pad_mask.shape)}"
                )
            pad_mask = pad_mask.to(device=x.device, dtype=torch.bool)

        attn_out, _ = self.mha(
            query=x,
            key=x,
            value=x,
            key_padding_mask=pad_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.attn_dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_layers=4,
        num_head=8,
        ffn_dim=None,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    emb_dim=emb_dim,
                    num_head=num_head,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, pad_mask=None):
        for layer in self.layers:
            x = layer(x, pad_mask=pad_mask)
        return x


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        pad_id: int,
        emb_dim,
        num_layers=4,
        num_head=8,
        ffn_dim=None,
        dropout=0.1,
        max_len=512,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.pe = PositionalEncoding(emb_dim=emb_dim, max_len=max_len, dropout=dropout)
        self.encoder = Encoder(
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_head=num_head,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, input_ids, lengths=None):
        pad_mask = input_ids == self.pad_id

        x = self.embedding(input_ids)
        x = x * math.sqrt(self.emb_dim)
        x = self.pe(x)
        x = self.encoder(x, pad_mask=pad_mask)

        keep_mask = (~pad_mask).unsqueeze(-1).to(dtype=x.dtype)
        x = x * keep_mask
        x = x.sum(dim=1) / keep_mask.sum(dim=1).clamp(min=1.0)

        x = self.dropout(x)
        x = self.classifier(x)
        return x
