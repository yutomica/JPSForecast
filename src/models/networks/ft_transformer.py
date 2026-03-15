# ft_transformer.py
from __future__ import annotations

from typing import Optional, Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericalFeatureTokenizer(nn.Module):
    """
    各数値特徴量 x_j を:
        T_j = b_j + x_j * W_j
    により d_token 次元へ写像する。
    """
    def __init__(self, n_num_features: int, d_token: int) -> None:
        super().__init__()
        self.n_num_features = n_num_features
        self.d_token = d_token

        if n_num_features > 0:
            self.weight = nn.Parameter(torch.empty(n_num_features, d_token))
            self.bias = nn.Parameter(torch.empty(n_num_features, d_token))
            self.reset_parameters()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.normal_(self.weight, std=0.02)
            nn.init.zeros_(self.bias)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        if x_num.ndim != 2:
            raise ValueError(f"x_num must be 2D, got shape={tuple(x_num.shape)}")

        batch_size = x_num.shape[0]

        if self.n_num_features == 0:
            return x_num.new_zeros(batch_size, 0, self.d_token)

        if x_num.shape[1] != self.n_num_features:
            raise ValueError(
                f"Expected {self.n_num_features} numerical features, got {x_num.shape[1]}"
            )

        # [B, N] -> [B, N, 1] * [1, N, D] + [1, N, D] = [B, N, D]
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalFeatureTokenizer(nn.Module):
    """
    各カテゴリ特徴量 x_j を embedding lookup で d_token 次元へ写像する。
    未知カテゴリは前段の前処理で 0 に寄せ、既知カテゴリは 1..K にする前提。
    """
    def __init__(self, cardinalities: Optional[Sequence[int]], d_token: int) -> None:
        super().__init__()
        self.cardinalities = list(cardinalities or [])
        self.n_cat_features = len(self.cardinalities)
        self.d_token = d_token

        if self.n_cat_features > 0:
            total_cardinality = sum(self.cardinalities)
            self.embedding = nn.Embedding(total_cardinality, d_token)
            self.bias = nn.Parameter(torch.empty(self.n_cat_features, d_token))

            offsets = []
            running = 0
            for card in self.cardinalities:
                offsets.append(running)
                running += card
            self.register_buffer(
                "category_offsets",
                torch.tensor(offsets, dtype=torch.long),
                persistent=False,
            )
            self.reset_parameters()
        else:
            self.embedding = None
            self.register_parameter("bias", None)
            self.register_buffer(
                "category_offsets",
                torch.zeros(0, dtype=torch.long),
                persistent=False,
            )

    def reset_parameters(self) -> None:
        if self.embedding is not None:
            nn.init.normal_(self.embedding.weight, std=0.02)
            nn.init.zeros_(self.bias)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        if x_cat.ndim != 2:
            raise ValueError(f"x_cat must be 2D, got shape={tuple(x_cat.shape)}")

        batch_size = x_cat.shape[0]

        if self.n_cat_features == 0:
            return torch.zeros(
                batch_size, 0, self.d_token, device=x_cat.device, dtype=torch.float32
            )

        if x_cat.shape[1] != self.n_cat_features:
            raise ValueError(
                f"Expected {self.n_cat_features} categorical features, got {x_cat.shape[1]}"
            )

        # 各列ごとに埋め込みテーブルの参照位置がぶつからないよう offset を加算
        x_cat = x_cat.long() + self.category_offsets.unsqueeze(0)  # [B, C]
        return self.embedding(x_cat) + self.bias.unsqueeze(0)      # [B, C, D]


class FeatureTokenizer(nn.Module):
    """
    数値特徴量トークン + カテゴリ特徴量トークン を連結して返す。
    """
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: Optional[Sequence[int]],
        d_token: int,
    ) -> None:
        super().__init__()
        self.n_num_features = n_num_features
        self.cat_cardinalities = list(cat_cardinalities or [])
        self.n_cat_features = len(self.cat_cardinalities)
        self.d_token = d_token

        self.num_tokenizer = NumericalFeatureTokenizer(n_num_features, d_token)
        self.cat_tokenizer = CategoricalFeatureTokenizer(self.cat_cardinalities, d_token)

        if self.n_num_features + self.n_cat_features == 0:
            raise ValueError("At least one feature is required.")

    def forward(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        parts = []

        if self.n_num_features > 0:
            if x_num is None:
                raise ValueError("x_num is required but missing.")
            parts.append(self.num_tokenizer(x_num))

        if self.n_cat_features > 0:
            if x_cat is None:
                raise ValueError("x_cat is required but missing.")
            parts.append(self.cat_tokenizer(x_cat))

        return torch.cat(parts, dim=1)  # [B, N_tokens, D]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_token: int, n_heads: int, attention_dropout: float) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError("d_token must be divisible by n_heads")

        self.d_token = d_token
        self.n_heads = n_heads
        self.d_head = d_token // n_heads

        self.q_proj = nn.Linear(d_token, d_token)
        self.k_proj = nn.Linear(d_token, d_token)
        self.v_proj = nn.Linear(d_token, d_token)
        self.out_proj = nn.Linear(d_token, d_token)

        self.attn_dropout = nn.Dropout(attention_dropout)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, H, T, Dh]
        bsz, n_tokens, _ = x.shape
        x = x.view(bsz, n_tokens, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, d_token = x.shape

        q = self._reshape(self.q_proj(x))
        k = self._reshape(self.k_proj(x))
        v = self._reshape(self.v_proj(x))

        # [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
        out = torch.matmul(attn, v)

        # [B, H, T, Dh] -> [B, T, D]
        out = out.transpose(1, 2).contiguous().view(bsz, n_tokens, d_token)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_token: int, hidden_multiplier: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(d_token, int(d_token * hidden_multiplier))
        self.net = nn.Sequential(
            nn.Linear(d_token, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_token),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer block
    """
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        ffn_hidden_multiplier: float,
        ffn_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_token)
        self.ffn_norm = nn.LayerNorm(d_token)

        self.attn = MultiHeadSelfAttention(
            d_token=d_token,
            n_heads=n_heads,
            attention_dropout=attention_dropout,
        )
        self.ffn = FeedForward(
            d_token=d_token,
            hidden_multiplier=ffn_hidden_multiplier,
            dropout=ffn_dropout,
        )
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.attn(self.attn_norm(x)))
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
        return x


class FTTransformer(nn.Module):
    """
    クリーンな standalone 実装。
    - Feature Tokenizer
    - [CLS] token
    - Transformer Encoder blocks
    - Prediction head: Linear(ReLU(LayerNorm(CLS)))
    """
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: Optional[Sequence[int]] = None,
        d_token: int = 192,
        n_blocks: int = 3,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.1,
        ffn_hidden_multiplier: float = 4.0,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        output_dim: int = 1,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_num_features = n_num_features
        self.cat_cardinalities = list(cat_cardinalities or [])
        self.output_dim = output_dim
        self.d_token = d_token

        self.tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=self.cat_cardinalities,
            d_token=d_token,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_token=d_token,
                    n_heads=attention_n_heads,
                    attention_dropout=attention_dropout,
                    ffn_hidden_multiplier=ffn_hidden_multiplier,
                    ffn_dropout=ffn_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(d_token)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(d_token, output_dim)

    def _append_cls(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, D]
        return torch.cat([cls, x], dim=1)

    def encode(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.tokenizer(x_num, x_cat)  # [B, T, D]
        x = self._append_cls(x)           # [B, T+1, D]

        for block in self.blocks:
            x = block(x)

        cls = x[:, 0]                     # [B, D]
        cls = self.final_norm(cls)
        cls = F.relu(cls)
        cls = self.head_dropout(cls)
        return cls

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cls = self.encode(x_num, x_cat)
        out = self.head(cls)
        if self.output_dim == 1:
            return out.squeeze(-1)
        return out