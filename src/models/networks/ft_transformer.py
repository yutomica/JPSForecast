import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class NumericalFeatureTokenizer(nn.Module):
    """
    Scalar numerical feature -> token vector.

    Each numerical feature x_j is converted as:
        token_j = x_j * W_j + b_j
    where W_j, b_j are learnable vectors in R^{d_token}.
    """

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.d_token = int(d_token)

        if self.n_features > 0:
            self.weight = nn.Parameter(torch.empty(self.n_features, self.d_token))
            self.bias = nn.Parameter(torch.empty(self.n_features, self.d_token))
            self.reset_parameters()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        if self.weight is None:
            return
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.size(1)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_num: torch.Tensor) -> Optional[torch.Tensor]:
        if self.n_features == 0:
            return None
        if x_num.ndim != 2:
            raise ValueError(f"x_num must be [B, N_num], got shape={tuple(x_num.shape)}")
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalFeatureTokenizer(nn.Module):
    """
    Categorical feature -> token vector via per-feature embedding tables
    packed into a single embedding with offsets.
    """

    def __init__(self, cardinalities: Sequence[int], d_token: int) -> None:
        super().__init__()
        self.cardinalities = [int(x) for x in cardinalities]
        self.n_features = len(self.cardinalities)
        self.d_token = int(d_token)

        if self.n_features > 0:
            offsets = [0]
            for size in self.cardinalities[:-1]:
                offsets.append(offsets[-1] + size)
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
            total = int(sum(self.cardinalities))
            self.embedding = nn.Embedding(total, self.d_token)
            self.bias = nn.Parameter(torch.empty(self.n_features, self.d_token))
            self.reset_parameters()
        else:
            self.register_buffer("offsets", torch.zeros(0, dtype=torch.long))
            self.embedding = None
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        if self.embedding is None:
            return
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
        fan_in = self.embedding.embedding_dim
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_cat: torch.Tensor) -> Optional[torch.Tensor]:
        if self.n_features == 0:
            return None
        if x_cat.ndim != 2:
            raise ValueError(f"x_cat must be [B, N_cat], got shape={tuple(x_cat.shape)}")

        x_cat = torch.clamp(x_cat.long(), min=0)
        x = x_cat + self.offsets.unsqueeze(0)
        return self.embedding(x) + self.bias.unsqueeze(0)


class FeatureTokenizer(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: Sequence[int],
        d_token: int,
    ) -> None:
        super().__init__()
        self.n_num_features = int(n_num_features)
        self.cat_cardinalities = [int(x) for x in cat_cardinalities]
        self.n_cat_features = len(self.cat_cardinalities)
        self.d_token = int(d_token)

        self.num_tokenizer = NumericalFeatureTokenizer(self.n_num_features, self.d_token)
        self.cat_tokenizer = CategoricalFeatureTokenizer(self.cat_cardinalities, self.d_token)

    @property
    def n_tokens(self) -> int:
        return self.n_num_features + self.n_cat_features

    def forward(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pieces = []

        if self.n_num_features > 0:
            if x_num is None:
                raise ValueError("x_num is required because n_num_features > 0")
            pieces.append(self.num_tokenizer(x_num))

        if self.n_cat_features > 0:
            if x_cat is None:
                raise ValueError("x_cat is required because n_cat_features > 0")
            pieces.append(self.cat_tokenizer(x_cat))

        if not pieces:
            raise ValueError("At least one numerical or categorical feature is required.")

        return torch.cat(pieces, dim=1)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        attention_dropout: float = 0.1,
        ffn_d_hidden: Optional[int] = None,
        ffn_multiplier: float = 4.0,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.d_token = int(d_token)
        self.n_heads = int(n_heads)
        self.ffn_d_hidden = int(ffn_d_hidden or (ffn_multiplier * d_token))

        self.norm_attn = nn.LayerNorm(self.d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_token,
            num_heads=self.n_heads,
            dropout=float(attention_dropout),
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(float(residual_dropout))

        self.norm_ffn = nn.LayerNorm(self.d_token)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_token, self.ffn_d_hidden),
            _make_activation(activation),
            nn.Dropout(float(ffn_dropout)),
            nn.Linear(self.ffn_d_hidden, self.d_token),
        )
        self.dropout_ffn = nn.Dropout(float(residual_dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm_attn(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout_attn(attn_out)

        ffn_input = self.norm_ffn(x)
        ffn_out = self.ffn(ffn_input)
        x = x + self.dropout_ffn(ffn_out)
        return x


class FTTransformer(nn.Module):
    """
    FT-Transformer for tabular data.

    Inputs:
      x_num: [B, N_num] float tensor or None
      x_cat: [B, N_cat] long tensor or None
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: Sequence[int],
        d_token: int = 192,
        n_blocks: int = 3,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_d_hidden: Optional[int] = None,
        ffn_multiplier: float = 4.0,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        activation: str = "gelu",
        output_dim: int = 1,
        head_hidden_dim: int = 0,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_num_features = int(n_num_features)
        self.cat_cardinalities = [int(x) for x in cat_cardinalities]
        self.d_token = int(d_token)
        self.n_blocks = int(n_blocks)
        self.output_dim = int(output_dim)

        self.tokenizer = FeatureTokenizer(
            n_num_features=self.n_num_features,
            cat_cardinalities=self.cat_cardinalities,
            d_token=self.d_token,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_token=self.d_token,
                    n_heads=attention_n_heads,
                    attention_dropout=attention_dropout,
                    ffn_d_hidden=ffn_d_hidden,
                    ffn_multiplier=ffn_multiplier,
                    ffn_dropout=ffn_dropout,
                    residual_dropout=residual_dropout,
                    activation=activation,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(self.d_token)

        if head_hidden_dim and int(head_hidden_dim) > 0:
            self.head = nn.Sequential(
                nn.Linear(self.d_token, int(head_hidden_dim)),
                _make_activation(activation),
                nn.Dropout(float(head_dropout)),
                nn.Linear(int(head_hidden_dim), self.output_dim),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(float(head_dropout)),
                nn.Linear(self.d_token, self.output_dim),
            )

    def forward(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.tokenizer(x_num=x_num, x_cat=x_cat)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        cls_rep = x[:, 0]
        return self.head(cls_rep)
