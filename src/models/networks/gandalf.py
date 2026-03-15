import math
from typing import Iterable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TSoftmax(nn.Module):
    """
    A lightweight sparse mask activation inspired by the t-softmax idea used in the
    GANDALF paper. The original paper references the t-softmax formulation from
    Balazy et al.; here we implement a practical, dependency-free approximation
    that preserves the key behavior we need for GFLU masks:

    - starts from a probability simplex (softmax)
    - applies a learnable threshold to induce exact zeros
    - renormalizes the surviving mass

    This keeps the implementation fully in PyTorch, without external sparse-activation
    dependencies.
    """

    def __init__(
        self,
        init_threshold: float = 0.05,
        learnable: bool = True,
        dim: int = -1,
        eps: float = 1e-8,
    ):
        super().__init__()
        init_threshold = float(max(1e-6, min(0.95, init_threshold)))
        threshold_tensor = torch.tensor(init_threshold, dtype=torch.float32)
        if learnable:
            self.threshold = nn.Parameter(threshold_tensor)
        else:
            self.register_buffer("threshold", threshold_tensor)
        self.dim = dim
        self.eps = eps

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        base = torch.softmax(logits, dim=self.dim)
        # keep threshold in a numerically stable range
        t = torch.clamp(self.threshold, min=0.0, max=0.95)
        masked = F.relu(base - t)
        denom = masked.sum(dim=self.dim, keepdim=True)
        # fallback to the dense softmax if every element gets zeroed out
        sparse = masked / denom.clamp_min(self.eps)
        use_dense = denom <= self.eps
        return torch.where(use_dense, base, sparse)


class GFLUStage(nn.Module):
    """
    Single Gated Feature Learning Unit (GFLU) stage.

    Equations follow the paper conceptually:
    - global learnable feature mask for the gate inputs
    - reset/update gating akin to GRU, but for tabular non-sequential processing
    - candidate path uses the raw input x (not masked x) to avoid choking gradient flow
    """

    def __init__(
        self,
        dim: int,
        feature_init_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.mask_logits = nn.Parameter(torch.empty(dim))
        self.mask_activation = TSoftmax(
            init_threshold=max(1e-4, feature_init_sparsity / max(dim, 1)),
            learnable=learnable_sparsity,
        )

        self.gate_linear = nn.Linear(dim * 2, dim * 2)
        self.candidate_linear = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.reset_parameters(feature_init_sparsity)

    def reset_parameters(self, feature_init_sparsity: float = 0.3) -> None:
        # Beta initialization is explicitly discussed in the paper for mask diversity.
        # We use a sparse-biased beta draw to initialize stage-wise feature preferences.
        sparsity = float(max(1e-4, min(0.999, feature_init_sparsity)))
        alpha = max(0.5, (1.0 - sparsity) * 2.0)
        beta = max(0.5, sparsity * 6.0)
        with torch.no_grad():
            beta_dist = torch.distributions.Beta(alpha, beta)
            mask_init = beta_dist.sample((self.dim,)).clamp_(1e-4, 1.0 - 1e-4)
            self.mask_logits.copy_(torch.log(mask_init))
        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.zeros_(self.gate_linear.bias)
        nn.init.xavier_uniform_(self.candidate_linear.weight)
        nn.init.zeros_(self.candidate_linear.bias)

    def get_mask(self) -> torch.Tensor:
        return self.mask_activation(self.mask_logits)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mask = self.get_mask()  # [dim]
        x_masked = x * mask.unsqueeze(0)

        gate_in = torch.cat([x_masked, h], dim=-1)
        z_pre, r_pre = self.gate_linear(gate_in).chunk(2, dim=-1)
        z = torch.sigmoid(z_pre)
        r = torch.sigmoid(r_pre)

        cand_in = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.candidate_linear(cand_in))
        h_next = (1.0 - z) * h + z * h_tilde
        return self.dropout(h_next)


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[Iterable[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [])
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GANDALFNet(nn.Module):
    """
    Fully-scratch PyTorch implementation of GANDALF for dense tabular inputs.

    Notes:
    - Input is assumed to already be numerical (imputed/scaled/encoded by the preprocessor).
    - Hidden representation keeps the same width as input_dim, matching the paper's
      default choice for simplicity and feature-importance interpretability.
    - Feature importance is the average of the global feature masks across GFLU stages.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        gflu_stages: int = 6,
        gflu_dropout: float = 0.0,
        feature_init_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        head_hidden_dims: Optional[Iterable[int]] = None,
        head_dropout: float = 0.0,
        target_bias: Optional[float] = None,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if gflu_stages <= 0:
            raise ValueError("gflu_stages must be > 0")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.gflu_stages = int(gflu_stages)

        self.stages = nn.ModuleList(
            [
                GFLUStage(
                    dim=self.input_dim,
                    feature_init_sparsity=feature_init_sparsity,
                    learnable_sparsity=learnable_sparsity,
                    dropout=gflu_dropout,
                )
                for _ in range(self.gflu_stages)
            ]
        )
        self.head = MLPHead(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=head_hidden_dims,
            dropout=head_dropout,
        )
        if target_bias is not None:
            last_linear = None
            for module in reversed(self.head.net):
                if isinstance(module, nn.Linear):
                    last_linear = module
                    break
            if last_linear is not None:
                nn.init.constant_(last_linear.bias, float(target_bias))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for stage in self.stages:
            h = stage(x, h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x)
        return self.head(h)

    @torch.no_grad()
    def get_feature_importance(self, normalize: bool = True) -> torch.Tensor:
        masks = [stage.get_mask() for stage in self.stages]
        if not masks:
            imp = torch.ones(self.input_dim, device=next(self.parameters()).device)
        else:
            imp = torch.stack(masks, dim=0).mean(dim=0)
        if normalize:
            imp = imp / imp.sum().clamp_min(1e-8)
        return imp
