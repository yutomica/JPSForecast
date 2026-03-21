import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):
    """
    Generic fully-connected N-BEATS block.

    Parameters
    ----------
    input_size:
        Flattened backcast input length.
    theta_size:
        Size of the learned basis coefficient vector.
    hidden_size:
        Width of each fully-connected layer.
    n_layers:
        Number of fully-connected layers before the theta projection.
    backcast_size:
        Backcast length.
    forecast_size:
        Forecast length.
    dropout:
        Dropout probability.
    activation:
        Activation module name: relu / gelu / selu / leaky_relu.
    """

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        hidden_size: int,
        n_layers: int,
        backcast_size: int,
        forecast_size: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

        layers: List[nn.Module] = []
        in_features = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(_build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        self.fc_stack = nn.Sequential(*layers)
        self.theta = nn.Linear(in_features, theta_size)
        self.backcast_head = nn.Linear(theta_size, backcast_size)
        self.forecast_head = nn.Linear(theta_size, forecast_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc_stack(x)
        theta = self.theta(h)
        backcast = self.backcast_head(theta)
        forecast = self.forecast_head(theta)
        return backcast, forecast


class TrendBlock(nn.Module):
    """Interpretable trend block using polynomial basis."""

    def __init__(
        self,
        input_size: int,
        degree: int,
        hidden_size: int,
        n_layers: int,
        backcast_size: int,
        forecast_size: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.degree = degree
        theta_size = degree + 1

        layers: List[nn.Module] = []
        in_features = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(_build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        self.fc_stack = nn.Sequential(*layers)
        self.theta_b = nn.Linear(in_features, theta_size)
        self.theta_f = nn.Linear(in_features, theta_size)

        backcast_grid = torch.linspace(-1.0, 0.0, backcast_size)
        forecast_grid = torch.linspace(0.0, 1.0, forecast_size)
        self.register_buffer("T_backcast", _polynomial_basis(backcast_grid, degree))
        self.register_buffer("T_forecast", _polynomial_basis(forecast_grid, degree))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc_stack(x)
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)
        backcast = theta_b @ self.T_backcast
        forecast = theta_f @ self.T_forecast
        return backcast, forecast


class SeasonalityBlock(nn.Module):
    """Interpretable seasonality block using Fourier basis."""

    def __init__(
        self,
        input_size: int,
        n_harmonics: int,
        hidden_size: int,
        n_layers: int,
        backcast_size: int,
        forecast_size: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics
        theta_size = 2 * n_harmonics + 1

        layers: List[nn.Module] = []
        in_features = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(_build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        self.fc_stack = nn.Sequential(*layers)
        self.theta_b = nn.Linear(in_features, theta_size)
        self.theta_f = nn.Linear(in_features, theta_size)

        backcast_grid = torch.linspace(-1.0, 0.0, backcast_size)
        forecast_grid = torch.linspace(0.0, 1.0, forecast_size)
        self.register_buffer("S_backcast", _fourier_basis(backcast_grid, n_harmonics))
        self.register_buffer("S_forecast", _fourier_basis(forecast_grid, n_harmonics))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc_stack(x)
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)
        backcast = theta_b @ self.S_backcast
        forecast = theta_f @ self.S_forecast
        return backcast, forecast


class NBeatsModel(nn.Module):
    """
    Full-scratch N-BEATS implementation.

    Supports two modes:
    - stack_type='generic': generic residual blocks.
    - stack_type='interpretable': trend + seasonality stacks.

    Input can be [B, T] or [B, T, F]. In the multivariate case, the tensor is
    flattened to [B, T*F] before being processed by N-BEATS blocks.
    """

    def __init__(
        self,
        input_size: int,
        forecast_size: int = 1,
        stack_type: str = "generic",
        n_stacks: int = 3,
        n_blocks_per_stack: int = 3,
        n_layers: int = 4,
        hidden_size: int = 256,
        theta_size: int = 64,
        dropout: float = 0.0,
        activation: str = "relu",
        trend_degree: int = 2,
        n_harmonics: int = 8,
        share_weights_in_stack: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.stack_type = stack_type
        self.n_stacks = n_stacks
        self.n_blocks_per_stack = n_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack

        self.stacks = nn.ModuleList()

        if stack_type == "generic":
            for _ in range(n_stacks):
                stack = self._build_generic_stack(
                    input_size=input_size,
                    forecast_size=forecast_size,
                    n_blocks=n_blocks_per_stack,
                    n_layers=n_layers,
                    hidden_size=hidden_size,
                    theta_size=theta_size,
                    dropout=dropout,
                    activation=activation,
                    share_weights=share_weights_in_stack,
                )
                self.stacks.append(stack)
        elif stack_type == "interpretable":
            trend_stack = self._build_interpretable_stack(
                block_cls=TrendBlock,
                block_kwargs=dict(degree=trend_degree),
                input_size=input_size,
                forecast_size=forecast_size,
                n_blocks=n_blocks_per_stack,
                n_layers=n_layers,
                hidden_size=hidden_size,
                dropout=dropout,
                activation=activation,
                share_weights=share_weights_in_stack,
            )
            seasonality_stack = self._build_interpretable_stack(
                block_cls=SeasonalityBlock,
                block_kwargs=dict(n_harmonics=n_harmonics),
                input_size=input_size,
                forecast_size=forecast_size,
                n_blocks=n_blocks_per_stack,
                n_layers=n_layers,
                hidden_size=hidden_size,
                dropout=dropout,
                activation=activation,
                share_weights=share_weights_in_stack,
            )
            self.stacks.extend([trend_stack, seasonality_stack])
        else:
            raise ValueError(f"Unsupported stack_type: {stack_type}")

    def _build_generic_stack(
        self,
        input_size: int,
        forecast_size: int,
        n_blocks: int,
        n_layers: int,
        hidden_size: int,
        theta_size: int,
        dropout: float,
        activation: str,
        share_weights: bool,
    ) -> nn.ModuleList:
        stack = nn.ModuleList()
        block = None
        for _ in range(n_blocks):
            if block is None or not share_weights:
                block = NBeatsBlock(
                    input_size=input_size,
                    theta_size=theta_size,
                    hidden_size=hidden_size,
                    n_layers=n_layers,
                    backcast_size=input_size,
                    forecast_size=forecast_size,
                    dropout=dropout,
                    activation=activation,
                )
            stack.append(block)
        return stack

    def _build_interpretable_stack(
        self,
        block_cls,
        block_kwargs: dict,
        input_size: int,
        forecast_size: int,
        n_blocks: int,
        n_layers: int,
        hidden_size: int,
        dropout: float,
        activation: str,
        share_weights: bool,
    ) -> nn.ModuleList:
        stack = nn.ModuleList()
        block = None
        for _ in range(n_blocks):
            if block is None or not share_weights:
                block = block_cls(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    n_layers=n_layers,
                    backcast_size=input_size,
                    forecast_size=forecast_size,
                    dropout=dropout,
                    activation=activation,
                    **block_kwargs,
                )
            stack.append(block)
        return stack

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        elif x.dim() != 2:
            raise ValueError(f"Expected 2D or 3D input, got shape {tuple(x.shape)}")

        residuals = x
        forecast = torch.zeros(x.size(0), self.forecast_size, device=x.device, dtype=x.dtype)

        for stack in self.stacks:
            for block in stack:
                backcast, block_forecast = block(residuals)
                residuals = residuals - backcast
                forecast = forecast + block_forecast
        return forecast


class NBeatsRegressor(nn.Module):
    """Thin task head wrapper for regression."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.backbone = NBeatsModel(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


class NBeatsClassifier(nn.Module):
    """Binary classifier using N-BEATS backbone and a logit output."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        kwargs = dict(kwargs)
        kwargs["forecast_size"] = 1
        self.backbone = NBeatsModel(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


def _build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "selu":
        return nn.SELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unsupported activation: {name}")


def _polynomial_basis(grid: torch.Tensor, degree: int) -> torch.Tensor:
    basis = [grid ** i for i in range(degree + 1)]
    return torch.stack(basis, dim=0)


def _fourier_basis(grid: torch.Tensor, n_harmonics: int) -> torch.Tensor:
    basis = [torch.ones_like(grid)]
    for i in range(1, n_harmonics + 1):
        basis.append(torch.cos(2 * math.pi * i * grid))
        basis.append(torch.sin(2 * math.pi * i * grid))
    return torch.stack(basis, dim=0)
