import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Left-padded causal Conv1d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """
    Standard TCN residual block:
    causal conv -> norm -> activation -> dropout -> causal conv -> norm -> residual add
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_weight_norm: bool = False,
        norm_type: str = "group",
    ):
        super().__init__()

        conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)

        if use_weight_norm:
            conv1.conv = nn.utils.weight_norm(conv1.conv)
            conv2.conv = nn.utils.weight_norm(conv2.conv)

        self.conv1 = conv1
        self.conv2 = conv2
        self.norm1 = self._build_norm(out_channels, norm_type)
        self.norm2 = self._build_norm(out_channels, norm_type)
        self.act = self._build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.reset_parameters()

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        name = (name or "relu").lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name in {"silu", "swish"}:
            return nn.SiLU()
        raise ValueError(f"Unsupported activation: {name}")

    @staticmethod
    def _build_norm(channels: int, norm_type: str) -> nn.Module:
        norm_type = (norm_type or "group").lower()
        if norm_type == "none":
            return nn.Identity()
        if norm_type == "batch":
            return nn.BatchNorm1d(channels)
        if norm_type == "layer":
            return nn.GroupNorm(1, channels)
        if norm_type == "group":
            return nn.GroupNorm(1, channels)
        raise ValueError(f"Unsupported norm_type: {norm_type}")

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.dropout(out)

        return out + residual


class TCN(nn.Module):
    """
    Full-scratch Temporal Convolutional Network for sequence-to-one prediction.

    Input:  [batch, seq_len, input_dim]
    Output: [batch, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        num_channels=None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_weight_norm: bool = False,
        norm_type: str = "group",
        pooling: str = "last",
        head_hidden_dim: int = 0,
        head_dropout: float = 0.0,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if num_channels is None:
            num_channels = [64, 64, 64]
        if not isinstance(num_channels, (list, tuple)) or len(num_channels) == 0:
            raise ValueError("num_channels must be a non-empty list or tuple.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_channels = list(num_channels)
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.pooling = pooling.lower()

        layers = []
        in_channels = input_dim
        for i, out_channels in enumerate(self.num_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                    use_weight_norm=use_weight_norm,
                    norm_type=norm_type,
                )
            )
            in_channels = out_channels
        self.backbone = nn.Sequential(*layers)

        head_layers = []
        if head_hidden_dim and head_hidden_dim > 0:
            head_layers.extend([
                nn.Linear(in_channels, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, output_dim),
            ])
        else:
            head_layers.append(nn.Linear(in_channels, output_dim))
        self.head = nn.Sequential(*head_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        if self.pooling == "last":
            return x[:, :, -1]
        if self.pooling == "mean":
            return x.mean(dim=-1)
        if self.pooling == "max":
            return x.max(dim=-1).values
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TCN expects 3D input [B, T, F], but got shape={tuple(x.shape)}")

        # [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2).contiguous()
        x = self.backbone(x)
        x = self._pool(x)
        x = self.head(x)
        return x
