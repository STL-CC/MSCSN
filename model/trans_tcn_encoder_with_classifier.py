import math

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


class Chomp1d(nn.Module):
    """Trim right padding to keep causal behavior."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Temporal convolution block with residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


class TemporalTransformerEncoderLayer(nn.Module):
    """Transformer attention plus temporal convolution feed-forward substitute."""

    def __init__(
        self,
        d_model=64,
        n_heads=4,
        tcn_filters=64,
        kernel_size=3,
        dilation=1,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.temporal_block = TemporalBlock(
            in_channels=d_model,
            out_channels=tcn_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout,
        )

        self.dim_adapter = (
            nn.Linear(tcn_filters, d_model) if tcn_filters != d_model else None
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + self.dropout(attn_output))

        tcn_input = src.permute(0, 2, 1)
        tcn_output = self.temporal_block(tcn_input)

        if self.dim_adapter is not None:
            tcn_output = self.dim_adapter(tcn_output.permute(0, 2, 1)).permute(0, 2, 1)

        tcn_output = tcn_output.permute(0, 2, 1)
        src = self.norm2(src + self.dropout(tcn_output))
        return src


class DynamicPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding generated dynamically by sequence length."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        position = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        return self.dropout(x + pe)


class TransTCNEncoder(nn.Module):
    """Paper-aligned encoder: attention with temporal convolution integration."""

    def __init__(
        self,
        in_chans=1,
        d_model=64,
        n_heads=4,
        kernel_size=3,
        depth=3,
        hidden_size=64,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_chans, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        self.positional_encoding = DynamicPositionalEncoding(d_model, dropout)

        self.encoder_layers = nn.ModuleList(
            [
                TemporalTransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    tcn_filters=d_model,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(depth)
            ]
        )

        self.feature_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.permute(0, 2, 1)
        return self.feature_pool(x)


class TransTCNEncoderWithClassifier(nn.Module):
    """Stage 1 model with paper-aligned encoder and MLP classifier head."""

    def __init__(self, args, _unused_encoder_cls=None):
        super().__init__()
        self.encoder = TransTCNEncoder(
            in_chans=args.in_channels,
            d_model=args.out_channels,
            hidden_size=args.out_channels,
            depth=args.depth,
            kernel_size=args.kernel_size,
        ).to(args.device)

        self.classifier = nn.Sequential(
            nn.Linear(args.out_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, args.num_classes),
        ).to(args.device)

    def forward(self, x):
        encoded = self.encoder(x)
        logits = self.classifier(encoded)
        return logits, encoded
