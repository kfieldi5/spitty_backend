from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


LABELS = ("kick", "snare", "hihat_closed", "hihat_open", "rest")


@dataclass(frozen=True)
class Config:
    sample_rate: int = 16_000
    clip_samples: int = 5_600
    pre_onset_samples: int = 1_600
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 64
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 12
    seed: int = 2026


class BeatboxClassifier(nn.Module):
    """Checkpoint-compatible copy of the mobile classifier architecture."""

    def __init__(self, config: Config, classes: int = len(LABELS)):
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.register_buffer("window", torch.hann_window(config.n_fft))
        self.register_buffer(
            "mel_filter",
            torch.zeros(config.n_mels, config.n_fft // 2 + 1),
        )
        self.features = nn.Sequential(
            self._block(1, 16, stride=(2, 1)),
            self._block(16, 32, stride=(2, 2)),
            self._block(32, 48, stride=(2, 2)),
            self._block(48, 64, stride=(2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(64, classes),
        )

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )


class FeatureClassifier(nn.Module):
    def __init__(self, source: BeatboxClassifier):
        super().__init__()
        self.features = source.features
        self.classifier = source.classifier

    def forward(self, normalized_log_mel: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(normalized_log_mel))
