# Copyright (c) 2026 LightSeek Foundation

from types import SimpleNamespace

import torch
import torch.nn as nn

from torchspec.models.target_hidden import capture_final_norm_boundary


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden):
        return hidden + self.linear(hidden)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(_Block() for _ in range(2))
        self.norm = nn.RMSNorm(4)

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


def test_capture_returns_both_sides_of_the_final_norm():
    decoder = _Decoder()
    inputs = torch.randn(1, 3, 4)

    boundary = capture_final_norm_boundary(decoder, hidden=inputs)

    with torch.no_grad():
        pre_norm = decoder.layers[1](decoder.layers[0](inputs))
    torch.testing.assert_close(boundary.pre_norm, pre_norm, rtol=0, atol=0)
    torch.testing.assert_close(boundary.post_norm, decoder.norm(pre_norm), rtol=0, atol=0)
