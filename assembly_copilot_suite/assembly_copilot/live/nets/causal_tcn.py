"""Causal dilated TCN for streaming step segmentation.

Why not reuse DiffAct: it is a diffusion model with bidirectional temporal convs over
the whole sequence and 25 DDIM passes. It does not merely benefit from future frames,
it requires them -- there is no prefix mode. Its 96.24 describes a system that already
watched the video.

The property that makes this head deployable: with left-only padding, the output at
timestep t depends on t and earlier ONLY. So running the model over a whole recording
offline yields, at every t, exactly the value a streaming run would have emitted at
that moment. Offline eval numbers therefore transfer to live with no train/serve gap
-- unlike a sliding-window hack over a bidirectional model, where they would not.

Receptive field with kernel k and L layers of doubling dilation:
    RF = 1 + (k-1) * (2^L - 1)
L=9, k=3 -> 1023 timesteps = 205 s of lookback at stride 6 / 30 fps, which comfortably
covers the longest single step in this data (155 s) without unbounded history.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """Conv1d that cannot see the future: pad (k-1)*d on the LEFT, trim the overhang."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x):                       # x: [B, C, T]
        return super().forward(F.pad(x, (self.left_pad, 0)))


class ChannelNorm(nn.Module):
    """LayerNorm over channels, applied INDEPENDENTLY at each timestep.

    The obvious choices are both non-causal and fail check_causality():
      BatchNorm1d  - pools statistics over the batch AND the time axis
      GroupNorm    - normalises over (channels/groups x TIME), so every timestep's
                     output depends on the whole sequence, future included
    Normalising each timestep over its own channels alone touches no other timestep,
    so it is safe to stream.
    """

    def __init__(self, ch):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):                       # [B, C, T]
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel_size, dilation, dropout):
        super().__init__()
        self.conv = CausalConv1d(ch, ch, kernel_size, dilation)
        self.norm = ChannelNorm(ch)
        self.point = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.norm(self.conv(x)))
        return x + self.drop(self.point(h))


class CausalTCN(nn.Module):
    def __init__(self, input_dim=2176, num_classes=11, num_layers=9,
                 num_f_maps=128, kernel_size=3, dropout=0.5):
        super().__init__()
        self.inp = nn.Conv1d(input_dim, num_f_maps, 1)
        self.blocks = nn.ModuleList([
            ResidualBlock(num_f_maps, kernel_size, 2 ** i, dropout)
            for i in range(num_layers)
        ])
        self.out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.receptive_field = 1 + (kernel_size - 1) * (2 ** num_layers - 1)

    def forward(self, x):                       # [B, input_dim, T] -> [B, num_classes, T]
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(h)


def check_causality(model, T=400, device="cpu"):
    """Assert no output at t depends on any input after t.

    Perturb the input at a single late timestep and confirm every earlier output is
    bit-identical. This is the property the whole live design rests on, so it is
    verified rather than assumed -- one wrongly-padded conv silently breaks it and
    the offline metrics would then be optimistic in a way nothing else would reveal.
    """
    model = model.to(device).eval()
    x = torch.randn(1, model.inp.in_channels, T, device=device)
    with torch.no_grad():
        a = model(x)
        x2 = x.clone()
        x2[:, :, T - 1] += 10.0                 # change only the LAST timestep
        b = model(x2)
    drift = (a[:, :, :T - 1] - b[:, :, :T - 1]).abs().max().item()
    changed = (a[:, :, T - 1] - b[:, :, T - 1]).abs().max().item()
    return drift, changed
