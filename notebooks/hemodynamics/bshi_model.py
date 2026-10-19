"""BSHI network: a GroupNorm ResNet over the (beat x phase) image, an MLP over the context vector, an optional SQI quality gate, and three task heads.

Design decisions:
* GroupNorm everywhere, no BatchNorm: the output for a window does not depend on the other windows in the batch, so two GPUs with different per-rank batch sizes see no statistics mismatch.
* ``ModelConfig.variant`` selects the input recipe (``bshi.VARIANTS``). Groups that are off are multiplied by an all-zero mask BEFORE the first layer (and ``map_only`` skips the image
  trunk altogether), so the network cannot read them even if the data pipeline forgot to zero them.
* SQI gate ("SQI-based fusion weighting"): per beat and per modality (ECG, ART, PPG) a weight ``2*sigmoid(z)`` from the four SQI bands and the three beat masks; it scales the three
  waveform planes before the trunk. The last layer starts at zero, so training starts from weight 1 (no gate effect) and the gate can only learn to down-weight or up-weight.
* Heads: event logit; delta MAP (regression on a train-standardized target); AUT65 in two parts (logit of AUT65 > 0 and a log1p magnitude, trained only where AUT65 > 0).
"""
from __future__ import annotations
# Combine a beat-image encoder with numerical context and quality weighting.

from dataclasses import asdict, dataclass

import torch
from torch import nn

from . import bshi as S

N_CONTEXT = len(S.CONTEXT_NAMES)


@dataclass
class ModelConfig:
    """Configure the active input recipe, residual encoder widths, and fusion layers."""
    variant: str = "eap"
    gate: bool = True
    widths: tuple = (32, 64, 128, 256)
    blocks: tuple = (2, 2, 2, 2)
    groups: int = 8
    ctx_hidden: int = 64
    head_hidden: int = 128
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


def input_masks(variant):
    """(image mask (1,7,1,200), context mask (25,)) of the groups the variant keeps: planes 0-3 whole, planes 4-6 band by band (timing 50 columns, morphology 25, SQI 50)."""
    # Keep each enabled waveform, band, and context field in the selected input recipe.
    off = S.VARIANTS[variant]
    m = torch.zeros(1, 7, 1, S.PHASE)
    for g, plane in S.PLANE_OF_GROUP.items():
        if g not in off:
            m[:, plane] = 1.0
    for plane, names, width in ((4, S.TIMING_BANDS, 50), (5, S.MORPH_BANDS, 25), (6, S.SQI_BANDS, 50)):
        for k, name in enumerate(names):
            if name not in off:
                m[:, plane, :, k * width:(k + 1) * width] = 1.0
    cm = torch.ones(N_CONTEXT)
    for name in S.context_off(off):
        cm[S.CONTEXT_NAMES.index(name)] = 0.0
    return m, cm


class SqiGate(nn.Module):
    """Convert per-beat quality bands and validity masks into modality fusion weights."""
    def __init__(self, hidden=16):
        # Start all quality weights at one so training learns any later up- or down-weighting.
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(7, hidden, 3, padding=1), nn.SiLU(), nn.Conv1d(hidden, 3, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, beat_mask):
        sqi = x[:, 6, :, ::50]                                            # (B,60,4): the first column of each SQI band (every column of a band is the same value)
        feats = torch.cat([sqi.transpose(1, 2), beat_mask.to(x.dtype)], dim=1)       # (B,7,60)
        return 2.0 * torch.sigmoid(self.net(feats))                                 # (B,3,60), 1.0 at initialization


class Block(nn.Module):
    """Apply a residual convolutional block with GroupNorm and an optional projection."""
    def __init__(self, cin, cout, stride, groups):
        # Use a projection only when the residual input and output shapes differ.
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(groups, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(groups, cout)
        self.skip = None if stride == 1 and cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(groups, cout))
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.n2(self.c2(self.act(self.n1(self.c1(x)))))
        return self.act(y + (x if self.skip is None else self.skip(x)))


class BshiNet(nn.Module):
    """Fuse beat-image and context encoders to predict an event, pressure drop, and burden."""
    def __init__(self, cfg: ModelConfig | None = None):
        # Build the encoders for the selected recipe, including the context-only MAP baseline.
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        mask, ctx_mask = input_masks(c.variant)
        self.register_buffer("image_mask", mask, persistent=False)
        self.register_buffer("context_mask", ctx_mask, persistent=False)
        self.use_trunk = bool(mask.sum() > 0)
        self.gate = SqiGate() if (c.gate and self.use_trunk and (mask[:, :3].sum() > 0)) else None
        if self.use_trunk:
            # Extract local beat/phase patterns before the deeper residual blocks.
            self.stem = nn.Sequential(nn.Conv2d(7, c.widths[0], (3, 7), (1, 2), (1, 3), bias=False), nn.GroupNorm(c.groups, c.widths[0]), nn.SiLU())
            layers, cin = [], c.widths[0]
            for si, (w, nb) in enumerate(zip(c.widths, c.blocks)):
                for bi in range(nb):
                    layers.append(Block(cin, w, 2 if (si > 0 and bi == 0) else 1, c.groups))
                    cin = w
            self.trunk = nn.Sequential(*layers)
            img_dim = c.widths[-1]
        else:
            # The MAP-only recipe skips image features and uses numerical context alone.
            img_dim = 0
        self.ctx = nn.Sequential(nn.Linear(N_CONTEXT, c.ctx_hidden), nn.SiLU(), nn.Dropout(c.dropout), nn.Linear(c.ctx_hidden, c.ctx_hidden), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(img_dim + c.ctx_hidden, c.head_hidden), nn.SiLU(), nn.Dropout(c.dropout))
        self.h_event, self.h_delta, self.h_aut_pos, self.h_aut_mag = (nn.Linear(c.head_hidden, 1) for _ in range(4))

    def forward(self, x, context, beat_mask):
        """x (B,7,60,200), context (B,25) standardized, beat_mask (B,3,60) bool -> dict of (B,) tensors: event, delta_map, aut_logit, aut_mag."""
        # Encode the enabled image and context inputs before combining their predictions.
        feats = []
        if self.use_trunk:
            x = x * self.image_mask.to(x.dtype)
            if self.gate is not None:
                w = self.gate(x, beat_mask)
                x = torch.cat([x[:, :3] * w.unsqueeze(-1), x[:, 3:]], dim=1)
            feats.append(self.trunk(self.stem(x)).mean(dim=(2, 3)))
        # Apply the context mask too, so a disabled feature cannot return through this branch.
        feats.append(self.ctx(context * self.context_mask.to(context.dtype)))
        h = self.fuse(torch.cat(feats, dim=1))
        return {"event": self.h_event(h).squeeze(1), "delta_map": self.h_delta(h).squeeze(1), "aut_logit": self.h_aut_pos(h).squeeze(1), "aut_mag": self.h_aut_mag(h).squeeze(1)}

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
