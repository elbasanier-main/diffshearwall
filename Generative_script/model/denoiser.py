"""
disc_diffusion/model/denoiser.py
==================================
Stage B: Conditional Discrete Diffusion Denoiser.

Unified architecture - no separate mask lookup needed.
MaskPredictor is integrated directly into WallSlotDenoiser.

Data flow (single forward pass):
    condition [B, 5]
        -> MaskPredictor
        -> freq_map [B, 2, GRID, GRID]   (learned slot probability for any layout)

    [noisy_occ, must_on, variable, freq_map] -> 6-channel input
        -> U-Net denoiser (FiLM conditioned)
        -> logits [B, 2, F, GRID, GRID]

Input channels:
    0: noisy H-slot occupancy
    1: noisy V-slot occupancy
    2: must_on H mask
    3: must_on V mask
    4: variable region indicator
    5: predicted frequency map (H | V merged, from MaskPredictor)

MaskPredictor also provides the analytical boundary mask for unseen layouts,
so no lookup table is needed at inference time.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID       = 31
MAX_FLOORS = 20


# -----------------------------------------------------------------------
# Sinusoidal timestep embedding
# -----------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


# -----------------------------------------------------------------------
# MaskPredictor: condition -> frequency map + analytical boundary mask
# -----------------------------------------------------------------------

class MaskPredictor(nn.Module):
    """
    Predicts slot frequency map from layout conditions.
    Replaces MaskBuilder lookup for unseen layouts.

    Trained jointly with denoiser using MSE loss against
    actual frequency maps from training data.

    Also computes analytical boundary masks (must_on, must_off)
    from geometry - these are always correct for any layout.

    Args:
        cond_dim: condition vector size (default 5)
        hidden:   MLP hidden size (default 256)
    """

    SLOTS_PER_BAY = 3

    def __init__(self, cond_dim: int = 5, hidden: int = 256):
        super().__init__()

        # MLP: condition -> flat frequency map
        # Output: 2 * GRID * GRID values in [0,1]
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2 * GRID * GRID),
            nn.Sigmoid(),   # frequency in [0, 1]
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cond: [B, cond_dim]
        Returns:
            freq_map: [B, 2, GRID, GRID]  predicted slot frequencies
        """
        B = cond.size(0)
        out = self.mlp(cond)                      # [B, 2*GRID*GRID]
        return out.view(B, 2, GRID, GRID)         # [B, 2, GRID, GRID]

    @staticmethod
    def analytical_masks(
        lx: int,
        ly: int,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """
        Compute must_on, must_off, variable masks analytically
        for any layout (lx, ly). No lookup needed.

        Returns dict with must_on, must_off, variable: [2, GRID, GRID] bool
        """
        S = MaskPredictor.SLOTS_PER_BAY

        must_on  = torch.zeros(2, GRID, GRID, dtype=torch.bool, device=device)
        must_off = torch.ones(2,  GRID, GRID, dtype=torch.bool, device=device)

        # Valid H-slots: xi in [0, lx*3), yi in [0, ly]
        hx = min(lx * S, GRID)
        hy = min(ly + 1, GRID)
        must_off[0, :hx, :hy] = False

        # Valid V-slots: xi in [0, lx], yi in [0, ly*3)
        vx = min(lx + 1, GRID)
        vy = min(ly * S, GRID)
        must_off[1, :vx, :vy] = False

        # 8 fixed corner walls (always present in all layouts)
        # H corners: (xi=0, yi=0), (xi=lx*3-1, yi=0),
        #            (xi=0, yi=ly), (xi=lx*3-1, yi=ly)
        for xi in [0, min(lx * S - 1, GRID - 1)]:
            for yi in [0, min(ly, GRID - 1)]:
                must_on[0, xi, yi]  = True
                must_off[0, xi, yi] = False

        # V corners: (xi=0, yi=0), (xi=lx, yi=0),
        #            (xi=0, yi=ly*3-1), (xi=lx, yi=ly*3-1)
        for xi in [0, min(lx, GRID - 1)]:
            for yi in [0, min(ly * S - 1, GRID - 1)]:
                must_on[1, xi, yi]  = True
                must_off[1, xi, yi] = False

        variable = ~must_on & ~must_off

        return {
            "must_on":  must_on,
            "must_off": must_off,
            "variable": variable,
        }

    @staticmethod
    def batch_analytical_masks(
        lx_list: List[int],
        ly_list: List[int],
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """
        Compute masks for a batch of layouts.

        Returns dict with must_on, must_off, variable: [B, 2, GRID, GRID] bool
        """
        B = len(lx_list)
        must_on_b  = torch.zeros(B, 2, GRID, GRID, dtype=torch.bool, device=device)
        must_off_b = torch.zeros(B, 2, GRID, GRID, dtype=torch.bool, device=device)
        variable_b = torch.zeros(B, 2, GRID, GRID, dtype=torch.bool, device=device)

        for i, (lx, ly) in enumerate(zip(lx_list, ly_list)):
            m = MaskPredictor.analytical_masks(lx, ly, device)
            must_on_b[i]  = m["must_on"]
            must_off_b[i] = m["must_off"]
            variable_b[i] = m["variable"]

        return {
            "must_on":  must_on_b,
            "must_off": must_off_b,
            "variable": variable_b,
        }


# -----------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------

class FiLMLayer(nn.Module):
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feature_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias[:feature_dim])
        nn.init.zeros_(self.proj.bias[feature_dim:])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        sc = self.proj(cond)
        scale, shift = sc.chunk(2, dim=1)
        for _ in range(x.dim() - 2):
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        return x * (1 + scale) + shift


class ResBlock3D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, groups: int = 8):
        super().__init__()
        g = min(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(g, channels)
        self.norm2 = nn.GroupNorm(g, channels)
        self.act   = nn.SiLU()
        self.film  = FiLMLayer(channels, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.film(h, cond)
        h = self.norm2(self.conv2(h))
        return self.act(h + x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.res  = ResBlock3D(in_ch, cond_dim)
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.res(x, cond)
        return self.act(self.norm(self.down(x)))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.res  = ResBlock3D(in_ch + skip_ch, cond_dim)
        self.proj = nn.Conv3d(in_ch + skip_ch, out_ch, 1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act  = nn.SiLU()

    def forward(self, x, skip, cond):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, cond)
        return self.act(self.norm(self.proj(x)))


class AxialAttention3D(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.attn_f = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.attn_x = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.attn_y = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_f = nn.LayerNorm(channels)
        self.norm_x = nn.LayerNorm(channels)
        self.norm_y = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, H, W = x.shape

        xf = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, F, C)
        xf, _ = self.attn_f(xf, xf, xf)
        xf = self.norm_f(xf).reshape(B, H, W, F, C).permute(0, 4, 3, 1, 2)
        x  = x + xf

        xx = x.permute(0, 2, 4, 3, 1).reshape(B * F * W, H, C)
        xx, _ = self.attn_x(xx, xx, xx)
        xx = self.norm_x(xx).reshape(B, F, W, H, C).permute(0, 4, 1, 3, 2)
        x  = x + xx

        xy = x.permute(0, 2, 3, 4, 1).reshape(B * F * H, W, C)
        xy, _ = self.attn_y(xy, xy, xy)
        xy = self.norm_y(xy).reshape(B, F, H, W, C).permute(0, 4, 1, 2, 3)
        x  = x + xy

        return x


# -----------------------------------------------------------------------
# Main Denoiser (unified with MaskPredictor)
# -----------------------------------------------------------------------

class WallSlotDenoiser(nn.Module):
    """
    Unified denoiser with integrated MaskPredictor.

    No external mask lookup needed - MaskPredictor learns frequency
    maps from condition [lx, ly, floors, shear_ratio].

    Boundary masks (must_on, must_off) are computed analytically
    from geometry, always correct for any layout.

    Args:
        in_channels:    6 (H, V, must_on_H, must_on_V, variable, freq_map)
        base_ch:        base channel count
        cond_in_dim:    condition vector size
        cond_embed_dim: FiLM embedding size
        t_embed_dim:    timestep embedding size
    """

    def __init__(
        self,
        in_channels:    int = 6,       # 5 original + 1 freq_map channel
        base_ch:        int = 32,
        cond_in_dim:    int = 5,
        cond_embed_dim: int = 128,
        t_embed_dim:    int = 128,
        max_floors:     int = MAX_FLOORS,
    ):
        super().__init__()

        self.max_floors = max_floors
        cond_total      = cond_embed_dim + t_embed_dim

        # --- Integrated MaskPredictor ---
        self.mask_predictor = MaskPredictor(cond_in_dim, hidden=256)

        # --- Timestep + condition encoders ---
        self.t_embed = TimestepEmbedding(t_embed_dim)
        self.c_embed = nn.Sequential(
            nn.Linear(cond_in_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )

        # --- Stem (now 6 input channels) ---
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
        )

        # --- U-Net backbone ---
        ch = base_ch
        self.down1 = DownBlock(ch,     ch * 2, cond_total)
        self.down2 = DownBlock(ch * 2, ch * 4, cond_total)
        self.down3 = DownBlock(ch * 4, ch * 8, cond_total)

        bn_ch = ch * 8
        self.bn_res1 = ResBlock3D(bn_ch, cond_total)
        self.bn_attn = AxialAttention3D(bn_ch, heads=4)
        self.bn_res2 = ResBlock3D(bn_ch, cond_total)

        self.up3 = UpBlock(bn_ch,  ch * 4, ch * 4, cond_total)
        self.up2 = UpBlock(ch * 4, ch * 2, ch * 2, cond_total)
        self.up1 = UpBlock(ch * 2, ch,     ch,     cond_total)

        # --- Output head ---
        self.out_head = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv3d(ch, 2, 1),
        )
        # Init bias: logit=-0.2 reflects ~45% active rate
        nn.init.constant_(self.out_head[-1].bias, -0.2)
        nn.init.normal_(self.out_head[-1].weight, std=0.01)

    def predict_freq_map(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Predict slot frequency map from condition.
        Used at inference for unseen layouts.

        Returns: [B, 2, GRID, GRID]
        """
        return self.mask_predictor(cond)

    def forward(
        self,
        x_t:  torch.Tensor,
        t:    torch.Tensor,
        cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass.

        Args:
            x_t:  [B, 6, F, GRID, GRID]  6-channel input (built by build_input)
            t:    [B]                      timestep
            cond: [B, cond_in_dim]         condition vector

        Returns:
            dict with:
                logits:   [B, 2, F, GRID, GRID]
                freq_map: [B, 2, GRID, GRID]  (for loss computation)
        """
        # Encode condition + timestep
        t_emb   = self.t_embed(t)
        c_emb   = self.c_embed(cond)
        film_v  = torch.cat([c_emb, t_emb], dim=-1)

        # Predict frequency map (also returned for loss)
        freq_map = self.mask_predictor(cond)   # [B, 2, GRID, GRID]

        # U-Net
        h  = self.stem(x_t)
        s1 = h
        h  = self.down1(h,  film_v)
        s2 = h
        h  = self.down2(h,  film_v)
        s3 = h
        h  = self.down3(h,  film_v)
        h  = self.bn_res1(h, film_v)
        h  = self.bn_attn(h)
        h  = self.bn_res2(h, film_v)
        h  = self.up3(h, s3, film_v)
        h  = self.up2(h, s2, film_v)
        h  = self.up1(h, s1, film_v)

        logits = self.out_head(h)

        return {
            "logits":   logits,
            "freq_map": freq_map,
        }

    def build_input(
        self,
        x_t:      torch.Tensor,
        must_on:  torch.Tensor,
        variable: torch.Tensor,
        freq_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assemble 6-channel input tensor.

        Args:
            x_t:      [B, 2, F, GRID, GRID]  noisy occupancy
            must_on:  [B, 2, GRID, GRID]
            variable: [B, 2, GRID, GRID]
            freq_map: [B, 2, GRID, GRID]      from MaskPredictor

        Returns:
            [B, 6, F, GRID, GRID]
        """
        B, _, F, G, _ = x_t.shape

        def exp(m):
            return m.unsqueeze(2).expand(B, -1, F, G, G).float()

        # Variable merged (H | V) as single channel
        var_merged = (variable[:, 0] | variable[:, 1]).float()
        var_ch     = var_merged.unsqueeze(1).unsqueeze(2).expand(B, 1, F, G, G)

        # Freq map merged (mean of H and V) as single channel
        freq_merged = freq_map.mean(dim=1, keepdim=True)               # [B, 1, GRID, GRID]
        freq_ch     = freq_merged.unsqueeze(2).expand(B, 1, F, G, G)   # [B, 1, F, GRID, GRID]

        inp = torch.cat([
            x_t,          # [B, 2, F, G, G]  ch 0,1
            exp(must_on), # [B, 2, F, G, G]  ch 2,3
            var_ch,       # [B, 1, F, G, G]  ch 4
            freq_ch,      # [B, 1, F, G, G]  ch 5  <- predicted frequency
        ], dim=1)         # [B, 6, F, G, G]

        return inp