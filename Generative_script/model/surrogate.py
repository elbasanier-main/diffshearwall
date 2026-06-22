"""
disc_diffusion/model/surrogate.py
===================================
Stage D: Surrogate ranking model for the discrete diffusion pipeline.

Inputs:
    occ:  [B, 2, F, GRID, GRID]  projected binary occupancy
    cond: [B, 5]   [lx/10, ly/10, num_floors/20, shear_ratio_target, 0]

Outputs:
    x_dir_pred, y_dir_pred      supervised on dataset labels
    symmetry, construct         pseudo-labels (geometric computation)
    shear_ratio_penalty         deviation from target shear ratio
    ranking_score               scalar (higher = better)

Score weights:
    x_dir          : 0.35
    y_dir          : 0.30
    shear_ratio    : 0.20
    symmetry       : 0.10
    construct      : 0.05
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.slot_repr import GRID, MAX_FLOORS, SLOTS_PER_BAY


# -----------------------------------------------------------------------
# Surrogate model
# -----------------------------------------------------------------------

class StructuralSurrogate(nn.Module):
    """
    Predicts structural quality metrics from discrete occupancy tensor.

    Ranking score is based on:
        - X-Dir drift       (supervised, lower = better)
        - Y-Dir drift       (supervised, lower = better)
        - Shear ratio error (deviation from target, lower = better)
        - Symmetry penalty  (geometric pseudo-label, lower = better)
        - Constructability  (geometric pseudo-label, lower = better)
    """

    SCORE_WEIGHTS = {
        "x_dir":       0.35,
        "y_dir":       0.30,
        "shear_ratio": 0.20,
        "symmetry":    0.10,
        "construct":   0.05,
    }

    def __init__(
        self,
        cond_dim:      int   = 5,
        base_ch:       int   = 32,
        cond_embed:    int   = 64,
        drift_scale_x: float = 100.0,
        drift_scale_y: float = 50.0,
    ):
        super().__init__()
        self.drift_scale_x = drift_scale_x
        self.drift_scale_y = drift_scale_y

        # Condition encoder
        self.cond_enc = nn.Sequential(
            nn.Linear(cond_dim, cond_embed), nn.SiLU(),
            nn.Linear(cond_embed, cond_embed),
        )

        # 3D CNN backbone: stride (1,2,2) halves X,Y each level, keeps F
        ch = base_ch
        self.net = nn.Sequential(
            nn.Conv3d(2,     ch,     kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, ch),     ch),     nn.SiLU(),
            nn.Conv3d(ch,    ch * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(min(8, ch * 2), ch * 2), nn.SiLU(),
            nn.Conv3d(ch*2,  ch * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(min(8, ch * 4), ch * 4), nn.SiLU(),
            nn.Conv3d(ch*4,  ch * 8, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(min(8, ch * 8), ch * 8), nn.SiLU(),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        trunk_in = ch * 8 + cond_embed
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),      nn.LayerNorm(128), nn.SiLU(),
        )

        def head(activate=False):
            layers = [nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1)]
            if activate:
                layers.append(nn.Sigmoid())
            return nn.Sequential(*layers)

        # Drift heads (raw mm values, no activation)
        self.h_xdir      = head(activate=False)
        self.h_ydir      = head(activate=False)

        # Geometric metric heads ([0,1] range)
        self.h_symmetry  = head(activate=True)
        self.h_construct = head(activate=True)

        # Bias init: start near typical dataset values
        nn.init.constant_(self.h_xdir[-1].bias, 45.0)
        nn.init.constant_(self.h_ydir[-1].bias, 23.0)

    def forward(
        self,
        occ:  torch.Tensor,
        cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            occ:  [B, 2, F, GRID, GRID]  binary occupancy
            cond: [B, 5]  condition vector
                    cond[:, 0] = lx / 10
                    cond[:, 1] = ly / 10
                    cond[:, 2] = num_floors / 20
                    cond[:, 3] = target shear ratio
                    cond[:, 4] = reserved (0)

        Returns:
            dict with all metrics and ranking_score
        """
        B = occ.size(0)

        # Feature extraction
        ce   = self.cond_enc(cond)                     # [B, cond_embed]
        feat = self.pool(self.net(occ)).flatten(1)     # [B, ch*8]
        h    = self.trunk(torch.cat([feat, ce], dim=-1))  # [B, 128]

        # Drift predictions
        xd = self.h_xdir(h)       # [B, 1]  mm
        yd = self.h_ydir(h)       # [B, 1]  mm

        # Geometric metrics
        sym = self.h_symmetry(h)   # [B, 1]  [0,1]
        con = self.h_construct(h)  # [B, 1]  [0,1]

        # Shear ratio penalty: compare actual wall count vs target
        # actual ratio = active slots / max possible slots for this layout
        lx_scaled = cond[:, 0] * 10.0   # [B]  recover lx
        ly_scaled = cond[:, 1] * 10.0   # [B]  recover ly
        target_ratio = cond[:, 3]        # [B]

        # Count active slots (sum over all dims except batch)
        active_walls = occ.float().sum(dim=(1, 2, 3, 4))   # [B]

        # Max possible slots for each sample
        max_slots = (
            (ly_scaled + 1) * lx_scaled * SLOTS_PER_BAY
          + (lx_scaled + 1) * ly_scaled * SLOTS_PER_BAY
        ).clamp(min=1)  # [B]

        # Average over floors
        num_floors = cond[:, 2] * 20.0  # [B]
        actual_ratio = active_walls / (max_slots * num_floors.clamp(min=1))  # [B]

        # Penalty: absolute deviation from target, normalized to [0,1]
        # max possible deviation = 1.0
        shear_penalty = (actual_ratio - target_ratio).abs().clamp(0, 1)  # [B]
        shear_penalty = shear_penalty.unsqueeze(1)                        # [B, 1]

        # Normalize drift to [0,1] for ranking
        xn = xd.clamp(0, self.drift_scale_x * 2) / (self.drift_scale_x * 2)
        yn = yd.clamp(0, self.drift_scale_y * 2) / (self.drift_scale_y * 2)

        # Composite quality score (lower = better)
        quality = (
            self.SCORE_WEIGHTS["x_dir"]       * xn
          + self.SCORE_WEIGHTS["y_dir"]       * yn
          + self.SCORE_WEIGHTS["shear_ratio"] * shear_penalty
          + self.SCORE_WEIGHTS["symmetry"]    * sym
          + self.SCORE_WEIGHTS["construct"]   * con
        )  # [B, 1]

        return {
            "x_dir_pred":      xd,              # [B, 1]  mm
            "y_dir_pred":      yd,              # [B, 1]  mm
            "shear_penalty":   shear_penalty,   # [B, 1]  deviation from target
            "actual_ratio":    actual_ratio.unsqueeze(1),  # [B, 1]
            "symmetry":        sym,             # [B, 1]  [0,1]
            "construct":       con,             # [B, 1]  [0,1]
            "quality_score":   quality,         # [B, 1]  lower = better
            "ranking_score":   1.0 - quality,   # [B, 1]  higher = better
        }


# -----------------------------------------------------------------------
# Stage C: Constraint projection
# -----------------------------------------------------------------------

class ConstraintProjector:
    """
    Stage C: Project a generated occupancy tensor to satisfy hard constraints.

    Guarantees:
        1. must_on slots are active
        2. must_off slots are inactive
        3. Active wall count matches target shear ratio (greedy add/remove)
    """

    def __init__(self, wall_unit: float = 2.0, bay_m: float = 6.0):
        self.wall_unit = wall_unit
        self.bay_m     = bay_m

    def project(
        self,
        occ:         torch.Tensor,
        must_on:     torch.Tensor,
        must_off:    torch.Tensor,
        variable:    torch.Tensor,
        lx:          int,
        ly:          int,
        shear_ratio: float,
        logits:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project a single occupancy tensor [2, F, GRID, GRID].

        Args:
            occ:         [2, F, GRID, GRID]  raw binary from diffusion
            must_on/off: [2, GRID, GRID]
            variable:    [2, GRID, GRID]
            lx, ly:      bay counts
            shear_ratio: target ratio
            logits:      [2, F, GRID, GRID]  optional logit scores for
                         smarter add/remove selection

        Returns:
            projected: [2, F, GRID, GRID]  binary, constraints satisfied
        """
        out = occ.clone().float()
        F   = out.size(1)

        # Step 1: enforce must_on / must_off
        for ch in range(2):
            on  = must_on[ch]
            off = must_off[ch]
            out[ch, :, on]  = 1.0
            out[ch, :, off] = 0.0

        # Step 2: shear ratio correction per floor
        max_slots    = (ly + 1) * lx * SLOTS_PER_BAY + (lx + 1) * ly * SLOTS_PER_BAY
        target_walls = max(8, int(shear_ratio * max_slots))

        for f in range(F):
            n_active = int(out[:, f].sum().item())
            diff     = target_walls - n_active

            if diff == 0:
                continue

            var_mask_flat = variable.reshape(-1).to(occ.device)    # [2*GRID*GRID]
            occ_flat      = out[:, f].reshape(-1)   # [2*GRID*GRID]

            if diff > 0:
                # Add walls: pick inactive variable slots with highest logit
                candidates = (
                    (occ_flat < 0.5) & (var_mask_flat > 0.5)
                ).nonzero(as_tuple=False).view(-1)

                if len(candidates) == 0:
                    continue

                if logits is not None:
                    scores = logits[:, f].reshape(-1)[candidates]
                    candidates = candidates[scores.argsort(descending=True)]

                for idx in candidates[:diff]:
                    ch  = int(idx) // (GRID * GRID)
                    rem = int(idx) %  (GRID * GRID)
                    out[ch, f, rem // GRID, rem % GRID] = 1.0

            else:
                # Remove walls: pick active variable slots with lowest logit
                candidates = (
                    (occ_flat > 0.5) & (var_mask_flat > 0.5)
                ).nonzero(as_tuple=False).view(-1)

                if len(candidates) == 0:
                    continue

                if logits is not None:
                    scores = logits[:, f].reshape(-1)[candidates]
                    candidates = candidates[scores.argsort(descending=False)]

                for idx in candidates[:abs(diff)]:
                    ch  = int(idx) // (GRID * GRID)
                    rem = int(idx) %  (GRID * GRID)
                    out[ch, f, rem // GRID, rem % GRID] = 0.0

        return out.long().float()