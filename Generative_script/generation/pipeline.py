"""
disc_diffusion/generation/pipeline.py
========================================
Full inference pipeline: conditions -> best wall layout.

Stages:
    A. Analytical mask from geometry (no lookup needed)
    B. Run conditional discrete diffusion sampling (N candidates)
    C. Project each candidate (ratio correction + hard masks)
    D. Score with surrogate, return best + all candidates
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import torch

from ..data.mask_builder import MaskBuilder
from ..data.slot_repr import tensor_to_walls, GRID, MAX_FLOORS, SLOTS_PER_BAY
from ..model.diffusion import DiscreteDiffusion
from ..model.surrogate import ConstraintProjector
from ..model.denoiser import MaskPredictor

logger = logging.getLogger(__name__)


class GenerationPipeline:
    def __init__(
        self,
        denoiser,
        surrogate,
        diffusion: DiscreteDiffusion,
        mask_builder: MaskBuilder,
        device: str = "cuda",
    ):
        self.denoiser    = denoiser
        self.surrogate   = surrogate
        self.diffusion   = diffusion
        self.masks       = mask_builder
        self.projector   = ConstraintProjector()
        self.device      = torch.device(device if torch.cuda.is_available() else "cpu")

        self.denoiser.to(self.device).eval()
        self.surrogate.to(self.device).eval()
        self.diffusion.device = self.device

    @torch.no_grad()
    def generate(
        self,
        lx:             int,
        ly:             int,
        num_floors:     int   = 10,
        shear_ratio:    Optional[float] = None,
        num_candidates: int   = 16,
        diffusion_steps: Optional[int] = None,
    ) -> Dict:
        t0 = time.time()

        # Stage A: analytical masks (works for ANY layout)
        masks    = MaskPredictor.analytical_masks(lx, ly, self.device)
        must_on  = masks["must_on"].unsqueeze(0).to(self.device)
        must_off = masks["must_off"].unsqueeze(0).to(self.device)
        variable = masks["variable"].unsqueeze(0).to(self.device)

        # Estimate shear ratio if not provided
        fmask = self.masks.get_mask(lx, ly)
        if shear_ratio is None:
            shear_ratio = self._estimate_shear_ratio(fmask)

        # Condition vector [1, 5]
        cond = torch.tensor([[
            lx / 10.0,
            ly / 10.0,
            num_floors / 20.0,
            float(shear_ratio),
            0.0,
        ]], dtype=torch.float32, device=self.device)

        # Stage B: generate N candidates via diffusion
        candidates = []
        for i in range(num_candidates):
            occ = self.diffusion.sample(
                model=self.denoiser,
                cond=cond,
                must_on=must_on,
                must_off=must_off,
                variable=variable,
                num_floors=num_floors,
                T_start=diffusion_steps,
            )  # [1, 2, F, GRID, GRID]

            # Stage C: project to satisfy ratio
            projected = self.projector.project(
                occ=occ[0],
                must_on=masks["must_on"],
                must_off=masks["must_off"],
                variable=masks["variable"],
                lx=lx, ly=ly,
                shear_ratio=shear_ratio,
                logits=None,
            )  # [2, F, GRID, GRID]

            # Copy floor 0 layout to all floors (walls same across floors)
            floor0    = projected[:, 0:1, :, :]
            projected = floor0.expand_as(projected).clone()

            candidates.append(projected)

        # Stage D: surrogate scoring
        occ_batch  = torch.stack(candidates, dim=0).to(self.device)
        cond_batch = cond.expand(num_candidates, -1)

        scores_out = self.surrogate(occ_batch, cond_batch)
        ranking    = scores_out["ranking_score"].view(-1).cpu().tolist()

        order    = sorted(range(num_candidates), key=lambda i: ranking[i], reverse=True)
        best_idx = order[0]
        best_occ = candidates[best_idx].cpu()

        # Convert best to wall lists (all floors)
        F_dim      = min(num_floors, MAX_FLOORS)
        best_walls = []
        for f in range(F_dim):
            floor_occ = best_occ[:, f, :, :]
            best_walls.append(tensor_to_walls(floor_occ, lx, ly, floor_idx=f))

        elapsed = time.time() - t0

        # Build all candidates list (sorted best first)
        all_candidates = []
        for rank_pos, orig_idx in enumerate(order):
            occ_i   = candidates[orig_idx].cpu()
            floor0_walls = tensor_to_walls(occ_i[:, 0, :, :], lx, ly, floor_idx=0)
            all_candidates.append({
                "rank":             rank_pos + 1,
                "sample_id":        orig_idx,
                "walls":            floor0_walls,
                "ranking_score":    float(ranking[orig_idx]),
                "x_dir_pred":       float(scores_out["x_dir_pred"][orig_idx]),
                "y_dir_pred":       float(scores_out["y_dir_pred"][orig_idx]),
                "symmetry_pred":    float(scores_out["symmetry"][orig_idx]),
                "constructability": float(scores_out["construct"][orig_idx]),
            })

        best_cand = all_candidates[0]
        report    = self._build_report(
            all_candidates, lx, ly, num_floors, shear_ratio, num_candidates, elapsed
        )

        logger.info(
            "Generated %d candidates for %dx%d in %.1fs. "
            "Best rank_score=%.4f  x_drift=%.1f  y_drift=%.1f",
            num_candidates, lx, ly, elapsed,
            best_cand["ranking_score"],
            best_cand["x_dir_pred"],
            best_cand["y_dir_pred"],
        )

        return {
            "best":          best_occ,
            "walls":         best_walls,
            "ranking_score": best_cand["ranking_score"],
            "all":           all_candidates,
            "report":        report,
            "lx": lx, "ly": ly,
            "num_floors":    num_floors,
            "shear_ratio":   shear_ratio,
            "x_dir_pred":    best_cand["x_dir_pred"],
            "y_dir_pred":    best_cand["y_dir_pred"],
        }

    def _build_report(
        self,
        all_candidates: List[Dict],
        lx: int, ly: int,
        num_floors: int,
        shear_ratio: float,
        total_n: int,
        elapsed: float,
    ) -> str:
        lines = [
            "=" * 60,
            f"Discrete Diffusion Generation Report",
            f"  Layout     : {lx}x{ly} bays  ({lx*6}m x {ly*6}m)",
            f"  Floors     : {num_floors}",
            f"  Shear ratio: {shear_ratio:.3f}",
            f"  Candidates : {total_n}",
            f"  Time       : {elapsed:.1f}s",
            "=" * 60,
        ]

        for cand in all_candidates[:3]:
            rank = cand["rank"]
            xd   = cand["x_dir_pred"]
            yd   = cand["y_dir_pred"]
            sym  = cand["symmetry_pred"]
            con  = cand["constructability"]
            rs   = cand["ranking_score"]

            lines.append(f"\nRank #{rank}  (sample #{cand['sample_id']})")
            lines.append(f"  Ranking score      : {rs:.4f}")
            lines.append(f"  X-Dir drift (pred) : {xd:.2f} mm")
            lines.append(f"  Y-Dir drift (pred) : {yd:.2f} mm")
            lines.append(f"  Symmetry penalty   : {sym:.3f}")
            lines.append(f"  Constructability   : {con:.3f}")

            flags = []
            if sym  > 0.5:  flags.append("ASYMMETRIC")
            if con  > 0.4:  flags.append("CONSTRUCTABILITY ISSUES")
            if xd   > 100:  flags.append("X-DRIFT EXCESSIVE")
            if yd   > 60:   flags.append("Y-DRIFT EXCESSIVE")
            lines.append(f"  Status             : {'ALL OK' if not flags else ', '.join(flags)}")

        lines += ["", "=" * 60,
                  f"Selected: Rank #1 candidate (sample #{all_candidates[0]['sample_id']}).",
                  "=" * 60]
        return "\n".join(lines)

    def _estimate_shear_ratio(self, fmask) -> float:
        max_slots = (fmask.ly + 1) * fmask.lx * SLOTS_PER_BAY \
                  + (fmask.lx + 1) * fmask.ly * SLOTS_PER_BAY
        baseline = float(fmask.must_on.sum()) / max(max_slots, 1)
        return max(0.2, min(0.7, baseline + 0.15))