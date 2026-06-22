"""
disc_diffusion/training/trainer.py
=====================================
Training loop for the discrete diffusion pipeline.

Trains two models jointly (or separately):
    1. WallSlotDenoiser   (Stage B)
    2. StructuralSurrogate (Stage D)

Denoiser loss:
    L_diff   = BCE(pred_logits_var, x0_variable)    [only on variable slots]
    L_ratio  = |predicted_active_ratio - target_ratio|
    L_vert   = floor-to-floor consistency of variable slots

Surrogate loss:
    L_xdir   = HuberLoss(x_dir_pred, x_dir_gt)
    L_ydir   = HuberLoss(y_dir_pred, y_dir_gt)
    L_pseudo = MSE on torsion, symmetry, constructability pseudo-labels
    L_rank   = pairwise ranking consistency

Usage
-----
    trainer = DiffusionTrainer(denoiser, surrogate, diffusion, train_loader, val_loader, cfg)
    trainer.train(num_epochs=100)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from ..model.diffusion import DiscreteDiffusion
from ..data.slot_repr import GRID

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Geometric pseudo-label helpers (fast, no numpy)
# -----------------------------------------------------------------------

def compute_shear_ratio_batch(
    occ: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    """
    Compute actual shear ratio deviation from target for each sample.

    Args:
        occ:  [B, 2, F, GRID, GRID]
        cond: [B, 5]  cond[:,3] = target shear ratio

    Returns:
        [B]  absolute deviation |actual - target| in [0,1]
    """
    from ..data.slot_repr import SLOTS_PER_BAY
    B = occ.size(0)

    active = occ.float().sum(dim=(1, 2, 3, 4))          # [B]
    lx     = (cond[:, 0] * 10.0).clamp(min=1)
    ly     = (cond[:, 1] * 10.0).clamp(min=1)
    nf     = (cond[:, 2] * 20.0).clamp(min=1)
    target = cond[:, 3]

    max_slots = ((ly + 1) * lx * SLOTS_PER_BAY
               + (lx + 1) * ly * SLOTS_PER_BAY) * nf
    actual = active / max_slots.clamp(min=1)

    return (actual - target).abs().clamp(0, 1)


def compute_symmetry_batch(occ: torch.Tensor) -> torch.Tensor:
    """[B] symmetry penalty from occupancy [B, 2, F, GRID, GRID]."""
    B = occ.size(0)
    both = occ[:, :, 0].sum(dim=1)    # [B, GRID, GRID]  combined H+V for floor 0
    half = GRID // 2
    asym_x = (both[:, :half] - both[:, GRID-half:].flip(-2)).abs().mean(dim=(-2, -1))
    asym_y = (both[:, :, :half] - both[:, :, GRID-half:].flip(-1)).abs().mean(dim=(-2, -1))
    return ((asym_x + asym_y) / 2.0).clamp(0, 1)


def compute_construct_batch(
    occ: torch.Tensor,
    must_on: torch.Tensor,
) -> torch.Tensor:
    """
    [B] constructability: penalize if must_on slots are missing.
    """
    B = occ.size(0)
    penalties = []
    for b in range(B):
        on   = must_on[b].float()         # [2, GRID, GRID]
        floor0 = occ[b, :, 0]             # [2, GRID, GRID]
        n_missing = ((on > 0.5) & (floor0 < 0.5)).float().sum()
        n_required = on.sum() + 1e-6
        penalties.append((n_missing / n_required).clamp(0, 1))
    return torch.stack(penalties)


# -----------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------

class DenoiserLoss(nn.Module):
    def __init__(
        self,
        w_diff:  float = 1.0,
        w_ratio: float = 0.5,
        w_vert:  float = 0.3,
        pos_weight: float = 2.0,   # upweight active slots to fix class imbalance
    ):
        super().__init__()
        self.w_diff   = w_diff
        self.w_ratio  = w_ratio
        self.w_vert   = w_vert
        # pos_weight > 1 penalizes false negatives more -> prevents collapse to 0
        self.register_buffer("pw", torch.tensor(pos_weight))

    def forward(
        self,
        logits:       torch.Tensor,    # [B, 2, F, GRID, GRID]
        x0:           torch.Tensor,    # [B, 2, F, GRID, GRID] ground truth
        variable:     torch.Tensor,    # [B, 2, GRID, GRID]
        shear_ratios: torch.Tensor,    # [B]
        lx: List[int],
        ly: List[int],
    ) -> Tuple[torch.Tensor, Dict]:
        B, _, F, G, _ = logits.shape
        from ..data.slot_repr import SLOTS_PER_BAY

        # BCE with separate pos_weight for H and V channels
        # H/V ratio varies by layout shape - MaskPredictor handles per-layout ratio
        # pos_weight here only prevents collapse, not force H/V bias
        var_exp = variable.unsqueeze(2).expand_as(logits).float()

        bce_h = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(2.0, device=logits.device),
            reduction="none",
        )(logits[:, 0:1], x0[:, 0:1].float())

        bce_v = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(3.0, device=logits.device),
            reduction="none",
        )(logits[:, 1:2], x0[:, 1:2].float())

        bce_all = torch.cat([bce_h, bce_v], dim=1)
        l_diff  = (bce_all * var_exp).sum() / (var_exp.sum() + 1e-8)

        # Shear ratio loss
        pred_active = torch.sigmoid(logits).sum(dim=(-4, -3, -2, -1)) / F
        max_slots   = torch.tensor(
            [(ly[i] + 1) * lx[i] * SLOTS_PER_BAY + (lx[i] + 1) * ly[i] * SLOTS_PER_BAY
             for i in range(B)],
            dtype=torch.float32, device=logits.device,
        )
        pred_ratio = pred_active / max_slots.clamp(min=1)
        l_ratio    = F_smooth_l1(pred_ratio, shear_ratios)

        # Vertical consistency loss
        if F > 1:
            pred_soft = torch.sigmoid(logits)
            l_vert    = (pred_soft[:, :, 1:] - pred_soft[:, :, :-1]).abs().mean()
        else:
            l_vert = torch.zeros(1, device=logits.device)[0]

        total = self.w_diff * l_diff + self.w_ratio * l_ratio + self.w_vert * l_vert
        return total, {
            "l_diff":  float(l_diff),
            "l_ratio": float(l_ratio),
            "l_vert":  float(l_vert),
            "total":   float(total),
        }


class SurrogateLoss(nn.Module):
    def __init__(self, w_drift=1.0, w_pseudo=0.5, w_rank=0.5):
        super().__init__()
        self.w_drift  = w_drift
        self.w_pseudo = w_pseudo
        self.w_rank   = w_rank
        self.huber    = nn.HuberLoss(delta=10.0)

    def forward(
        self,
        preds:     Dict[str, torch.Tensor],
        x_dir:     torch.Tensor,
        y_dir:     torch.Tensor,
        symmetry:  torch.Tensor,
        construct: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        # Drift regression (supervised on dataset labels)
        l_x   = self.huber(preds["x_dir_pred"].view(-1), x_dir)
        l_y   = self.huber(preds["y_dir_pred"].view(-1), y_dir)

        # Geometric pseudo-labels
        l_sym = F.mse_loss(preds["symmetry"].view(-1),  symmetry)
        l_con = F.mse_loss(preds["construct"].view(-1), construct)

        # Pairwise ranking: samples with lower x_dir should rank higher
        B = x_dir.size(0)
        if B > 1:
            qs     = preds["quality_score"].view(-1)
            si     = qs.unsqueeze(1).expand(B, B)
            sj     = qs.unsqueeze(0).expand(B, B)
            ti     = x_dir.unsqueeze(1).expand(B, B)
            tj     = x_dir.unsqueeze(0).expand(B, B)
            better = (ti < tj - 0.5).float()
            l_rank = (F.relu(si - sj) * better).sum() / (better.sum() + 1e-8)
        else:
            l_rank = torch.zeros(1, device=x_dir.device)[0]

        total = (self.w_drift  * (l_x + l_y)
               + self.w_pseudo * (l_sym + l_con)
               + self.w_rank   * l_rank)

        return total, {
            "l_x":    float(l_x),
            "l_y":    float(l_y),
            "l_sym":  float(l_sym),
            "l_con":  float(l_con),
            "l_rank": float(l_rank),
            "total":  float(total),
        }


def F_smooth_l1(pred, target, beta=0.1):
    return F.smooth_l1_loss(pred, target.to(pred.device), beta=beta)


# -----------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------

class DiffusionTrainer:
    """
    Trains denoiser (+ optionally surrogate) on the SlotOccupancyDataset.
    """

    def __init__(
        self,
        denoiser,
        diffusion: DiscreteDiffusion,
        train_loader,
        val_loader,
        config: Optional[Dict] = None,
        surrogate=None,
    ):
        self.denoiser      = denoiser
        self.diffusion     = diffusion
        self.surrogate     = surrogate
        self.train_loader  = train_loader
        self.val_loader    = val_loader

        self.cfg = {
            "lr":              1e-4,
            "weight_decay":    1e-4,
            "epochs":          100,
            "grad_clip":       1.0,
            "save_freq":       5,
            "patience":        15,
            "ckpt_dir":        "outputs/diffusion_checkpoints",
            "w_diff":          1.0,
            "w_ratio":         0.5,
            "w_vert":          0.3,
            "w_drift":         1.0,
            "w_pseudo":        0.5,
            "w_rank":          0.5,
            "train_surrogate": surrogate is not None,
        }
        if config:
            self.cfg.update(config)

        Path(self.cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.denoiser.to(self.device)
        self.diffusion.device = self.device

        params = list(self.denoiser.parameters())
        if self.surrogate and self.cfg["train_surrogate"]:
            self.surrogate.to(self.device)
            params += list(self.surrogate.parameters())

        self.opt = AdamW(params, lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        self.sch = CosineAnnealingWarmRestarts(self.opt, T_0=10, T_mult=2)

        self.den_loss = DenoiserLoss(self.cfg["w_diff"], self.cfg["w_ratio"], self.cfg["w_vert"])
        self.sur_loss = SurrogateLoss(self.cfg["w_drift"], self.cfg["w_pseudo"], self.cfg["w_rank"])

        self.best_val = float("inf")
        self.patience = 0

        logger.info(
            "DiffusionTrainer ready. Device=%s  Denoiser params=%s",
            self.device,
            f"{sum(p.numel() for p in self.denoiser.parameters()):,}",
        )

    def train(self, num_epochs: Optional[int] = None) -> None:
        epochs = num_epochs or self.cfg["epochs"]
        logger.info("Starting diffusion training for %d epochs", epochs)

        for epoch in range(1, epochs + 1):
            tr = self._train_epoch(epoch)
            vl = self._val_epoch(epoch)
            self.sch.step()

            logger.info(
                "Epoch %3d | train=%.4f  val=%.4f  "
                "den_diff=%.4f  den_ratio=%.4f  l_freq=%.4f  sur_x=%.4f  sur_y=%.4f",
                epoch, tr.get("total", 0), vl.get("total", 0),
                vl.get("den_l_diff",  0),
                vl.get("den_l_ratio", 0),
                vl.get("l_freq",      0),
                vl.get("sur_l_x",     0),
                vl.get("sur_l_y",     0),
            )

            if vl["total"] < self.best_val:
                self.best_val = vl["total"]
                self.patience = 0
                self._save("best_diffusion.pt")
                logger.info("  -> New best (val=%.4f)", vl["total"])
            else:
                self.patience += 1

            if self.patience >= self.cfg["patience"]:
                logger.info("Early stopping at epoch %d", epoch)
                break

            if epoch % self.cfg["save_freq"] == 0:
                self._save(f"diffusion_epoch_{epoch}.pt")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.denoiser.train()
        if self.surrogate and self.cfg["train_surrogate"]:
            self.surrogate.train()

        agg = defaultdict(float)
        n   = 0

        for batch in tqdm(self.train_loader, desc=f"Train {epoch}", leave=False):
            try:
                loss, log = self._step(batch)
            except Exception as e:
                logger.warning("Step error: %s", e)
                continue

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.denoiser.parameters(), self.cfg["grad_clip"])
            if self.surrogate and self.cfg["train_surrogate"]:
                nn.utils.clip_grad_norm_(self.surrogate.parameters(), self.cfg["grad_clip"])
            self.opt.step()

            for k, v in log.items():
                agg[k] += v
            n += 1

        return {k: v / max(n, 1) for k, v in agg.items()}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        self.denoiser.eval()
        if self.surrogate:
            self.surrogate.eval()

        agg = defaultdict(float)
        n   = 0

        for batch in self.val_loader:
            try:
                _, log = self._step(batch, grad=False)
            except Exception as e:
                continue
            for k, v in log.items():
                agg[k] += v
            n += 1

        return {k: v / max(n, 1) for k, v in agg.items()}

    def _step(self, batch: Dict, grad: bool = True) -> Tuple[torch.Tensor, Dict]:
        occ_full  = batch["occ_full"].to(self.device)      # [B, 2, F, GRID, GRID]
        must_on   = batch["must_on"].to(self.device)       # [B, 2, GRID, GRID]
        cond      = batch["condition"].to(self.device)     # [B, 5]
        x_dir     = batch["x_dir"].to(self.device)
        y_dir     = batch["y_dir"].to(self.device)
        shear_r   = torch.tensor(batch["shear_ratio"], dtype=torch.float32, device=self.device)
        lx        = batch["lx"]
        ly        = batch["ly"]

        B = occ_full.size(0)

        # Compute analytical masks from geometry (no lookup needed)
        from ..model.denoiser import MaskPredictor
        masks    = MaskPredictor.batch_analytical_masks(lx, ly, self.device)
        variable = masks["variable"]   # [B, 2, GRID, GRID]
        must_on  = masks["must_on"]    # override batch must_on with analytical

        # Sample timesteps and noise
        t       = self.diffusion.sample_timesteps(B).to(self.device)
        noise_d = self.diffusion.training_losses(occ_full, t, variable)
        x_t     = noise_d["x_t"]

        # Predict frequency map from condition (integrated MaskPredictor)
        freq_map = self.denoiser.predict_freq_map(cond)    # [B, 2, GRID, GRID]

        # Build 6-channel input and run denoiser
        inp    = self.denoiser.build_input(x_t, must_on, variable, freq_map)
        out    = self.denoiser(inp, t, cond)
        logits = out["logits"]         # [B, 2, F, GRID, GRID]

        # Denoiser loss (BCE on variable region)
        den_total, den_log = self.den_loss(logits, occ_full, variable, shear_r, lx, ly)

        # Freq map loss: MSE against actual floor-0 occupancy (ground truth frequency)
        # Use floor 0 as representative (walls repeat across floors)
        freq_target = occ_full[:, :, 0, :, :].float()     # [B, 2, GRID, GRID]
        l_freq = F.mse_loss(freq_map, freq_target)

        total = den_total + 0.3 * l_freq
        log   = {f"den_{k}": v for k, v in den_log.items()}
        log["l_freq"] = float(l_freq)

        # Surrogate loss
        if self.surrogate and self.cfg["train_surrogate"]:
            sur_out   = self.surrogate(occ_full, cond)
            symmetry  = compute_symmetry_batch(occ_full).detach()
            construct = compute_construct_batch(occ_full, must_on).detach()
            sur_total, sur_log = self.sur_loss(
                sur_out, x_dir, y_dir, symmetry, construct
            )
            total = total + sur_total
            log.update({f"sur_{k}": v for k, v in sur_log.items()})

        log["total"] = float(total)
        return total, log

    def _save(self, filename: str) -> None:
        path = Path(self.cfg["ckpt_dir"]) / filename
        ckpt = {
            "denoiser":  self.denoiser.state_dict(),
            "optimizer": self.opt.state_dict(),
            "best_val":  self.best_val,
            "config":    self.cfg,
        }
        if self.surrogate:
            ckpt["surrogate"] = self.surrogate.state_dict()
        torch.save(ckpt, path)
        logger.info("Checkpoint saved: %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.denoiser.load_state_dict(ckpt["denoiser"])
        if self.surrogate and "surrogate" in ckpt:
            self.surrogate.load_state_dict(ckpt["surrogate"])
        logger.info("Loaded checkpoint: %s", path)