"""
disc_diffusion/model/diffusion.py
===================================
Discrete diffusion process for binary wall-slot occupancy maps.

Forward process: Bernoulli bit-flip corruption.
    At timestep t, each variable slot is independently flipped with
    probability beta_t (cumulative: alpha_bar_t).

    q(x_t | x_0) = Bernoulli(x_0 * (1 - alpha_bar_t) + (1 - x_0) * alpha_bar_t)

Reverse process: predict x_0 logits from x_t, then resample x_{t-1}.

    p(x_{t-1} | x_t, x_0_pred) = Bernoulli posterior

This formulation is based on:
    "Structured Denoising Diffusion Models in Discrete State-Spaces"
    Austin et al. (2021)  -- absorbing state / uniform transition variant.

Only the variable region is noised; must_on / must_off slots stay fixed.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Noise schedule
# -----------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule from Nichol & Dhariwal (2021).
    Returns beta_t for t = 1..T  shape [T].
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas     = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0.0001, 0.999).float()


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


# -----------------------------------------------------------------------
# Diffusion engine
# -----------------------------------------------------------------------

class DiscreteDiffusion:
    """
    Handles forward noising and reverse sampling for binary occupancy maps.

    All operations assume:
        x in {0, 1}  (binary occupancy)
        masks computed by MaskBuilder

    Args:
        T:        total diffusion steps (default 200)
        schedule: 'cosine' or 'linear'
        device:   torch device
    """

    def __init__(
        self,
        T:        int    = 200,
        schedule: str    = "cosine",
        device:   str    = "cpu",
    ):
        self.T      = T
        self.device = torch.device(device)

        # Build schedules
        if schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            betas = linear_beta_schedule(T)

        # alpha_bar_t = product_{s=1}^{t} (1 - beta_s)
        # Interpretation: probability a slot keeps its original value at time t
        alphas      = 1.0 - betas
        alpha_bar   = torch.cumprod(alphas, dim=0)           # [T]
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])  # alpha_bar_{t-1}

        # Pre-compute to device
        self.betas          = betas.to(self.device)
        self.alpha_bar      = alpha_bar.to(self.device)
        self.alpha_bar_prev = alpha_bar_prev.to(self.device)

    # ------------------------------------------------------------------
    # Forward process  q(x_t | x_0)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0:       torch.Tensor,
        t:        torch.Tensor,
        variable: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to x0 at timesteps t.

        Args:
            x0:       [B, 2, F, GRID, GRID]  clean binary occupancy
            t:        [B]                     integer timestep indices (1-indexed)
            variable: [B, 2, GRID, GRID]      variable region mask

        Returns:
            x_t:  [B, 2, F, GRID, GRID]  noisy occupancy
        """
        B, C, F, G, _ = x0.shape

        # alpha_bar for each sample  [B, 1, 1, 1, 1]
        ab = self.alpha_bar[t - 1].view(B, 1, 1, 1, 1)

        # Flip probability per slot:
        #   If x0 = 1: prob(x_t = 1) = ab   (stays 1 with prob ab)
        #   If x0 = 0: prob(x_t = 1) = 1-ab (flipped to 1 with prob 1-ab)
        flip_prob = x0.float() * ab + (1.0 - x0.float()) * (1.0 - ab)
        # = prob(x_t = 1 | x_0)

        noise = torch.bernoulli(flip_prob)   # [B, 2, F, GRID, GRID]

        # Only noise the variable slots; keep fixed slots intact
        var_exp = variable.unsqueeze(2).expand_as(x0).float()  # [B, 2, F, GRID, GRID]
        x_t = x0.float() * (1.0 - var_exp) + noise * var_exp

        return x_t

    # ------------------------------------------------------------------
    # Posterior  q(x_{t-1} | x_t, x_0)
    # ------------------------------------------------------------------

    def q_posterior_mean(
        self,
        x0_pred:  torch.Tensor,
        x_t:      torch.Tensor,
        t:        torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute probability of x_{t-1} = 1 given x_t and predicted x0.

        For binary discrete diffusion with bit-flip:
            p(x_{t-1} = 1 | x_t, x_0) proportional to
                sum_{x0} p(x_t | x_{t-1}) * p(x_{t-1} | x_0) * p_theta(x_0 | x_t)

        Simplified closed form for bit-flip (see Austin et al. Eq. 6):
            theta = [alpha_bar_{t-1} * x_0 + (1-alpha_bar_{t-1}) * (1-x_0)]
                  * [beta_t * (1-x_t) + alpha_t * x_t]
        (unnormalized, then normalize over {0,1})

        Returns:
            prob_1: [B, 2, F, GRID, GRID]  probability that x_{t-1} = 1
        """
        B = x0_pred.size(0)

        ab_t   = self.alpha_bar[t - 1].view(B, 1, 1, 1, 1)
        ab_tm1 = self.alpha_bar_prev[t - 1].view(B, 1, 1, 1, 1)
        beta_t = self.betas[t - 1].view(B, 1, 1, 1, 1)
        alpha_t = 1.0 - beta_t

        x0 = x0_pred.float().clamp(0.0, 1.0)  # soft prediction in [0,1]
        xt = x_t.float()

        # Numerator for x_{t-1}=1:
        # p(x_{t-1}=1|x0) * p(x_t|x_{t-1}=1)
        p_tm1_given_x0_1 = ab_tm1 * x0 + (1 - ab_tm1) * (1 - x0)
        p_xt_given_tm1_1 = alpha_t * xt + beta_t * (1 - xt)
        num1 = p_tm1_given_x0_1 * p_xt_given_tm1_1

        # Numerator for x_{t-1}=0:
        p_tm1_given_x0_0 = ab_tm1 * (1 - x0) + (1 - ab_tm1) * x0
        p_xt_given_tm1_0 = alpha_t * (1 - xt) + beta_t * xt
        num0 = p_tm1_given_x0_0 * p_xt_given_tm1_0

        prob_1 = num1 / (num1 + num0 + 1e-8)
        return prob_1

    # ------------------------------------------------------------------
    # Reverse sampling step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        model,
        x_t:      torch.Tensor,
        t:        torch.Tensor,
        cond:     torch.Tensor,
        must_on:  torch.Tensor,
        must_off: torch.Tensor,
        variable: torch.Tensor,
    ) -> torch.Tensor:
        """Single reverse step: x_t -> x_{t-1}."""
        # Predict freq_map from condition (integrated MaskPredictor)
        freq_map = model.predict_freq_map(cond)             # [B, 2, GRID, GRID]

        inp    = model.build_input(x_t, must_on, variable, freq_map)
        out    = model(inp, t, cond)
        logits = out["logits"]
        x0_pred = torch.sigmoid(logits)

        prob_1 = self.q_posterior_mean(x0_pred, x_t, t)
        x_tm1  = torch.bernoulli(prob_1)
        x_tm1  = self._apply_masks(x_tm1, must_on, must_off)
        return x_tm1

    @torch.no_grad()
    def sample(
        self,
        model,
        cond:     torch.Tensor,
        must_on:  torch.Tensor,
        must_off: torch.Tensor,
        variable: torch.Tensor,
        num_floors: int = 10,
        T_start:  Optional[int] = None,
    ) -> torch.Tensor:
        """
        Full reverse sampling from x_T to x_0.

        Args:
            model:      WallSlotDenoiser
            cond:       [B, cond_dim]
            must_on:    [B, 2, GRID, GRID]
            must_off:   [B, 2, GRID, GRID]
            variable:   [B, 2, GRID, GRID]
            num_floors: number of floors to generate
            T_start:    start from this timestep (default: self.T)

        Returns:
            x0: [B, 2, F, GRID, GRID]  binary occupancy
        """
        from ..data.slot_repr import GRID as GRID_SIZE, MAX_FLOORS

        B      = cond.size(0)
        device = cond.device
        F      = min(num_floors, MAX_FLOORS)
        T_run  = T_start or self.T

        # Start from pure noise in the variable region
        x_t = self._initialize_noise(B, F, must_on, must_off, variable, device)

        for step in range(T_run, 0, -1):
            t = torch.full((B,), step, dtype=torch.long, device=device)
            x_t = self.p_sample(model, x_t, t, cond, must_on, must_off, variable)

        # Final hard mask enforcement
        x_t = self._apply_masks(x_t, must_on, must_off)
        return x_t

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _initialize_noise(
        self,
        B: int, F: int,
        must_on:  torch.Tensor,
        must_off: torch.Tensor,
        variable: torch.Tensor,
        device:   torch.device,
    ) -> torch.Tensor:
        """Initialize x_T: random in variable region, fixed in mask regions."""
        from ..data.slot_repr import GRID
        x = torch.bernoulli(torch.full((B, 2, F, GRID, GRID), 0.5, device=device))
        return self._apply_masks(x, must_on, must_off)

    @staticmethod
    def _apply_masks(
        x:        torch.Tensor,
        must_on:  torch.Tensor,
        must_off: torch.Tensor,
    ) -> torch.Tensor:
        """Force must_on=1, must_off=0 across all floors."""
        # must_on/must_off: [B, 2, GRID, GRID]
        F = x.size(2)
        on_exp  = must_on.unsqueeze(2).expand_as(x).float()
        off_exp = must_off.unsqueeze(2).expand_as(x).float()
        x = x.float()
        x = torch.where(on_exp  > 0.5, torch.ones_like(x),  x)
        x = torch.where(off_exp > 0.5, torch.zeros_like(x), x)
        return x

    # ------------------------------------------------------------------
    # Training: compute loss-ready quantities
    # ------------------------------------------------------------------

    def training_losses(
        self,
        x0:       torch.Tensor,
        t:        torch.Tensor,
        variable: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample x_t from x_0 and return noisy input for denoiser training.

        Args:
            x0:       [B, 2, F, GRID, GRID]  clean binary
            t:        [B]  integer timesteps
            variable: [B, 2, GRID, GRID]     variable mask

        Returns:
            dict with x_t (noisy) and x0 (target)
        """
        x_t = self.q_sample(x0, t, variable)
        return {"x_t": x_t, "x0_target": x0, "t": t}

    def sample_timesteps(self, B: int) -> torch.Tensor:
        """Uniformly sample B timesteps in [1, T]."""
        return torch.randint(1, self.T + 1, (B,))