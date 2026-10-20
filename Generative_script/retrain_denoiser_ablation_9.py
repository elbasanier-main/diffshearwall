#!/usr/bin/env python3
"""
Retrain all 9 denoiser ablation configurations for generation-based selection.

Key differences from the original ablation:
1) Trains the DENOISER ONLY. No auxiliary StructuralSurrogate is instantiated.
2) Runs a fixed training budget (default 100 epochs); no early stopping is used
   for the final denoiser-selection protocol.
3) Saves periodic denoiser checkpoints every N epochs (default 5).
4) The periodic checkpoints are intended to be selected LATER using genuine
   generation metrics, not validation loss.
5) A denoiser-validation-loss "audit best" checkpoint is also saved, but it is
   explicitly NOT the final selection criterion.

This script does not overwrite outputs/ablation_full. It writes to a new root.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

CONFIGS = [
    (0.3, 0.1),
    (0.3, 0.3),
    (0.3, 0.5),
    (0.5, 0.1),
    (0.5, 0.3),
    (0.5, 0.5),
    (1.0, 0.1),
    (1.0, 0.3),
    (1.0, 0.5),
]


def tag_float(x: float) -> str:
    return str(x).replace(".", "p")


def config_name(w_ratio: float, w_vert: float) -> str:
    return f"denoiser__w_ratio_{tag_float(w_ratio)}_w_vert_{tag_float(w_vert)}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility-oriented CuDNN settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_metrics(prefix: str, d: dict) -> dict:
    return {f"{prefix}_{k}": float(v) for k, v in d.items()}


# ---------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------

def train_one_config(
    repo: Path,
    data_root: Path,
    masks_path: Path,
    out_dir: Path,
    w_ratio: float,
    w_vert: float,
    epochs: int,
    save_freq: int,
    batch_size: int,
    workers: int,
    lr: float,
    T: int,
    schedule: str,
    base_ch: int,
    seed: int,
) -> None:
    from disc_diffusion.data.slot_dataset import create_dataloaders
    from disc_diffusion.model.denoiser import WallSlotDenoiser
    from disc_diffusion.model.diffusion import DiscreteDiffusion
    from disc_diffusion.training.trainer import DiffusionTrainer

    log = logging.getLogger("retrain9")

    # Same initialization seed for every hyperparameter configuration.
    set_seed(seed)

    # Rebuild loaders per configuration so every run starts from the same split.
    train_loader, val_loader, _ = create_dataloaders(
        data_root=str(data_root),
        masks_path=str(masks_path),
        batch_size=batch_size,
        num_workers=workers,
        max_floors=20,
    )

    denoiser = WallSlotDenoiser(
        in_channels=6,
        base_ch=base_ch,
        cond_in_dim=5,
        cond_embed_dim=128,
        t_embed_dim=128,
    )
    diffusion = DiscreteDiffusion(
        T=T,
        schedule=schedule,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # IMPORTANT:
    # surrogate=None and train_surrogate=False => auxiliary surrogate is absent.
    cfg = {
        "lr": lr,
        "epochs": epochs,
        "save_freq": save_freq,
        # We do NOT use trainer.train(), so patience is irrelevant.
        "patience": epochs + 999,
        "ckpt_dir": str(out_dir),
        "train_surrogate": False,
        "w_diff": 1.0,
        "w_ratio": w_ratio,
        "w_vert": w_vert,
    }

    trainer = DiffusionTrainer(
        denoiser=denoiser,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        surrogate=None,
    )

    history_path = out_dir / "training_log.csv"
    fieldnames = None
    audit_best = math.inf
    audit_best_epoch = None

    log.info("=" * 84)
    log.info("CONFIG %s", config_name(w_ratio, w_vert))
    log.info(
        "w_ratio=%.3f w_vert=%.3f | epochs=%d save_freq=%d | surrogate=OFF",
        w_ratio, w_vert, epochs, save_freq,
    )
    log.info("output=%s", out_dir)
    log.info("=" * 84)

    for epoch in range(1, epochs + 1):
        tr = trainer._train_epoch(epoch)
        vl = trainer._val_epoch(epoch)
        trainer.sch.step()

        row = {
            "epoch": epoch,
            "lr": float(trainer.opt.param_groups[0]["lr"]),
            **flatten_metrics("train", tr),
            **flatten_metrics("val", vl),
        }

        if fieldnames is None:
            fieldnames = list(row.keys())
            with history_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()

        with history_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(row)

        val_total = float(vl.get("total", math.inf))

        log.info(
            "Epoch %3d/%3d | train_total=%.6f | val_total=%.6f | "
            "val_diff=%.6f | val_ratio=%.6f | val_vert=%.6f | val_freq=%.6f",
            epoch,
            epochs,
            float(tr.get("total", float("nan"))),
            val_total,
            float(vl.get("den_l_diff", float("nan"))),
            float(vl.get("den_l_ratio", float("nan"))),
            float(vl.get("den_l_vert", float("nan"))),
            float(vl.get("l_freq", float("nan"))),
        )

        # Audit-only validation-loss best. Do NOT use this as final denoiser selection.
        if val_total < audit_best:
            audit_best = val_total
            audit_best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "denoiser": trainer.denoiser.state_dict(),
                    "config": cfg,
                    "train_metrics": tr,
                    "val_metrics": vl,
                    "selection_role": "AUDIT_ONLY_denoiser_validation_loss",
                },
                out_dir / "audit_best_by_denoiser_val_loss.pt",
            )

        # Periodic checkpoints for genuine generation-based selection.
        if epoch % save_freq == 0 or epoch == epochs:
            ckpt_path = out_dir / f"diffusion_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "denoiser": trainer.denoiser.state_dict(),
                    "config": cfg,
                    "train_metrics": tr,
                    "val_metrics": vl,
                    "selection_role": "CANDIDATE_FOR_GENERATION_METRIC_SELECTION",
                },
                ckpt_path,
            )
            log.info("Saved periodic checkpoint: %s", ckpt_path.name)

    marker = out_dir / "TRAINING_COMPLETE.txt"
    marker.write_text(
        "\n".join(
            [
                f"config={config_name(w_ratio, w_vert)}",
                f"w_ratio={w_ratio}",
                f"w_vert={w_vert}",
                f"epochs={epochs}",
                f"save_freq={save_freq}",
                f"seed={seed}",
                "auxiliary_surrogate=NOT_USED",
                "early_stopping=NOT_USED",
                "final_checkpoint_selection=PENDING_GENERATION_METRICS",
                f"audit_best_val_epoch={audit_best_epoch}",
                f"audit_best_val_total={audit_best}",
            ]
        )
        + "\n"
    )

    log.info(
        "DONE %s | audit best val epoch=%s, val_total=%.6f | "
        "FINAL selection still pending generation metrics",
        config_name(w_ratio, w_vert),
        audit_best_epoch,
        audit_best,
    )

    # Explicit cleanup before the next large model.
    del trainer, denoiser, diffusion, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Retrain all 9 denoiser ablation configs for generation-based checkpoint selection."
    )
    ap.add_argument("--repo", default="/home/elbasanier/CGVAE_RL_Tran")
    ap.add_argument("--data-root", required=True, help="Original JSON training-data root used by diffusion.")
    ap.add_argument("--masks", required=True)
    ap.add_argument(
        "--out-root",
        default="/home/elbasanier/CGVAE_RL_Tran/disc_diffusion/outputs/ablation_retrain_genmetric",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--save-freq", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="If a config folder exists without TRAINING_COMPLETE.txt, delete it and retrain that config from scratch.",
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    data_root = Path(args.data_root).resolve()
    masks = Path(args.masks).resolve()
    out_root = Path(args.out_root).resolve()

    if not repo.exists():
        raise SystemExit(f"Repo not found: {repo}")
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")
    if not masks.exists():
        raise SystemExit(f"Masks file not found: {masks}")

    # Make imports work when launched from anywhere.
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    out_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    log = logging.getLogger("retrain9")

    manifest = out_root / "run_manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "Corrected denoiser ablation retraining",
                f"repo={repo}",
                f"data_root={data_root}",
                f"masks={masks}",
                f"device={'cuda' if torch.cuda.is_available() else 'cpu'}",
                f"epochs={args.epochs}",
                f"save_freq={args.save_freq}",
                f"batch_size={args.batch_size}",
                f"workers={args.workers}",
                f"lr={args.lr}",
                f"T={args.T}",
                f"schedule={args.schedule}",
                f"base_ch={args.base_ch}",
                f"seed={args.seed}",
                "auxiliary_surrogate=NOT_USED",
                "early_stopping=NOT_USED",
                "checkpoint_selection=PENDING_GENERATION_METRICS",
                "configs=" + ",".join(config_name(a, b) for a, b in CONFIGS),
            ]
        )
        + "\n"
    )

    print("\nCorrected denoiser ablation")
    print("===========================")
    print(f"Repo        : {repo}")
    print(f"Data root   : {data_root}")
    print(f"Masks       : {masks}")
    print(f"Output root : {out_root}")
    print(f"Device      : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Configs     : {len(CONFIGS)}")
    print(f"Epochs      : {args.epochs}")
    print(f"Save every  : {args.save_freq} epochs")
    print("Aux surrogate: NOT USED")
    print("Early stopping: NOT USED")
    print()

    for i, (wr, wv) in enumerate(CONFIGS, start=1):
        name = config_name(wr, wv)
        out_dir = out_root / name
        complete = out_dir / "TRAINING_COMPLETE.txt"

        if complete.exists():
            log.info("[%d/9] SKIP completed: %s", i, name)
            continue

        if out_dir.exists() and any(out_dir.iterdir()):
            if args.restart_incomplete:
                log.warning("[%d/9] Removing incomplete folder: %s", i, out_dir)
                shutil.rmtree(out_dir)
            else:
                raise SystemExit(
                    f"Incomplete/non-empty folder exists: {out_dir}\n"
                    "Re-run with --restart-incomplete to delete and retrain that config from scratch."
                )

        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("[%d/9] START %s", i, name)
        train_one_config(
            repo=repo,
            data_root=data_root,
            masks_path=masks,
            out_dir=out_dir,
            w_ratio=wr,
            w_vert=wv,
            epochs=args.epochs,
            save_freq=args.save_freq,
            batch_size=args.batch_size,
            workers=args.workers,
            lr=args.lr,
            T=args.T,
            schedule=args.schedule,
            base_ch=args.base_ch,
            seed=args.seed,
        )
        log.info("[%d/9] FINISH %s", i, name)

    (out_root / "ALL_TRAINING_COMPLETE.txt").write_text(
        "All 9 denoiser configurations completed.\n"
        "Do NOT choose a final checkpoint from validation loss.\n"
        "Next step: evaluate periodic checkpoints using generation metrics.\n"
    )
    log.info("ALL 9 CONFIGS COMPLETE.")
    print("\nALL 9 CONFIGS COMPLETE.")
    print("Next: generation-metric checkpoint selection.")


if __name__ == "__main__":
    main()
