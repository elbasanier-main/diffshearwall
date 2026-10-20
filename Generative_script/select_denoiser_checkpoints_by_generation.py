#!/usr/bin/env python3
"""
Generation-based checkpoint selection for the corrected 9-config denoiser ablation.

Selection within each hyperparameter config:
1) minimize mean projection_edit_fraction;
2) checkpoints within +5% of that minimum are eligible;
3) among eligible checkpoints choose highest mean diversity;
4) tie-break by highest uniqueness_rate, then lower raw_ratio_abs_error.

No auxiliary surrogate, standalone drift surrogate, joint validation loss, or
arbitrary weighted composite score is used for checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

DEFAULT_PLANS = [
    "2x10", "2x4", "2x7", "3x10", "3x3", "3x6", "3x7",
    "4x2", "4x3", "4x4", "5x5", "6x2", "6x4", "6x6",
    "7x3", "8x2", "8x3",
]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def torch_load_trusted(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_plan(s: str):
    a, b = s.lower().split("x", 1)
    return int(a), int(b)


def infer_base_ch(state_dict: dict, fallback: int = 32) -> int:
    for key in ("stem.0.weight", "module.stem.0.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return fallback


def normalize_state_dict(state_dict: dict) -> dict:
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def checkpoint_epoch(path: Path) -> int:
    m = re.search(r"diffusion_epoch_(\d+)\.pt$", path.name)
    if not m:
        raise ValueError(f"Cannot parse epoch from {path}")
    return int(m.group(1))


def valid_slot_keys(occ0: torch.Tensor, lx: int, ly: int):
    x = occ0.detach().cpu()
    keys = set()
    h = x[0, :3 * lx, :ly + 1]
    for idx in (h > 0.5).nonzero(as_tuple=False):
        keys.add(("H", int(idx[0]), int(idx[1])))
    v = x[1, :lx + 1, :3 * ly]
    for idx in (v > 0.5).nonzero(as_tuple=False):
        keys.add(("V", int(idx[0]), int(idx[1])))
    return keys


def wall_count(occ0: torch.Tensor, lx: int, ly: int) -> int:
    return len(valid_slot_keys(occ0, lx, ly))


def directional_balance(keys) -> float:
    nh = sum(1 for k in keys if k[0] == "H")
    nv = sum(1 for k in keys if k[0] == "V")
    mx = max(nh, nv)
    return min(nh, nv) / mx if mx else 0.0


def eccentricity_norm(keys, lx: int, ly: int) -> float:
    if not keys:
        return 0.0
    xs, ys = [], []
    for orient, xi, yi in keys:
        if orient == "H":
            x, y = (xi + 0.5) * 2.0, yi * 6.0
        else:
            x, y = xi * 6.0, (yi + 0.5) * 2.0
        xs.append(x)
        ys.append(y)
    cx, cy = mean(xs), mean(ys)
    gx, gy = lx * 3.0, ly * 3.0
    diag = math.hypot(lx * 6.0, ly * 6.0)
    return math.hypot(cx - gx, cy - gy) / diag if diag else 0.0


def symmetry_fraction(keys, lx: int, ly: int, axis: str) -> float:
    if not keys:
        return 0.0
    matched = 0
    for orient, xi, yi in keys:
        if axis == "x":
            mirror = ("H", 3 * lx - 1 - xi, yi) if orient == "H" else ("V", lx - xi, yi)
        else:
            mirror = ("H", xi, ly - yi) if orient == "H" else ("V", xi, 3 * ly - 1 - yi)
        matched += int(mirror in keys)
    return matched / len(keys)


def quadrant_cv(keys, lx: int, ly: int) -> float:
    if not keys:
        return 0.0
    gx, gy = lx * 3.0, ly * 3.0
    q = [0, 0, 0, 0]
    for orient, xi, yi in keys:
        if orient == "H":
            x, y = (xi + 0.5) * 2.0, yi * 6.0
        else:
            x, y = xi * 6.0, (yi + 0.5) * 2.0
        qi = (0 if x < gx else 1) + (0 if y < gy else 2)
        q[qi] += 1
    mq = sum(q) / 4.0
    if mq <= 0:
        return 0.0
    var = sum((v - mq) ** 2 for v in q) / 4.0
    return math.sqrt(var) / mq


def case_diversity(key_sets) -> float:
    if len(key_sets) < 2:
        return 0.0
    vals = []
    for a, b in itertools.combinations(key_sets, 2):
        union = len(a | b)
        iou = len(a & b) / union if union else 1.0
        vals.append(1.0 - iou)
    return mean(vals) if vals else 0.0


def uniqueness_rate(key_sets) -> float:
    if not key_sets:
        return 0.0
    return len({tuple(sorted(s)) for s in key_sets}) / len(key_sets)


def adjacent_floor_inconsistency(raw_occ: torch.Tensor, lx: int, ly: int) -> float:
    floors = raw_occ.shape[1]
    if floors <= 1:
        return 0.0
    max_slots = (ly + 1) * lx * 3 + (lx + 1) * ly * 3
    vals = []
    for f in range(floors - 1):
        a = valid_slot_keys(raw_occ[:, f], lx, ly)
        b = valid_slot_keys(raw_occ[:, f + 1], lx, ly)
        vals.append(len(a ^ b) / max(max_slots, 1))
    return mean(vals) if vals else 0.0


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def aggregate_case_rows(case_rows):
    keys = [
        "raw_ratio_abs_error_mean",
        "raw_wall_count_abs_error_mean",
        "projection_edit_fraction_mean",
        "post_ratio_abs_error_mean",
        "dir_balance_mean",
        "eccentricity_mean",
        "sym_x_mean",
        "sym_y_mean",
        "quad_cv_mean",
        "vertical_inconsistency_mean",
        "diversity",
        "uniqueness_rate",
    ]
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in case_rows]
        out[k] = mean(vals)
        out[k + "_case_std"] = pstdev(vals) if len(vals) > 1 else 0.0
    return out


def evaluate_checkpoint(ckpt_path, config_name, plans, ratios, seeds, candidates,
                        floors, T, schedule, device, deterministic, per_ckpt_dir):
    from disc_diffusion.model.denoiser import WallSlotDenoiser, MaskPredictor
    from disc_diffusion.model.diffusion import DiscreteDiffusion
    from disc_diffusion.model.surrogate import ConstraintProjector

    ckpt = torch_load_trusted(ckpt_path, map_location="cpu")
    sd = normalize_state_dict(ckpt["denoiser"])
    base_ch = infer_base_ch(sd, 32)
    model = WallSlotDenoiser(
        in_channels=6, base_ch=base_ch, cond_in_dim=5,
        cond_embed_dim=128, t_embed_dim=128,
    )
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    diffusion = DiscreteDiffusion(T=T, schedule=schedule, device=str(device))
    projector = ConstraintProjector()

    cand_rows, case_rows = [], []
    ep = checkpoint_epoch(ckpt_path)

    for lx, ly in plans:
        masks = MaskPredictor.analytical_masks(lx, ly, device)
        must_on = masks["must_on"].unsqueeze(0).to(device)
        must_off = masks["must_off"].unsqueeze(0).to(device)
        variable = masks["variable"].unsqueeze(0).to(device)
        max_slots = (ly + 1) * lx * 3 + (lx + 1) * ly * 3

        for ratio in ratios:
            target_walls = max(8, int(float(ratio) * max_slots))
            effective_target_ratio = target_walls / max(max_slots, 1)
            for seed in seeds:
                seed_everything(int(seed), deterministic)
                cond = torch.tensor(
                    [[lx / 10.0, ly / 10.0, floors / 20.0, float(ratio), 0.0]],
                    dtype=torch.float32, device=device,
                )
                rows_this_case, projected_sets = [], []

                with torch.no_grad():
                    for cand_idx in range(candidates):
                        raw = diffusion.sample(
                            model=model, cond=cond, must_on=must_on,
                            must_off=must_off, variable=variable,
                            num_floors=floors,
                        )[0]
                        raw0 = raw[:, 0]
                        raw_keys = valid_slot_keys(raw0, lx, ly)
                        raw_count = len(raw_keys)
                        raw_ratio = raw_count / max(max_slots, 1)

                        projected = projector.project(
                            occ=raw, must_on=masks["must_on"],
                            must_off=masks["must_off"], variable=masks["variable"],
                            lx=lx, ly=ly, shear_ratio=float(ratio), logits=None,
                        )
                        proj0 = projected[:, 0]
                        proj_keys = valid_slot_keys(proj0, lx, ly)
                        projected_sets.append(proj_keys)
                        proj_count = len(proj_keys)
                        proj_ratio = proj_count / max(max_slots, 1)
                        edit_count = len(raw_keys ^ proj_keys)

                        row = {
                            "config": config_name,
                            "checkpoint": str(ckpt_path),
                            "epoch": ep,
                            "lx": lx,
                            "ly": ly,
                            "ratio": float(ratio),
                            "target_wall_count": target_walls,
                            "effective_target_ratio": effective_target_ratio,
                            "seed": int(seed),
                            "candidate": cand_idx,
                            "raw_wall_count": raw_count,
                            "raw_ratio": raw_ratio,
                            "raw_ratio_abs_error": abs(raw_ratio - effective_target_ratio),
                            "raw_requested_ratio_abs_error": abs(raw_ratio - float(ratio)),
                            "raw_wall_count_abs_error": abs(raw_count - target_walls),
                            "projection_edit_count": edit_count,
                            "projection_edit_fraction": edit_count / max(max_slots, 1),
                            "post_wall_count": proj_count,
                            "post_ratio": proj_ratio,
                            "post_ratio_abs_error": abs(proj_ratio - effective_target_ratio),
                            "dir_balance": directional_balance(proj_keys),
                            "eccentricity": eccentricity_norm(proj_keys, lx, ly),
                            "sym_x": symmetry_fraction(proj_keys, lx, ly, "x"),
                            "sym_y": symmetry_fraction(proj_keys, lx, ly, "y"),
                            "quad_cv": quadrant_cv(proj_keys, lx, ly),
                            "vertical_inconsistency": adjacent_floor_inconsistency(raw, lx, ly),
                        }
                        cand_rows.append(row)
                        rows_this_case.append(row)

                case_rows.append({
                    "config": config_name,
                    "checkpoint": str(ckpt_path),
                    "epoch": ep,
                    "lx": lx,
                    "ly": ly,
                    "ratio": float(ratio),
                    "seed": int(seed),
                    "n_candidates": candidates,
                    "raw_ratio_abs_error_mean": mean(r["raw_ratio_abs_error"] for r in rows_this_case),
                    "raw_wall_count_abs_error_mean": mean(r["raw_wall_count_abs_error"] for r in rows_this_case),
                    "projection_edit_fraction_mean": mean(r["projection_edit_fraction"] for r in rows_this_case),
                    "post_ratio_abs_error_mean": mean(r["post_ratio_abs_error"] for r in rows_this_case),
                    "dir_balance_mean": mean(r["dir_balance"] for r in rows_this_case),
                    "eccentricity_mean": mean(r["eccentricity"] for r in rows_this_case),
                    "sym_x_mean": mean(r["sym_x"] for r in rows_this_case),
                    "sym_y_mean": mean(r["sym_y"] for r in rows_this_case),
                    "quad_cv_mean": mean(r["quad_cv"] for r in rows_this_case),
                    "vertical_inconsistency_mean": mean(r["vertical_inconsistency"] for r in rows_this_case),
                    "diversity": case_diversity(projected_sets),
                    "uniqueness_rate": uniqueness_rate(projected_sets),
                })

    write_csv(per_ckpt_dir / "candidate_metrics.csv", cand_rows)
    write_csv(per_ckpt_dir / "case_summary.csv", case_rows)
    summary = {
        "config": config_name,
        "checkpoint": str(ckpt_path),
        "epoch": ep,
        "n_cases": len(case_rows),
        "n_candidates_total": len(cand_rows),
        **aggregate_case_rows(case_rows),
    }
    write_csv(per_ckpt_dir / "checkpoint_summary.csv", [summary])
    (per_ckpt_dir / "COMPLETE.txt").write_text(json.dumps(summary, indent=2) + "\n")

    del model, diffusion, projector, ckpt, sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def read_one_summary(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row in {path}")
    r = rows[0]
    for k in list(r.keys()):
        if k in ("config", "checkpoint"):
            continue
        try:
            r[k] = float(r[k])
        except Exception:
            pass
    r["epoch"] = int(float(r["epoch"]))
    r["n_cases"] = int(float(r["n_cases"]))
    r["n_candidates_total"] = int(float(r["n_candidates_total"]))
    return r


def select_within_config(rows, tolerance_frac: float):
    min_edit = min(float(r["projection_edit_fraction_mean"]) for r in rows)
    threshold = min_edit + 1e-12 if min_edit <= 1e-12 else min_edit * (1.0 + tolerance_frac)
    eligible = [r for r in rows if float(r["projection_edit_fraction_mean"]) <= threshold + 1e-15]
    eligible = sorted(
        eligible,
        key=lambda r: (
            -float(r["diversity"]),
            -float(r["uniqueness_rate"]),
            float(r["raw_ratio_abs_error_mean"]),
            int(r["epoch"]),
        ),
    )
    chosen = dict(eligible[0])
    chosen["min_projection_edit_fraction_in_config"] = min_edit
    chosen["eligibility_threshold"] = threshold
    chosen["n_eligible"] = len(eligible)
    chosen["selection_rule"] = (
        f"edit<=min*(1+{tolerance_frac:g}); then max diversity; "
        "then max uniqueness; then min raw ratio error"
    )
    return chosen


def main():
    ap = argparse.ArgumentParser(description="Select periodic denoiser checkpoints using generation metrics.")
    ap.add_argument("--repo", default="/home/elbasanier/CGVAE_RL_Tran")
    ap.add_argument("--ckpt-root", default="/home/elbasanier/CGVAE_RL_Tran/disc_diffusion/outputs/ablation_retrain_genmetric")
    ap.add_argument("--out-dir", default="/home/elbasanier/CGVAE_RL_Tran/disc_diffusion/outputs/checkpoint_selection_genmetric")
    ap.add_argument("--plans", nargs="+", default=DEFAULT_PLANS)
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.40, 0.45, 0.50])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345])
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--floors", type=int, default=10)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--tolerance-frac", type=float, default=0.05)
    ap.add_argument("--max-checkpoints", type=int, default=0, help="Debug only; 0 means all.")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    ckpt_root = Path(args.ckpt_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        raise SystemExit(f"Repo not found: {repo}")
    if not ckpt_root.exists():
        raise SystemExit(f"Checkpoint root not found: {ckpt_root}")

    sys.path.insert(0, str(repo))
    os.chdir(repo)
    device = torch.device("cpu" if args.device.startswith("cuda") and not torch.cuda.is_available() else args.device)
    plans = [parse_plan(s) for s in args.plans]
    ckpts = sorted(ckpt_root.glob("denoiser__w_ratio_*_w_vert_*/diffusion_epoch_*.pt"))
    if args.max_checkpoints > 0:
        ckpts = ckpts[:args.max_checkpoints]
    if not ckpts:
        raise SystemExit(f"No periodic checkpoints found under {ckpt_root}")

    expected_cases = len(plans) * len(args.ratios) * len(args.seeds)
    expected_per_ckpt = expected_cases * args.candidates
    manifest = {
        "repo": str(repo),
        "ckpt_root": str(ckpt_root),
        "out_dir": str(out_dir),
        "device": str(device),
        "plans": [f"{a}x{b}" for a, b in plans],
        "ratios": args.ratios,
        "selection_seeds": args.seeds,
        "held_out_confirmation_seed": 3456,
        "candidates_per_case": args.candidates,
        "floors": args.floors,
        "T": args.T,
        "schedule": args.schedule,
        "deterministic": bool(args.deterministic),
        "tolerance_frac": args.tolerance_frac,
        "n_checkpoints": len(ckpts),
        "cases_per_checkpoint": expected_cases,
        "layouts_per_checkpoint": expected_per_ckpt,
        "expected_layouts_total": len(ckpts) * expected_per_ckpt,
        "auxiliary_surrogate": "NOT_USED",
        "standalone_drift_surrogate": "NOT_USED",
        "validation_loss_for_selection": "NOT_USED",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("=" * 88)
    print("GENERATION-BASED PERIODIC CHECKPOINT SELECTION")
    print("=" * 88)
    print(f"Periodic checkpoints  : {len(ckpts)}")
    print(f"Plans                 : {len(plans)}")
    print(f"Ratios                : {args.ratios}")
    print(f"Selection seeds       : {args.seeds}")
    print("Held-out seed 3456    : NOT USED in this stage")
    print(f"Candidates/case       : {args.candidates}")
    print(f"Cases/checkpoint      : {expected_cases}")
    print(f"Layouts/checkpoint    : {expected_per_ckpt}")
    print(f"Expected layouts      : {len(ckpts) * expected_per_ckpt}")
    print("Aux surrogate         : NOT USED")
    print("Standalone surrogate  : NOT USED")
    print("=" * 88)

    per_root = out_dir / "per_checkpoint"
    per_root.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    for i, ckpt_path in enumerate(ckpts, start=1):
        cfg = ckpt_path.parent.name
        ep = checkpoint_epoch(ckpt_path)
        pc_dir = per_root / cfg / f"epoch_{ep:03d}"
        pc_dir.mkdir(parents=True, exist_ok=True)
        complete = pc_dir / "COMPLETE.txt"
        summary_csv = pc_dir / "checkpoint_summary.csv"
        if complete.exists() and summary_csv.exists():
            print(f"[{i:03d}/{len(ckpts):03d}] SKIP complete {cfg} epoch={ep}")
            continue

        for p in (pc_dir / "candidate_metrics.csv", pc_dir / "case_summary.csv",
                  pc_dir / "checkpoint_summary.csv", pc_dir / "COMPLETE.txt"):
            if p.exists():
                p.unlink()

        print(f"[{i:03d}/{len(ckpts):03d}] START {cfg} epoch={ep}")
        t0 = time.time()
        s = evaluate_checkpoint(
            ckpt_path, cfg, plans, args.ratios, args.seeds, args.candidates,
            args.floors, args.T, args.schedule, device, args.deterministic, pc_dir,
        )
        print(
            f"[{i:03d}/{len(ckpts):03d}] DONE  {cfg} epoch={ep} | "
            f"edit={s['projection_edit_fraction_mean']:.6f} "
            f"div={s['diversity']:.6f} uniq={s['uniqueness_rate']:.6f} | "
            f"{time.time()-t0:.1f}s"
        )

    summaries = []
    for summary_csv in sorted(per_root.glob("*/epoch_*/checkpoint_summary.csv")):
        if (summary_csv.parent / "COMPLETE.txt").exists():
            summaries.append(read_one_summary(summary_csv))
    write_csv(out_dir / "checkpoint_summary.csv", summaries)

    by_cfg = {}
    for r in summaries:
        by_cfg.setdefault(r["config"], []).append(r)

    selected, selected_dir = [], out_dir / "selected_checkpoints"
    selected_dir.mkdir(exist_ok=True)
    for cfg in sorted(by_cfg):
        chosen = select_within_config(sorted(by_cfg[cfg], key=lambda r: r["epoch"]), args.tolerance_frac)
        src = Path(chosen["checkpoint"])
        dst = selected_dir / f"{cfg}__epoch_{chosen['epoch']:03d}.pt"
        if not dst.exists() or src.stat().st_size != dst.stat().st_size:
            shutil.copy2(src, dst)
        chosen["selected_copy"] = str(dst)
        selected.append(chosen)
    write_csv(out_dir / "selected_checkpoints.csv", selected)

    lines = ["Generation-based checkpoint selection", "=" * 72]
    for r in selected:
        lines.append(
            f"{r['config']} -> epoch {int(r['epoch']):03d} | "
            f"edit={float(r['projection_edit_fraction_mean']):.6f} | "
            f"div={float(r['diversity']):.6f} | uniq={float(r['uniqueness_rate']):.6f}"
        )
    lines += ["", "No final denoiser config is declared at this stage.",
              "Next: compare the 9 selected checkpoints on the fuller benchmark,",
              "then freeze one final denoiser and use seed 3456 only for confirmation."]
    (out_dir / "selection_report.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nTotal elapsed: {(time.time()-t_all)/3600:.2f} h")
    print(f"Checkpoint summary : {out_dir / 'checkpoint_summary.csv'}")
    print(f"Selected 9         : {out_dir / 'selected_checkpoints.csv'}")
    print(f"Report             : {out_dir / 'selection_report.txt'}")


if __name__ == "__main__":
    main()
