"""
eval_ood.py
Evaluate the FROZEN final surrogate on the out-of-distribution (OOD) atypical
test set (4x3 / 5x5 unusual wall arrangements) -- Reviewer Group 2.

Uses the exact model, dataset and metric definitions from O2train_v9.py, and loads
the frozen operational checkpoint (final_surrogate/best_model_final.pt). It performs
inference ONLY (no training, no tuning) -- the model is frozen.

Reports per-building and aggregate R2/MAE/RMSE in the model's (transformed) space,
i.e. the same space the paper's OOD numbers were reported in, so the revised results
are directly comparable to the original OOD table.

Usage:
    python eval_ood.py \
        --checkpoint final_surrogate/best_model_final.pt \
        --test_dir "03. test folder/09. all_after_2steps_v3_v9_test"
"""

import os
import glob
import argparse
import numpy as np
import torch
from torch_geometric.data import Data

# reuse the exact definitions used for training/CV
from O2train_v9 import BuildingGATv2Model, BuildingDataset, compute_metrics


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    keys = ["node_features", "edge_features", "hidden_channels", "gat_layers",
            "gat_heads", "gru_layers", "transformer_layers", "transformer_heads",
            "dropout"]
    model_config = {k: v for k, v in config.items() if k in keys}
    model = BuildingGATv2Model(**model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("epoch"), config


def predict_building(model, building, device):
    building_gpu, targets, elevations = [], [], []
    for floor_data in building:
        building_gpu.append(Data(
            x=floor_data.x.to(device),
            edge_index=floor_data.edge_index.to(device),
            edge_attr=floor_data.edge_attr.to(device),
            y=floor_data.y.to(device)))
        targets.append(floor_data.y.mean(dim=0))
        elevations.append(float(floor_data.story_id.reshape(-1)[0].item()))
    if len(building_gpu) == 0:
        return None, None, None
    targets = torch.stack(targets).to(device)
    with torch.no_grad():
        preds = model(building_gpu)
    return preds.cpu().numpy(), targets.cpu().numpy(), np.asarray(elevations)


def write_building_xlsx(name, preds, targets, elevations, out_dir):
    """Write one per-building .xlsx in the exact format plot_drift_profile.py
    expects: columns Floor, Elevation, X-Dir_True, X-Dir_Pred, Y-Dir_True,
    Y-Dir_Pred. Rows are sorted by elevation DESCENDING and labelled F1..Fn
    (F1 = top), matching O3inference_v9.py's convention."""
    import pandas as pd
    # F1 = lowest elevation (ground), Fn = top. Rows written top-first
    # (descending elevation) so the profile reads top->bottom, but the F
    # label is tied to elevation rank (ascending), i.e. F1 at the base.
    asc = np.argsort(elevations)             # ascending elevation
    rank = {idx: r + 1 for r, idx in enumerate(asc)}   # idx -> F-number
    order = np.argsort(-elevations)          # write top (highest elev) first
    rows = []
    for idx in order:
        rows.append({
            "Floor": f"F{rank[idx]}",
            "Elevation": elevations[idx],
            "X-Dir_True": float(targets[idx, 0]),
            "X-Dir_Pred": float(preds[idx, 0]),
            "Y-Dir_True": float(targets[idx, 1]),
            "Y-Dir_Pred": float(preds[idx, 1]),
        })
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.xlsx")
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


def dir_metrics(pred, true):
    """MAE and MAPE per direction. pred/true shape (n_floors, 2) = [X, Y]."""
    pred = np.asarray(pred); true = np.asarray(true)
    out = {}
    for i, d in enumerate(["X", "Y"]):
        p, t = pred[:, i], true[:, i]
        mae = np.mean(np.abs(p - t))
        # MAPE with a small floor to avoid div-by-zero (matches typical convention)
        denom = np.where(np.abs(t) < 1e-8, 1e-8, np.abs(t))
        mape = np.mean(np.abs(p - t) / denom) * 100.0
        out[f"{d}_MAE"] = mae
        out[f"{d}_MAPE"] = mape
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="final_surrogate/best_model_final.pt")
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", default="ood_metrics.csv")
    ap.add_argument("--xlsx_dir", default=None,
                    help="if set, write one per-building .xlsx here "
                         "(format for plot_drift_profile.py)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ep, config = load_model(args.checkpoint, device)
    print(f"Loaded frozen final surrogate (best_epoch={ep}) on {device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    files = sorted(glob.glob(os.path.join(args.test_dir, "*.pt")))
    if not files:
        raise SystemExit(f"No .pt files in {args.test_dir}")

    rows = []
    print(f"{'building':<40}{'X_MAE':>8}{'Y_MAE':>8}{'X_MAPE':>9}{'Y_MAPE':>9}")
    for f in files:
        ds = BuildingDataset([f])
        building = ds[0]
        p, t, elev = predict_building(model, building, device)
        if p is None:
            continue
        m = dir_metrics(p, t)
        name = os.path.basename(f).replace("supergraph_before_", "").replace(".pt", "")
        print(f"{name:<40}{m['X_MAE']:>8.3f}{m['Y_MAE']:>8.3f}{m['X_MAPE']:>9.3f}{m['Y_MAPE']:>9.3f}")
        rows.append((name, m["X_MAE"], m["Y_MAE"], m["X_MAPE"], m["Y_MAPE"]))
        if args.xlsx_dir:
            xp = write_building_xlsx(name, p, t, elev, args.xlsx_dir)

    print("-" * 74)
    import numpy as _np
    arr = _np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    mean = arr.mean(axis=0)
    print(f"{'MEAN':<40}{mean[0]:>8.3f}{mean[1]:>8.3f}{mean[2]:>9.3f}{mean[3]:>9.3f}")

    import csv
    with open(args.out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["building", "X_MAE", "Y_MAE", "X_MAPE", "Y_MAPE"])
        w.writerows(rows)
        w.writerow(["MEAN", *mean.tolist()])
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()