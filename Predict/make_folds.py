"""
make_folds.py
Materialize k-fold train/val/test splits to file lists you can INSPECT before
training (instead of splitting inside the training loop).

Two modes:
  * traj_kfold (default) -- Kieu B: within EACH plan, cut the sorted trajectory
    into n_folds contiguous blocks; fold i uses block i as test, block (i+1) as val,
    the rest as train. With n_folds=10 => ~80/10/10 per fold, EVERY plan appears in
    train/val/test, contiguous blocks so no near-duplicate neighbour leaks, and the
    test blocks tile the whole trajectory over the folds.
  * plan_kfold -- Kieu A: leave-plan-out (whole plan types held out as test).

Train-only oversampling (repeat) is baked into train.txt; val/test stay unique.

Output (under --out_dir):
    fold0/train.txt  fold0/val.txt  fold0/test.txt   ... foldN/...
    manifest.csv                 # per-fold counts, for eyeballing
    plan_presence.csv            # per-plan: #files in train/val/test each fold

Usage:
    python make_folds.py --folder "all_after_2steps_v3_v9" \
        --mode traj_kfold --n_folds 10 --train_repeat 3 --out_dir folds_v9 --seed 42

Then train a fold by reading the lists:
    python O2train_repeat_v9.py --fold_dir folds_v9/fold0
"""

import os
import csv
import argparse
from collections import Counter, defaultdict

from split_utils import split_trajectory_kfold, split_trajectory_kfold_strided, split_trajectory_kfold_purged, split_final, split_kfold, parse_name


def _plans(files):
    return sorted({parse_name(f)[0] for f in files})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="folder of .pt graph files")
    ap.add_argument("--mode", default="traj_kfold",
                    choices=["traj_kfold", "traj_kfold_strided", "traj_kfold_purged", "plan_kfold", "final"])
    ap.add_argument("--n_folds", type=int, default=10,
                    help="10 => 80/10/10 per fold for traj_kfold")
    ap.add_argument("--purge", type=int, default=2,
                    help="purge +/- k trajectory neighbours around each test point (traj_kfold_purged)")
    ap.add_argument("--train_repeat", type=int, default=1,
                    help="repeat TRAIN files x N (val/test stay unique)")
    ap.add_argument("--out_dir", default="folds_v9")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_files = sorted(os.path.join(args.folder, f)
                       for f in os.listdir(args.folder) if f.endswith(".pt"))
    if not all_files:
        raise SystemExit(f"No .pt files in {args.folder}")
    print(f"Found {len(all_files)} .pt files in {args.folder}")

    plan_counts = Counter(parse_name(f)[0] for f in all_files)
    real = [f for f in all_files if parse_name(f)[1] >= 0]
    print(f"{len(plan_counts)} plan types; {len(real)} files have a real "
          f"'between' step ({len(all_files) - len(real)} min/max/u aliases dropped "
          f"in traj mode).")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- FINAL operational split (single split, NOT evaluation) ----
    if args.mode == "final":
        tr, va, _ = split_final(all_files, val_frac=0.10, seed=args.seed)
        n_tr = len(tr)
        tr_out = tr * max(1, args.train_repeat)
        fdir = os.path.join(args.out_dir, "fold0")
        os.makedirs(fdir, exist_ok=True)
        with open(os.path.join(fdir, "train.txt"), "w") as f:
            f.write("\n".join(tr_out) + "\n")
        with open(os.path.join(fdir, "val.txt"), "w") as f:
            f.write("\n".join(va) + "\n")
        open(os.path.join(fdir, "test.txt"), "w").close()   # empty: no test
        tot = n_tr + len(va)
        print(f"FINAL split: train {n_tr}u x{args.train_repeat}={len(tr_out)} "
              f"({100*n_tr/tot:.1f}%) | val {len(va)} ({100*len(va)/tot:.1f}%, "
              f"early-stopping only) | test 0")
        print("NOTE: this split is operational only; do NOT report its performance. "
              "Generalization is estimated by the strided/purged 10-fold CV.")
        print(f"Wrote {args.out_dir}/fold0/")
        return


    def do_split(fold):
        if args.mode == "traj_kfold":
            return split_trajectory_kfold(all_files, n_folds=args.n_folds, fold=fold)
        elif args.mode == "traj_kfold_strided":
            return split_trajectory_kfold_strided(all_files, n_folds=args.n_folds, fold=fold)
        elif args.mode == "traj_kfold_purged":
            return split_trajectory_kfold_purged(all_files, n_folds=args.n_folds, fold=fold, purge=args.purge)
        else:
            return split_kfold(all_files, n_folds=args.n_folds, fold=fold, seed=args.seed)

    rows = []
    presence = defaultdict(dict)   # plan -> {f'f{k}_train': n, ...}
    covered_test = set()

    for k in range(args.n_folds):
        tr, va, te = do_split(k)

        # ---- integrity: mutually-exclusive FILE sets across train/val/test ----
        # NOTE: exact-file exclusivity is necessary but NOT sufficient for this
        # dataset. Reviewers' concern is NEAR-DUPLICATE configs (differ by ~1 wall),
        # i.e. distinct .pt files that are structurally almost identical. Strided
        # assignment mitigates but does not remove that; the traj_kfold_purged mode
        # (sensitivity test) additionally purges +/- k trajectory neighbours.
        s_tr, s_va, s_te = set(tr), set(va), set(te)
        assert not (s_tr & s_va), f"fold {k}: train/val file overlap"
        assert not (s_tr & s_te), f"fold {k}: train/test file overlap"
        assert not (s_va & s_te), f"fold {k}: val/test file overlap"
        covered_test |= s_te

        # per-plan presence (should be >0 in all three for traj mode)
        for tag, lst in [("train", tr), ("val", va), ("test", te)]:
            cnt = Counter(parse_name(f)[0] for f in lst)
            for p in plan_counts:
                presence[p][f"f{k}_{tag}"] = cnt.get(p, 0)

        n_train_unique = len(tr)
        tr_out = tr * max(1, args.train_repeat)   # train-only repeat

        fdir = os.path.join(args.out_dir, f"fold{k}")
        os.makedirs(fdir, exist_ok=True)
        for name, lst in [("train.txt", tr_out), ("val.txt", va), ("test.txt", te)]:
            with open(os.path.join(fdir, name), "w") as f:
                f.write("\n".join(lst) + "\n")

        tot = n_train_unique + len(va) + len(te)
        rows.append(dict(
            fold=k, train_unique=n_train_unique, train_repeated=len(tr_out),
            val=len(va), test=len(te),
            pct_train=round(100 * n_train_unique / tot, 1),
            pct_val=round(100 * len(va) / tot, 1),
            pct_test=round(100 * len(te) / tot, 1),
            n_plans_in_test=len(_plans(te)),
        ))
        print(f"fold{k}: train {n_train_unique}u x{args.train_repeat}={len(tr_out)} | "
              f"val {len(va)} | test {len(te)}  "
              f"[{rows[-1]['pct_train']}/{rows[-1]['pct_val']}/{rows[-1]['pct_test']}] "
              f"plans_in_test={len(_plans(te))}")

    if args.mode == "plan_kfold":
        assert covered_test == set(plan_counts), "not every plan tested once"
        print(f"\nOK (plan_kfold): all {len(plan_counts)} plans held out as test once.")
    else:
        # in traj mode every plan should appear in every fold's train/val/test
        bad = [p for p in plan_counts
               if any(presence[p][f"f{k}_test"] == 0 for k in range(args.n_folds))]
        if bad:
            print(f"\nWARNING: plans missing from some test fold (too few configs): {bad}")
        else:
            print(f"\nOK (traj_kfold): every plan appears in train/val/test in all "
                  f"{args.n_folds} folds; test tiles the full trajectory.")

    with open(os.path.join(args.out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with open(os.path.join(args.out_dir, "plan_presence.csv"), "w", newline="") as f:
        cols = ["plan", "total"] + [f"f{k}_{t}" for k in range(args.n_folds)
                                     for t in ("train", "val", "test")]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in sorted(plan_counts):
            row = {"plan": p, "total": plan_counts[p]}
            row.update(presence[p])
            w.writerow(row)

    print(f"Wrote lists + manifest.csv + plan_presence.csv to {args.out_dir}/")


if __name__ == "__main__":
    main()
