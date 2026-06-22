# Discrete Diffusion Generation

Generates candidate wall layouts with a conditional discrete diffusion model and
ranks them with the GAT drift predictor. See the main README for how this
component fits into the framework.

Files: `run.py` (CLI), `trainer.py` (training), `pipeline.py` (inference).

## Build masks

```bash
python run.py build_masks --data_root <DATA> --out masks.pt
```

## Train

```bash
python run.py train \
    --data_root <DATA> --masks masks.pt \
    --epochs 100 --batch_size 16 \
    --T 200 --schedule cosine \
    --ckpt_dir outputs/diffusion_checkpoints
```

## Generate

```bash
python run.py generate \
    --masks masks.pt \
    --ckpt outputs/diffusion_checkpoints/best_diffusion.pt \
    --lx 5 --ly 4 --floors 10 --candidates 16 \
    --drift_checkpoint <DRIFT_MODEL> \
    --graph_converter  <GRAPH_CONVERTER> \
    --inference_script <INFERENCE_SCRIPT> \
    --save_before_json --save_etabs_xlsx
```

Provide all three drift-workflow paths (`--drift_checkpoint`, `--graph_converter`,
`--inference_script`) to rank candidates with the GAT drift predictor. If they are
omitted, ranking falls back to the internal surrogate (less accurate). Candidates
are ranked by `0.40*x_drift + 0.35*y_drift + 0.15*symmetry + 0.10*constructability`
(lower drift is better).

## Outputs

Each run writes a timestamped folder containing the best layout (`result.json`), a
text report, per-candidate plan images and a comparison table, and — when enabled —
a `before_*.json`, an ETABS-importable `etabs_*.xlsx`, and per-candidate inference
outputs.
