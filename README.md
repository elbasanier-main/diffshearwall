<!-- Place the framework overview image at docs/overview.png (PNG or SVG), then it renders below. -->
<p align="center">
  <img src="docs/overview.png" alt="Framework overview" width="820">
</p>

# RC Shear Wall Layout Generation & Drift Prediction

An integrated framework for the **early-stage design of reinforced-concrete shear
wall buildings**. From only the plan dimensions, the number of floors, and a
target shear-wall ratio, it generates candidate wall layouts, predicts the
wind-induced inter-story drift of each in both principal directions, ranks them,
and exports the selected design as a labeled drawing and an **ETABS-importable
spreadsheet** — with no finite element analysis required at inference.

The pipeline is deployed as a Model Context Protocol (MCP) server, so the whole
workflow can be driven through natural language from any compatible AI assistant.

> Part of a larger research project, provided here as a **working sample** that
> will be extended in later releases. It accompanies a manuscript currently under
> peer review; the full title, citation, and trained checkpoints will be added
> once review completes, and the authors can be contacted at that stage.

---

## Built on ETABS data

The models are trained on structural data extracted from **ETABS** models, and the
selected layout is exported back as an ETABS-importable spreadsheet. The framework
fits directly into a standard structural-analysis workflow — ETABS in, ETABS out —
with the learned models replacing repeated finite element runs during the
candidate search.

---

## Model structure

The framework couples two learned models with a lightweight scoring and constraint
layer:

- **Layout generator — discrete diffusion.** A conditional discrete diffusion
  model proposes candidate wall layouts that meet the requested plan size, floor
  count, and target shear-wall ratio, without architectural drawings.
- **Drift predictor — graph neural network.** A GATv2-based surrogate represents
  each building as a graph of walls and predicts per-floor X- and Y-direction
  drift, replacing finite element analysis during the search.
- **Scoring and selection.** A structural surrogate and a constraint projector
  enforce the target ratio and rank candidates; the best layout is exported as a
  drawing and an ETABS spreadsheet.

Generation and evaluation run in one pass: the diffusion model proposes, the graph
model evaluates, and the highest-ranked layout is returned. Under ASCE 7-22 wind
loading the drift surrogate reaches R2 = 0.997.

```
conditions  ->  diffusion generator  ->  candidates  ->  GAT drift predictor  ->  ranked best  ->  ETABS export
```

---

## Documentation

| File | Component | Contents |
|------|-----------|----------|
| `GAT_DRIFT_PREDICTION.md` | Graph drift predictor | Commands to build graphs and run prediction |
| `DIFFUSION_GENERATION.md` | Diffusion generator | Commands to train and generate layouts |

This README gives the overall picture; each component file holds only the commands
needed to run that part.
