# GAT Drift Prediction

Predicts per-floor X/Y inter-story drift for shear-wall buildings. Run the three
scripts in order. Each uses path constants set inside the file (no CLI arguments):
edit the constants, then run.

## 1. Convert JSON to graphs

Edit at the top of `O1graph_converter_updated_v3.py`:

```python
INPUT_DIR  = "<folder of building JSON files>"
OUTPUT_DIR = "<folder for generated graphs>"
```

Run:

```bash
python O1graph_converter_updated_v3.py
```

## 2. Train the model

Edit inside `train()` in `O2train_updated_v3.py`:

```python
folder = "<folder of graphs from step 1>"
```

Run:

```bash
python O2train_updated_v3.py
```

Produces `best_model.pt`.

## 3. Predict and evaluate

Edit at the bottom of `O3corrected_inference_paper_v3.py`:

```python
model_path    = "<path to best_model.pt>"
graph_folder  = "<folder of graphs to evaluate>"
output_folder = "<folder for results>"
```

Run:

```bash
python O3corrected_inference_paper_v3.py
```

Produces per-floor drift predictions, metric spreadsheets (MAE, RMSE, MAPE, R2),
and plots.
