# -*- coding: utf-8 -*-
"""
Inference Script for Paper Publication (ICCES 2026)
- Uses SAME model architecture as train_updated.py
- Adds Base floor (elevation 0, drift = 0) to results
- Per-plan-type aggregation from filename
- Floor-wise MAPE aggregation across all buildings
- R^2 scores included
- Publication-quality plots
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from sklearn.metrics import r2_score


# ============================================================
# EXACT COPY OF TRAINING MODEL ARCHITECTURE
# ============================================================

class ResidualGATv2Layer(nn.Module):
    """GATv2 layer with residual connection"""
    def __init__(self, in_channels, out_channels, edge_dim=8, heads=4, dropout=0.1):
        super().__init__()
        self.gatv2 = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True
        )
        self.norm = nn.LayerNorm(out_channels * heads)
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels * heads:
            self.residual_proj = nn.Linear(in_channels, out_channels * heads)
        else:
            self.residual_proj = None
        
    def forward(self, x, edge_index, edge_attr):
        residual = x
        x = self.gatv2(x, edge_index, edge_attr)
        x = self.norm(x)
        x = F.elu(x)
        x = self.dropout(x)
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        x = x + residual
        return x


class FloorGATEncoder(nn.Module):
    """GATv2 encoder with residual connections for processing walls within each floor"""
    def __init__(self, in_channels, hidden_channels, edge_dim=8, num_layers=3, 
                 heads=4, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(in_channels, hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)
        
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            ResidualGATv2Layer(hidden_channels, hidden_channels // heads, 
                              edge_dim, heads, dropout)
        )
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                ResidualGATv2Layer(hidden_channels, hidden_channels // heads,
                                  edge_dim, heads, dropout)
            )
        
        self.final_gat = GATv2Conv(hidden_channels, hidden_channels, heads=1, 
                                   edge_dim=edge_dim, dropout=dropout, concat=False)
        self.final_norm = nn.LayerNorm(hidden_channels)
        self.pool_projection = nn.Linear(hidden_channels * 3, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = x[:, :self.input_projection.in_features]  # slice to model input size
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        for layer in self.gat_layers:
            x = layer(x, edge_index, edge_attr)
        
        residual = x
        x = self.final_gat(x, edge_index, edge_attr)
        x = self.final_norm(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = x + residual
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        
        x = torch.cat([x_mean, x_max, x_sum], dim=-1)
        x = self.pool_projection(x)
        return x


class BuildingGATv2Model(nn.Module):
    """Hierarchical model: GATv2 -> GRU -> Transformer with residual connections"""
    def __init__(self, node_features=19, edge_features=8, hidden_channels=128,
                 gat_layers=3, gat_heads=4, gru_layers=2, transformer_layers=2,
                 transformer_heads=8, dropout=0.1):
        super().__init__()
        
        self.floor_encoder = FloorGATEncoder(
            node_features, hidden_channels, edge_features, 
            gat_layers, gat_heads, dropout
        )
        self.floor_position_encoding = nn.Embedding(20, hidden_channels)
        
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=True
        )
        self.gru_projection = nn.Linear(hidden_channels * 2, hidden_channels)
        self.gru_norm = nn.LayerNorm(hidden_channels)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=transformer_heads,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, building_data):
        floor_representations = []
        
        for floor_idx, floor_data in enumerate(building_data):
            batch = torch.zeros(floor_data.x.size(0), dtype=torch.long, device=floor_data.x.device)
            floor_repr = self.floor_encoder(
                floor_data.x, floor_data.edge_index, floor_data.edge_attr, batch
            )
            pos_encoding = self.floor_position_encoding(
                torch.tensor([floor_idx], device=floor_data.x.device)
            )
            floor_repr = floor_repr + pos_encoding
            floor_representations.append(floor_repr.squeeze(0))
        
        x = torch.stack(floor_representations, dim=0).unsqueeze(0)
        
        gru_input = x
        x, _ = self.gru(x)
        x = self.gru_projection(x)
        x = self.gru_norm(x)
        x = x + gru_input
        
        transformer_input = x
        x = self.transformer(x)
        x = x + transformer_input
        
        predictions = self.output_mlp(x)
        return predictions.squeeze(0)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_subgraphs_with_edges(graph, edge_dim=8):
    """Extract subgraphs per floor USING THE EXISTING EDGES from converter."""
    device = graph.x.device
    elevations = torch.unique(graph.story_id, sorted=True)  # ascending F1->F10, matches training
    
    subgraphs = []
    for elev in elevations:
        floor_mask = graph.story_id == elev
        floor_indices = torch.where(floor_mask)[0]
        
        if len(floor_indices) < 2:
            continue
        
        floor_x = graph.x[floor_mask]
        floor_y = graph.y[floor_mask] if hasattr(graph, 'y') and graph.y is not None else None
        
        node_map = torch.full((graph.x.size(0),), -1, dtype=torch.long, device=device)
        for new_idx, old_idx in enumerate(floor_indices):
            node_map[old_idx] = new_idx
        
        if graph.edge_index.size(1) > 0:
            src_in_floor = node_map[graph.edge_index[0]] >= 0
            dst_in_floor = node_map[graph.edge_index[1]] >= 0
            edge_mask = src_in_floor & dst_in_floor
            
            if edge_mask.any():
                floor_edges = graph.edge_index[:, edge_mask]
                floor_edges_remapped = torch.stack([
                    node_map[floor_edges[0]], node_map[floor_edges[1]]
                ])
                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    floor_edge_attr = graph.edge_attr[edge_mask]
                else:
                    floor_edge_attr = torch.ones(floor_edges_remapped.size(1), edge_dim, device=device)
            else:
                floor_edges_remapped = torch.zeros((2, 0), dtype=torch.long, device=device)
                floor_edge_attr = torch.zeros((0, edge_dim), dtype=torch.float, device=device)
        else:
            floor_edges_remapped = torch.zeros((2, 0), dtype=torch.long, device=device)
            floor_edge_attr = torch.zeros((0, edge_dim), dtype=torch.float, device=device)
        
        sub_data = Data(
            x=floor_x, edge_index=floor_edges_remapped,
            edge_attr=floor_edge_attr, y=floor_y,
            story_id=elev.expand(len(floor_indices))
        )
        sub_data.elevation = elev.item()
        subgraphs.append(sub_data)
    
    return subgraphs


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model = BuildingGATv2Model(
        node_features=config.get('node_features', 18),
        edge_features=config.get('edge_features', 8),
        hidden_channels=config.get('hidden_channels', 128),
        gat_layers=config.get('gat_layers', 4),
        gat_heads=config.get('gat_heads', 4),
        gru_layers=config.get('gru_layers', 3),
        transformer_layers=config.get('transformer_layers', 3),
        transformer_heads=config.get('transformer_heads', 8),
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"[INFO] Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"[INFO] Config: {config}")
    return model, config


def parse_plan_type(filename):
    """Extract plan type from filename, e.g. '5x5' from 'supergraph_before_test_5x5_between_176'"""
    match = re.search(r'(\d+x\d+)', filename)
    return match.group(1) if match else "unknown"


def calculate_error_metrics(true_values, pred_values):
    """Calculate error metrics including R^2."""
    true_values = np.array(true_values, dtype=float)
    pred_values = np.array(pred_values, dtype=float)
    
    abs_error = np.abs(true_values - pred_values)
    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean((true_values - pred_values) ** 2))
    
    mask = true_values != 0
    if mask.any():
        mape = np.mean(np.abs((true_values[mask] - pred_values[mask]) / true_values[mask])) * 100
    else:
        mape = np.nan
    
    pct_error = np.zeros_like(true_values)
    pct_error[mask] = np.abs((true_values[mask] - pred_values[mask]) / true_values[mask]) * 100
    
    # R^2 score
    if len(true_values) > 1:
        r2 = r2_score(true_values, pred_values)
    else:
        r2 = np.nan
    
    return {
        'abs_error': abs_error,
        'pct_error': pct_error,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
    }


def add_base_floor(results):
    """
    Add Base floor (elevation 0) with drift ratio = 0 for both directions.
    This represents the fixed base condition of the building.
    """
    base_row = {
        "Floor": "Base",
        "Elevation": 0.0,
        "X-Dir_Pred": 0.0,
        "Y-Dir_Pred": 0.0,
    }
    # If true values exist, add them too
    if any("X-Dir_True" in r for r in results):
        base_row["X-Dir_True"] = 0.0
        base_row["Y-Dir_True"] = 0.0
        base_row["X-Dir_AbsError"] = 0.0
        base_row["Y-Dir_AbsError"] = 0.0
        base_row["X-Dir_PctError"] = 0.0
        base_row["Y-Dir_PctError"] = 0.0
    
    # Add floor labels to existing results (sorted top to bottom)
    sorted_results = sorted(results, key=lambda r: r["Elevation"], reverse=True)
    for i, r in enumerate(sorted_results):
        r["Floor"] = f"F{i+1}"
    
    # Append base at the end (bottom)
    sorted_results.append(base_row)
    return sorted_results


# ============================================================
# PUBLICATION-QUALITY PLOT FUNCTIONS
# ============================================================

# Common style settings for all plots
PLOT_STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}


def apply_plot_style():
    """Apply publication-quality style to all plots."""
    plt.rcParams.update(PLOT_STYLE)


def plot_building_comparison(results_with_base, output_folder, building_name=None):
    """
    Plot comparison for a single building.
    Row 1: 3 charts side-by-side (drift profile, X-bar, Y-bar)
    Row 2: Error summary table centered below (easy to crop for paper)
    Base floor excluded from plots (values are in transformed space).
    """
    apply_plot_style()
    import matplotlib.gridspec as gridspec

    df = pd.DataFrame(results_with_base)
    has_true = 'X-Dir_True' in df.columns
    if not has_true:
        return None, None

    title_suffix = f" - {building_name}" if building_name else ""

    # Exclude Base floor from plotting (transformed space, Base=0 is artificial)
    df_plot = df[df['Floor'] != 'Base'].copy()

    elevations = df_plot['Elevation'].values
    floor_labels = df_plot['Floor'].values
    num_floors = len(floor_labels)
    floor_indices = np.arange(num_floors)

    # Metrics
    x_metrics = calculate_error_metrics(df_plot['X-Dir_True'].values, df_plot['X-Dir_Pred'].values)
    y_metrics = calculate_error_metrics(df_plot['Y-Dir_True'].values, df_plot['Y-Dir_Pred'].values)

    # Color scheme: X = blue, Y = red (matching line chart colors)
    c_x_true, c_x_pred = '#2c5f8a', '#7fb3e0'
    c_y_true, c_y_pred = '#b22222', '#f08080'

    # ================================================================
    # Layout: top row = 3 charts, bottom row = centered table
    # ================================================================
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1], hspace=0.35, wspace=0.32)

    # ---- (a) Drift-Elevation Profile ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df_plot['X-Dir_True'], elevations, '-o', color='#1f77b4',
             label='X-Dir (True)', linewidth=2, markersize=6, zorder=3)
    ax1.plot(df_plot['X-Dir_Pred'], elevations, '--s', color='#aec7e8',
             label='X-Dir (Pred)', linewidth=2, markersize=6,
             markerfacecolor='white', markeredgecolor='#1f77b4',
             markeredgewidth=1.5, zorder=2)
    ax1.plot(df_plot['Y-Dir_True'], elevations, '-o', color='#d62728',
             label='Y-Dir (True)', linewidth=2, markersize=6, zorder=3)
    ax1.plot(df_plot['Y-Dir_Pred'], elevations, '--s', color='#ff9896',
             label='Y-Dir (Pred)', linewidth=2, markersize=6,
             markerfacecolor='white', markeredgecolor='#d62728',
             markeredgewidth=1.5, zorder=2)

    ax1.set_xlabel('Transformed Drift Value')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_title('(a) Drift profile vs. elevation')
    ax1.legend(loc='center', framealpha=0.9, edgecolor='gray', fontsize=7,
               borderpad=0.3, labelspacing=0.15, handlelength=1.2, handletextpad=0.3)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylim(ax1.get_ylim())
    ax1_twin.set_yticks(elevations)
    ax1_twin.set_yticklabels(floor_labels, fontsize=9)
    ax1_twin.set_ylabel('Story')

    # ---- (b) X-Direction Bar Chart ----
    ax2 = fig.add_subplot(gs[0, 1])
    bar_w = 0.35
    ax2.bar(floor_indices - bar_w/2, df_plot['X-Dir_True'], bar_w,
            label='True', color=c_x_true, edgecolor='white', linewidth=0.5)
    ax2.bar(floor_indices + bar_w/2, df_plot['X-Dir_Pred'], bar_w,
            label='Predicted', color=c_x_pred, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Story')
    ax2.set_ylabel('X-Direction Transformed Value')
    ax2.set_title(f'(b) X-Direction (MAPE = {x_metrics["mape"]:.2f}%)')
    ax2.set_xticks(floor_indices)
    ax2.set_xticklabels(floor_labels, fontsize=9)
    ax2.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_axisbelow(True)

    # ---- (c) Y-Direction Bar Chart (RED tones) ----
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(floor_indices - bar_w/2, df_plot['Y-Dir_True'], bar_w,
            label='True', color=c_y_true, edgecolor='white', linewidth=0.5)
    ax3.bar(floor_indices + bar_w/2, df_plot['Y-Dir_Pred'], bar_w,
            label='Predicted', color=c_y_pred, edgecolor='white', linewidth=0.5)
    ax3.set_xlabel('Story')
    ax3.set_ylabel('Y-Direction Transformed Value')
    ax3.set_title(f'(c) Y-Direction (MAPE = {y_metrics["mape"]:.2f}%)')
    ax3.set_xticks(floor_indices)
    ax3.set_xticklabels(floor_labels, fontsize=9)
    ax3.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_axisbelow(True)

    # ---- Row 2: Error Summary Table (centered, spans all 3 columns) ----
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')

    table_data = [
        ['Metric', 'MAE', 'RMSE', 'MAPE (%)', 'R\u00b2', 'Max Error', 'Min Error'],
        ['X-Direction',
         f'{x_metrics["mae"]:.4f}', f'{x_metrics["rmse"]:.4f}',
         f'{x_metrics["mape"]:.2f}%', f'{x_metrics["r2"]:.4f}',
         f'{x_metrics["abs_error"].max():.4f}', f'{x_metrics["abs_error"].min():.4f}'],
        ['Y-Direction',
         f'{y_metrics["mae"]:.4f}', f'{y_metrics["rmse"]:.4f}',
         f'{y_metrics["mape"]:.2f}%', f'{y_metrics["r2"]:.4f}',
         f'{y_metrics["abs_error"].max():.4f}', f'{y_metrics["abs_error"].min():.4f}'],
    ]

    col_widths = [0.14, 0.12, 0.12, 0.12, 0.10, 0.12, 0.12]
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header row
    for j in range(7):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold', fontsize=10)
    # Style direction labels
    for i in [1, 2]:
        table[(i, 0)].set_text_props(fontweight='bold')
    # Alternating row colors
    for j in range(7):
        table[(1, j)].set_facecolor('#dce6f1')
        table[(2, j)].set_facecolor('#f5d6d6')

    ax4.set_title(f'Error Summary{title_suffix}', fontsize=12, pad=8)

    # Save
    plot_filename = f"comparison_{building_name}.png" if building_name else "comparison.png"
    plt.savefig(os.path.join(output_folder, plot_filename), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
    return x_metrics, y_metrics


def plot_scatter_global(all_x_true, all_x_pred, all_y_true, all_y_pred, output_folder):
    """Publication-quality scatter: Predicted vs True with R^2 and perfect line."""
    apply_plot_style()
    
    all_x_true = np.array(all_x_true)
    all_x_pred = np.array(all_x_pred)
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    x_metrics = calculate_error_metrics(all_x_true, all_x_pred)
    y_metrics = calculate_error_metrics(all_y_true, all_y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ---- X-Direction ----
    ax1 = axes[0]
    ax1.scatter(all_x_true, all_x_pred, alpha=0.5, c='steelblue', edgecolors='darkblue', s=40)
    vmin = min(all_x_true.min(), all_x_pred.min()) * 0.95
    vmax = max(all_x_true.max(), all_x_pred.max()) * 1.05
    ax1.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1.5, label='Perfect prediction')
    ax1.set_xlabel('True Value')
    ax1.set_ylabel('Predicted Value')
    ax1.set_title(f'X-Direction\nR\u00b2={x_metrics["r2"]:.4f}, MAPE={x_metrics["mape"]:.2f}%')
    ax1.legend()
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)
    
    # ---- Y-Direction ----
    ax2 = axes[1]
    ax2.scatter(all_y_true, all_y_pred, alpha=0.5, c='coral', edgecolors='darkred', s=40)
    vmin = min(all_y_true.min(), all_y_pred.min()) * 0.95
    vmax = max(all_y_true.max(), all_y_pred.max()) * 1.05
    ax2.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1.5, label='Perfect prediction')
    ax2.set_xlabel('True Value')
    ax2.set_ylabel('Predicted Value')
    ax2.set_title(f'Y-Direction\nR\u00b2={y_metrics["r2"]:.4f}, MAPE={y_metrics["mape"]:.2f}%')
    ax2.legend()
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'scatter_true_vs_pred.png'), bbox_inches='tight')
    plt.close()
    return x_metrics, y_metrics


def plot_error_distribution(all_x_true, all_x_pred, all_y_true, all_y_pred, output_folder):
    """Error distribution: histogram + box plot + CDF."""
    apply_plot_style()
    
    x_errors = np.array(all_x_pred) - np.array(all_x_true)
    y_errors = np.array(all_y_pred) - np.array(all_y_true)
    x_pct = np.abs(x_errors / np.array(all_x_true)) * 100
    y_pct = np.abs(y_errors / np.array(all_y_true)) * 100
    x_pct = x_pct[np.isfinite(x_pct)]
    y_pct = y_pct[np.isfinite(y_pct)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram of absolute error
    ax1 = axes[0, 0]
    ax1.hist(np.abs(x_errors), bins=30, color='steelblue', alpha=0.6, edgecolor='darkblue', label='X-Dir')
    ax1.hist(np.abs(y_errors), bins=30, color='coral', alpha=0.6, edgecolor='darkred', label='Y-Dir')
    ax1.set_xlabel('Absolute Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Absolute Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of percentage error
    ax2 = axes[0, 1]
    ax2.hist(x_pct, bins=30, color='steelblue', alpha=0.6, edgecolor='darkblue', label='X-Dir')
    ax2.hist(y_pct, bins=30, color='coral', alpha=0.6, edgecolor='darkred', label='Y-Dir')
    ax2.set_xlabel('Percentage Error (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Percentage Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Box plot
    ax3 = axes[1, 0]
    bp = ax3.boxplot([np.abs(x_errors), np.abs(y_errors)], labels=['X-Direction', 'Y-Direction'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax3.set_ylabel('Absolute Error')
    ax3.set_title('Error Box Plot')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # CDF
    ax4 = axes[1, 1]
    x_sorted = np.sort(np.abs(x_errors))
    y_sorted = np.sort(np.abs(y_errors))
    x_cdf = np.arange(1, len(x_sorted)+1) / len(x_sorted) * 100
    y_cdf = np.arange(1, len(y_sorted)+1) / len(y_sorted) * 100
    ax4.plot(x_sorted, x_cdf, 'b-', linewidth=2, label='X-Direction')
    ax4.plot(y_sorted, y_cdf, 'r-', linewidth=2, label='Y-Direction')
    ax4.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='90th percentile')
    ax4.set_xlabel('Absolute Error')
    ax4.set_ylabel('Cumulative Percentage (%)')
    ax4.set_title('Cumulative Distribution of Absolute Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'error_distribution.png'), bbox_inches='tight')
    plt.close()


def plot_floor_wise_mape(all_results_by_building, output_folder):
    """
    Floor-wise MAPE aggregated across ALL buildings.
    Floor labels: F1 (top) ... F10 (bottom), Base excluded from MAPE.
    """
    apply_plot_style()
    
    all_floor_data = []
    for building_results in all_results_by_building:
        for row in building_results:
            if row.get('Floor', '') == 'Base':
                continue
            if 'X-Dir_True' not in row:
                continue
            all_floor_data.append({
                'floor': row['Floor'],
                'x_pct': abs(row['X-Dir_Pred'] - row['X-Dir_True']) / abs(row['X-Dir_True']) * 100 if row['X-Dir_True'] != 0 else 0,
                'y_pct': abs(row['Y-Dir_Pred'] - row['Y-Dir_True']) / abs(row['Y-Dir_True']) * 100 if row['Y-Dir_True'] != 0 else 0,
                'x_abs': abs(row['X-Dir_Pred'] - row['X-Dir_True']),
                'y_abs': abs(row['Y-Dir_Pred'] - row['Y-Dir_True']),
            })
    
    if not all_floor_data:
        return
    
    df = pd.DataFrame(all_floor_data)
    
    # Sort floors properly: F1, F2, ..., F10
    def floor_sort_key(f):
        return int(f.replace('F', ''))
    
    floor_order = sorted(df['floor'].unique(), key=floor_sort_key)
    
    floor_stats = []
    for f in floor_order:
        fdata = df[df['floor'] == f]
        floor_stats.append({
            'floor': f,
            'x_mape_mean': fdata['x_pct'].mean(),
            'x_mape_std': fdata['x_pct'].std(),
            'y_mape_mean': fdata['y_pct'].mean(),
            'y_mape_std': fdata['y_pct'].std(),
            'x_mae_mean': fdata['x_abs'].mean(),
            'y_mae_mean': fdata['y_abs'].mean(),
        })
    
    fs = pd.DataFrame(floor_stats)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bar_w = 0.35
    indices = np.arange(len(fs))
    
    # MAE by floor
    ax1 = axes[0]
    ax1.bar(indices - bar_w/2, fs['x_mae_mean'], bar_w, label='X-Dir', color='steelblue', alpha=0.8)
    ax1.bar(indices + bar_w/2, fs['y_mae_mean'], bar_w, label='Y-Dir', color='coral', alpha=0.8)
    ax1.set_xlabel('Floor')
    ax1.set_ylabel('MAE')
    ax1.set_title('MAE by Floor (All Buildings)')
    ax1.set_xticks(indices)
    ax1.set_xticklabels(fs['floor'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # MAPE by floor
    ax2 = axes[1]
    ax2.bar(indices - bar_w/2, fs['x_mape_mean'], bar_w, 
            yerr=fs['x_mape_std'], label='X-Dir', color='steelblue', alpha=0.8, capsize=3)
    ax2.bar(indices + bar_w/2, fs['y_mape_mean'], bar_w,
            yerr=fs['y_mape_std'], label='Y-Dir', color='coral', alpha=0.8, capsize=3)
    ax2.set_xlabel('Floor')
    ax2.set_ylabel('MAPE (%)')
    ax2.set_title('MAPE by Floor (All Buildings)')
    ax2.set_xticks(indices)
    ax2.set_xticklabels(fs['floor'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'floor_wise_mape_aggregated.png'), bbox_inches='tight')
    plt.close()
    
    # Save floor-wise data to Excel
    fs.to_excel(os.path.join(output_folder, '..', 'floor_wise_metrics.xlsx'), index=False)
    return fs


def plot_plan_type_comparison(plan_type_metrics, output_folder):
    """Bar chart comparing MAPE across plan types."""
    apply_plot_style()
    
    df = pd.DataFrame(plan_type_metrics)
    df = df.sort_values('Plan_Type')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_w = 0.3
    indices = np.arange(len(df))
    
    ax.bar(indices - bar_w, df['X_MAPE'], bar_w, label='X-Direction', color='steelblue', alpha=0.8)
    ax.bar(indices, df['Y_MAPE'], bar_w, label='Y-Direction', color='coral', alpha=0.8)
    ax.bar(indices + bar_w, df['Combined_MAPE'], bar_w, label='Combined', color='mediumpurple', alpha=0.8)
    
    ax.set_xlabel('Plan Type')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('MAPE by Plan Type')
    ax.set_xticks(indices)
    ax.set_xticklabels(df['Plan_Type'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, row in df.iterrows():
        idx = list(df.index).index(i)
        ax.text(idx - bar_w, row['X_MAPE'] + 0.3, f'{row["X_MAPE"]:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.text(idx, row['Y_MAPE'] + 0.3, f'{row["Y_MAPE"]:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.text(idx + bar_w, row['Combined_MAPE'] + 0.3, f'{row["Combined_MAPE"]:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mape_by_plan_type.png'), bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN INFERENCE FUNCTION
# ============================================================

def infer_for_paper(model_path, graph_folder, output_folder, plot_individual=True):
    """
    Inference with all outputs needed for paper publication.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    model, config = load_model_from_checkpoint(model_path, device)
    edge_dim = config.get('edge_features', 8)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(output_folder, f"inference_19Feat_correctdim_frindlydb_newx33_5721files_gen_raw_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    plots_folder = os.path.join(run_folder, 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    print(f"[INFO] Output folder: {run_folder}")
    
    graph_files = sorted([f for f in os.listdir(graph_folder) if f.endswith(".pt")])
    print(f"[INFO] Found {len(graph_files)} graph files")
    
    # Collectors
    all_x_true, all_x_pred = [], []
    all_y_true, all_y_pred = [], []
    all_building_metrics = []
    all_results_by_building = []
    plan_type_data = {}  # {plan_type: {"x_true":[], "x_pred":[], ...}}
    
    for file in tqdm(graph_files, desc="Inference"):
        graph = torch.load(os.path.join(graph_folder, file), map_location=device)
        building_name = file.replace(".pt", "")
        plan_type = parse_plan_type(building_name)
        
        # Feature dimension check
        expected_features = config.get('node_features', 19)
        actual_features = graph.x.size(1)
        if actual_features != expected_features:
            print(f"[WARNING] {file}: Features {actual_features} != expected {expected_features}")
            if actual_features < expected_features:
                padding = torch.zeros(graph.x.size(0), expected_features - actual_features, device=device)
                graph.x = torch.cat([graph.x, padding], dim=1)
            else:
                graph.x = graph.x[:, :expected_features]
        
        subgraphs = extract_subgraphs_with_edges(graph, edge_dim)
        if len(subgraphs) == 0:
            print(f"[WARNING] {file}: No valid subgraphs")
            continue
        
        # Move to device
        building_gpu = []
        for sub in subgraphs:
            sub_gpu = Data(
                x=sub.x.to(device), edge_index=sub.edge_index.to(device),
                edge_attr=sub.edge_attr.to(device), story_id=sub.story_id.to(device)
            )
            building_gpu.append(sub_gpu)
        
        with torch.no_grad():
            predictions = model(building_gpu).cpu().numpy()
        
        # Collect results per floor
        results = []
        has_true_values = False
        
        for idx, sub in enumerate(subgraphs):
            result_row = {
                "Elevation": sub.elevation,
                "X-Dir_Pred": predictions[idx, 0],
                "Y-Dir_Pred": predictions[idx, 1],
            }
            
            if sub.y is not None and len(sub.y) > 0:
                has_true_values = True
                if sub.y.dim() == 2 and sub.y.size(1) >= 2:
                    x_true = sub.y[:, 0].mean().item()
                    y_true = sub.y[:, 1].mean().item()
                else:
                    x_true = sub.y[0].item() if len(sub.y) > 0 else 0
                    y_true = sub.y[1].item() if len(sub.y) > 1 else 0
                
                result_row["X-Dir_True"] = x_true
                result_row["Y-Dir_True"] = y_true
                result_row["X-Dir_AbsError"] = abs(predictions[idx, 0] - x_true)
                result_row["Y-Dir_AbsError"] = abs(predictions[idx, 1] - y_true)
                result_row["X-Dir_PctError"] = abs(predictions[idx, 0] - x_true) / abs(x_true) * 100 if x_true != 0 else np.nan
                result_row["Y-Dir_PctError"] = abs(predictions[idx, 1] - y_true) / abs(y_true) * 100 if y_true != 0 else np.nan
                
                all_x_true.append(x_true)
                all_x_pred.append(predictions[idx, 0])
                all_y_true.append(y_true)
                all_y_pred.append(predictions[idx, 1])
                
                # Collect per plan type
                if plan_type not in plan_type_data:
                    plan_type_data[plan_type] = {"x_true": [], "x_pred": [], "y_true": [], "y_pred": []}
                plan_type_data[plan_type]["x_true"].append(x_true)
                plan_type_data[plan_type]["x_pred"].append(predictions[idx, 0])
                plan_type_data[plan_type]["y_true"].append(y_true)
                plan_type_data[plan_type]["y_pred"].append(predictions[idx, 1])
            
            results.append(result_row)
        
        # === ADD BASE FLOOR ===
        results_with_base = add_base_floor(results)
        all_results_by_building.append(results_with_base)
        
        # Save Excel (with base floor)
        df = pd.DataFrame(results_with_base)
        
        if has_true_values:
            df_no_base = df[df['Floor'] != 'Base']
            summary_row = {
                "Floor": "SUMMARY",
                "Elevation": "",
                "X-Dir_Pred": "", "Y-Dir_Pred": "",
                "X-Dir_True": "", "Y-Dir_True": "",
                "X-Dir_AbsError": df_no_base["X-Dir_AbsError"].mean(),
                "Y-Dir_AbsError": df_no_base["Y-Dir_AbsError"].mean(),
                "X-Dir_PctError": df_no_base["X-Dir_PctError"].mean(),
                "Y-Dir_PctError": df_no_base["Y-Dir_PctError"].mean(),
            }
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
            
            all_building_metrics.append({
                "Building": building_name,
                "Plan_Type": plan_type,
                "X_MAE": df_no_base["X-Dir_AbsError"].mean(),
                "Y_MAE": df_no_base["Y-Dir_AbsError"].mean(),
                "X_MAPE": df_no_base["X-Dir_PctError"].mean(),
                "Y_MAPE": df_no_base["Y-Dir_PctError"].mean(),
            })
        
        output_file = os.path.join(run_folder, file.replace(".pt", ".xlsx"))
        df.to_excel(output_file, index=False)
        
        if plot_individual and has_true_values:
            plot_building_comparison(results_with_base, plots_folder, building_name)
    
    # ============================================================
    # GLOBAL SUMMARY
    # ============================================================
    if len(all_x_true) > 0:
        print("\n" + "=" * 70)
        print("GLOBAL SUMMARY METRICS (for Paper)")
        print("=" * 70)
        
        x_global = calculate_error_metrics(np.array(all_x_true), np.array(all_x_pred))
        y_global = calculate_error_metrics(np.array(all_y_true), np.array(all_y_pred))
        
        all_true = np.concatenate([all_x_true, all_y_true])
        all_pred = np.concatenate([all_x_pred, all_y_pred])
        combined_global = calculate_error_metrics(all_true, all_pred)
        
        print(f"\n{'Metric':<12} {'X-Direction':>14} {'Y-Direction':>14} {'Combined':>14}")
        print("-" * 56)
        print(f"{'MAE':<12} {x_global['mae']:>14.4f} {y_global['mae']:>14.4f} {combined_global['mae']:>14.4f}")
        print(f"{'RMSE':<12} {x_global['rmse']:>14.4f} {y_global['rmse']:>14.4f} {combined_global['rmse']:>14.4f}")
        print(f"{'MAPE (%)':<12} {x_global['mape']:>13.2f}% {y_global['mape']:>13.2f}% {combined_global['mape']:>13.2f}%")
        print(f"{'R2':<12} {x_global['r2']:>14.4f} {y_global['r2']:>14.4f} {combined_global['r2']:>14.4f}")
        
        # Save global summary Excel
        summary_df = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "MAPE (%)", "R\u00b2", "Num Samples"],
            "X-Direction": [x_global['mae'], x_global['rmse'], x_global['mape'], x_global['r2'], len(all_x_true)],
            "Y-Direction": [y_global['mae'], y_global['rmse'], y_global['mape'], y_global['r2'], len(all_y_true)],
            "Combined": [combined_global['mae'], combined_global['rmse'], combined_global['mape'], combined_global['r2'], len(all_true)],
        })
        summary_df.to_excel(os.path.join(run_folder, "global_summary.xlsx"), index=False)
        
        # ============================================================
        # PER-PLAN-TYPE METRICS
        # ============================================================
        print("\n" + "=" * 70)
        print("PER-PLAN-TYPE METRICS")
        print("=" * 70)
        
        plan_type_metrics = []
        print(f"\n{'Plan Type':<12} {'N_floors':>10} {'X_MAPE':>10} {'Y_MAPE':>10} {'Comb_MAPE':>12} {'X_R2':>8} {'Y_R2':>8}")
        print("-" * 72)
        
        for pt in sorted(plan_type_data.keys()):
            ptd = plan_type_data[pt]
            x_m = calculate_error_metrics(np.array(ptd['x_true']), np.array(ptd['x_pred']))
            y_m = calculate_error_metrics(np.array(ptd['y_true']), np.array(ptd['y_pred']))
            
            all_t = np.concatenate([ptd['x_true'], ptd['y_true']])
            all_p = np.concatenate([ptd['x_pred'], ptd['y_pred']])
            c_m = calculate_error_metrics(all_t, all_p)
            
            plan_type_metrics.append({
                'Plan_Type': pt,
                'N_floors': len(ptd['x_true']),
                'N_buildings': len(ptd['x_true']) // 10 if len(ptd['x_true']) >= 10 else 1,
                'X_MAE': x_m['mae'], 'X_RMSE': x_m['rmse'], 'X_MAPE': x_m['mape'], 'X_R2': x_m['r2'],
                'Y_MAE': y_m['mae'], 'Y_RMSE': y_m['rmse'], 'Y_MAPE': y_m['mape'], 'Y_R2': y_m['r2'],
                'Combined_MAE': c_m['mae'], 'Combined_RMSE': c_m['rmse'], 
                'Combined_MAPE': c_m['mape'], 'Combined_R2': c_m['r2'],
            })
            
            print(f"{pt:<12} {len(ptd['x_true']):>10} {x_m['mape']:>9.2f}% {y_m['mape']:>9.2f}% {c_m['mape']:>11.2f}% {x_m['r2']:>8.4f} {y_m['r2']:>8.4f}")
        
        # Save per-plan-type
        pt_df = pd.DataFrame(plan_type_metrics)
        pt_df.to_excel(os.path.join(run_folder, "per_plan_type_metrics.xlsx"), index=False)
        
        # Save per-building metrics
        if all_building_metrics:
            building_df = pd.DataFrame(all_building_metrics)
            building_df.to_excel(os.path.join(run_folder, "per_building_metrics.xlsx"), index=False)
        
        # ============================================================
        # GENERATE ALL PLOTS
        # ============================================================
        print("\n[INFO] Generating publication plots...")
        
        # 1. Scatter plot (True vs Predicted)
        plot_scatter_global(all_x_true, all_x_pred, all_y_true, all_y_pred, plots_folder)
        print("  [OK] scatter_true_vs_pred.png")
        
        # 2. Error distribution
        plot_error_distribution(all_x_true, all_x_pred, all_y_true, all_y_pred, plots_folder)
        print("  [OK] error_distribution.png")
        
        # 3. Floor-wise MAPE (aggregated)
        plot_floor_wise_mape(all_results_by_building, plots_folder)
        print("  [OK] floor_wise_mape_aggregated.png")
        
        # 4. Plan type comparison
        if len(plan_type_metrics) > 1:
            plot_plan_type_comparison(plan_type_metrics, plots_folder)
            print("  [OK] mape_by_plan_type.png")
        
        print("\n" + "=" * 70)
    else:
        print("\n[WARNING] No true values found in graphs. Only predictions saved.")
    
    print(f"\n[DONE] Results saved to {run_folder}")
    print(f"[DONE] Plots saved to {plots_folder}")
    print(f"\n[PAPER OUTPUTS]")
    print(f"  - global_summary.xlsx         -> Table: Overall metrics (MAE, RMSE, MAPE, R2)")
    print(f"  - per_plan_type_metrics.xlsx   -> Table: Metrics by plan type")
    print(f"  - per_building_metrics.xlsx    -> Table: Metrics per building")
    print(f"  - floor_wise_metrics.xlsx      -> Table: Metrics by floor")
    print(f"  - plots/scatter_true_vs_pred.png     -> Fig: Predicted vs True")
    print(f"  - plots/error_distribution.png       -> Fig: Error histograms")
    print(f"  - plots/floor_wise_mape_aggregated.png -> Fig: MAPE by floor")
    print(f"  - plots/mape_by_plan_type.png        -> Fig: MAPE by plan type")
    print(f"  - plots/comparison_*.png             -> Fig: Individual buildings")


if __name__ == "__main__":
    # UPDATE THESE PATHS
    model_path = "02. train results/02.18 features/gatv2_updated_fulldata_model_frindlydb_newx33_5721files_18feat_20260330_170833/best_model.pt"
    graph_folder = "03. test folder/all_after_2steps_diffusion_generated_plan/"
    output_folder = "04. inference results/02.18feat"
    
    # Set plot_individual=False to skip individual building plots (faster)
    infer_for_paper(model_path, graph_folder, output_folder, plot_individual=True)