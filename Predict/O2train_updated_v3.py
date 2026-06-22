import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import time
import gc

def compute_metrics(y_true, y_pred):
    """Compute regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}


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
        
        # Residual projection if dimensions don't match
        if in_channels != out_channels * heads:
            self.residual_proj = nn.Linear(in_channels, out_channels * heads)
        else:
            self.residual_proj = None
        
    def forward(self, x, edge_index, edge_attr):
        # Store input for residual
        residual = x
        
        # GATv2 forward
        x = self.gatv2(x, edge_index, edge_attr)
        x = self.norm(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Add residual connection
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
        
        # Stack of GATv2 layers with residual connections
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            ResidualGATv2Layer(hidden_channels, hidden_channels // heads, 
                              edge_dim, heads, dropout)
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                ResidualGATv2Layer(hidden_channels, hidden_channels // heads,
                                  edge_dim, heads, dropout)
            )
        
        # Last layer - single head with residual
        self.final_gat = GATv2Conv(hidden_channels, hidden_channels, heads=1, 
                                   edge_dim=edge_dim, dropout=dropout, concat=False)
        self.final_norm = nn.LayerNorm(hidden_channels)
        
        # Pooling projection
        self.pool_projection = nn.Linear(hidden_channels * 3, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr, batch):
        """Process walls in a floor using GATv2 with residual connections"""
        # Slice only the features the model was designed for (supervisor approach:
        # data may have more features than model uses — model reads what it needs)
        x = x[:, :self.input_projection.in_features]
        # Initial projection
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply GATv2 layers with residual
        for layer in self.gat_layers:
            x = layer(x, edge_index, edge_attr)
        
        # Final layer with residual
        residual = x
        x = self.final_gat(x, edge_index, edge_attr)
        x = self.final_norm(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = x + residual
        
        # Multi-pooling aggregation
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        
        # Combine pooling strategies
        x = torch.cat([x_mean, x_max, x_sum], dim=-1)
        x = self.pool_projection(x)
        
        return x


class BuildingGATv2Model(nn.Module):
    """Hierarchical model: GATv2 -> GRU -> Transformer with residual connections"""
    def __init__(self, node_features=19, edge_features=8, hidden_channels=128,
                 gat_layers=3, gat_heads=4, gru_layers=2, transformer_layers=2,
                 transformer_heads=8, dropout=0.1):
        super().__init__()
        
        # GATv2 encoder for floor-level processing
        self.floor_encoder = FloorGATEncoder(
            node_features, hidden_channels, edge_features, 
            gat_layers, gat_heads, dropout
        )
        
        # Positional encoding for floors
        self.floor_position_encoding = nn.Embedding(20, hidden_channels)
        
        # GRU for sequential floor processing
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=True
        )
        
        # Projection after bidirectional GRU
        self.gru_projection = nn.Linear(hidden_channels * 2, hidden_channels)
        self.gru_norm = nn.LayerNorm(hidden_channels)
        
        # Transformer for global building context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=transformer_heads,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Output prediction heads
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
        """Process a building with multiple floors"""
        floor_representations = []
        
        # Process each floor through GATv2
        for floor_idx, floor_data in enumerate(building_data):
            batch = torch.zeros(floor_data.x.size(0), dtype=torch.long, device=floor_data.x.device)
            
            floor_repr = self.floor_encoder(
                floor_data.x, 
                floor_data.edge_index, 
                floor_data.edge_attr,
                batch
            )
            
            # Add positional encoding
            pos_encoding = self.floor_position_encoding(
                torch.tensor([floor_idx], device=floor_data.x.device)
            )
            floor_repr = floor_repr + pos_encoding
            
            floor_representations.append(floor_repr.squeeze(0))
        
        # Stack floor representations
        x = torch.stack(floor_representations, dim=0)
        x = x.unsqueeze(0)
        
        # GRU with residual
        gru_input = x
        x, _ = self.gru(x)
        x = self.gru_projection(x)
        x = self.gru_norm(x)
        x = x + gru_input  # Residual connection
        
        # Transformer with residual
        transformer_input = x
        x = self.transformer(x)
        x = x + transformer_input  # Residual connection
        
        # Generate predictions
        predictions = self.output_mlp(x)
        
        return predictions.squeeze(0)


class BuildingDataset(Dataset):
    """Dataset that loads one building at a time"""
    def __init__(self, file_paths):
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Load and process a single building"""
        file_path = self.file_paths[idx]
        
        try:
            data = torch.load(file_path, map_location='cpu')
            
            if not all(hasattr(data, attr) for attr in ['x', 'edge_index', 'y', 'story_id']):
                print(f"Warning: {file_path} missing required attributes")
                return None
            
            if not hasattr(data, 'edge_attr') or data.edge_attr is None:
                num_edges = data.edge_index.size(1)
                data.edge_attr = torch.ones(num_edges, 8)
            
            # Group by floors
            story_ids = torch.unique(data.story_id, sorted=True)
            building_floors = []
            
            for story_id in story_ids:
                floor_mask = data.story_id == story_id
                floor_indices = torch.where(floor_mask)[0]
                
                if len(floor_indices) == 0:
                    continue
                
                floor_x = data.x[floor_mask]
                floor_y = data.y[floor_mask]
                
                # Extract edges within this floor
                node_map = torch.full((data.x.size(0),), -1, dtype=torch.long)
                for new_idx, old_idx in enumerate(floor_indices):
                    node_map[old_idx] = new_idx
                
                src_in_floor = node_map[data.edge_index[0]] >= 0
                dst_in_floor = node_map[data.edge_index[1]] >= 0
                edge_mask = src_in_floor & dst_in_floor
                
                if edge_mask.any():
                    floor_edges = data.edge_index[:, edge_mask]
                    floor_edge_attr = data.edge_attr[edge_mask]
                    
                    floor_edges_remapped = torch.stack([
                        node_map[floor_edges[0]],
                        node_map[floor_edges[1]]
                    ])
                else:
                    floor_edges_remapped = torch.zeros((2, 0), dtype=torch.long)
                    floor_edge_attr = torch.zeros((0, 8), dtype=torch.float)
                
                floor_data = Data(
                    x=floor_x,
                    edge_index=floor_edges_remapped,
                    edge_attr=floor_edge_attr,
                    y=floor_y,
                    story_id=story_id
                )
                
                building_floors.append(floor_data)
            
            return building_floors if len(building_floors) > 0 else None
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None


def collate_fn(batch):
    """Custom collate function"""
    valid_buildings = [b for b in batch if b is not None]
    return valid_buildings


def train():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    folder = "01. supergraphs/supergrap_19feat_correct_layout_3times"
    
    # Hyperparameters - UPDATED 
    config = {
        'node_features': 18,
        'edge_features': 8,
        'hidden_channels': 128,
        'gat_layers': 4,
        'gat_heads': 4,
        'gru_layers': 3,
        'transformer_layers': 3,
        'transformer_heads': 8,
        'dropout': 0.1,
        'learning_rate': 0.0003,      # Reduced from 0.001
        'weight_decay': 1e-4,
        'batch_size': 32,              # Increased from 1
        'accumulation_steps': 2,      # Gradient accumulation
        'epochs': 300,
        'patience': 50,
        'warmup_epochs': 10,          # Warmup for scheduler
    }
    
    # Setup - set all seeds
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load file paths
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pt")]
    
    if len(all_files) == 0:
        print(f"No .pt files found in {folder}")
        return
    
    print(f"Found {len(all_files)} files")
    
    # SHUFFLE before split
    random.shuffle(all_files)
    
    # Split data
    split = int(0.8 * len(all_files))
    train_files = all_files[:split]
    val_files = all_files[split:]
    
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    
    # Create datasets
    train_dataset = BuildingDataset(train_files)
    val_dataset = BuildingDataset(val_files)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize model
    model_config = {k: v for k, v in config.items() 
                   if k in ['node_features', 'edge_features', 'hidden_channels', 
                           'gat_layers', 'gat_heads', 'gru_layers', 
                           'transformer_layers', 'transformer_heads', 'dropout']}
    
    model = BuildingGATv2Model(**model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # ReduceLROnPlateau — adaptive learning rate
    # Reduces LR by factor when val_loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # Monitor val_loss (minimize)
        factor=0.5,        # Halve LR on plateau
        patience=10,       # Wait 10 epochs before reducing
        min_lr=1e-6,       # Floor LR
        verbose=True       # Print when LR changes
    )
    
    loss_fn = nn.MSELoss()
    
    # Training setup
    results = []
    best_val_loss = float('inf')
    save_path = f"02. train results/03.19feat_correct_layout/gatv2_19feat_correct_layout_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_path, exist_ok=True)
    
    # Save config
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    patience_counter = 0
    accumulation_steps = config['accumulation_steps']
    
    # Training loop
    val_loss = float('inf')  # Initialize before first epoch
    for epoch in range(1, config['epochs'] + 1):
        # Training phase
        model.train()
        train_losses = []
        train_preds_all = []
        train_targets_all = []
        
        optimizer.zero_grad()
        accumulated_loss = 0
        step_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 0:
                continue
                
            # Process each building in the batch
            for building in batch:
                if building is None or len(building) == 0:
                    continue
                
                # Move to device
                building_gpu = []
                targets = []
                
                for floor_data in building:
                    floor_data_gpu = Data(
                        x=floor_data.x.to(device),
                        edge_index=floor_data.edge_index.to(device),
                        edge_attr=floor_data.edge_attr.to(device),
                        y=floor_data.y.to(device)
                    )
                    building_gpu.append(floor_data_gpu)
                    targets.append(floor_data.y.mean(dim=0))
                
                if len(building_gpu) == 0:
                    continue
                
                targets = torch.stack(targets).to(device)
                
                # Forward pass
                try:
                    predictions = model(building_gpu)
                    loss = loss_fn(predictions, targets)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    accumulated_loss += loss.item()
                    step_count += 1
                    
                    # Store results (use unscaled loss for logging)
                    train_losses.append(loss.item() * accumulation_steps)
                    train_preds_all.extend(predictions.detach().cpu().numpy())
                    train_targets_all.extend(targets.cpu().numpy())
                    
                except Exception as e:
                    print(f"\nError in forward pass: {e}")
                    continue
            
            # Gradient accumulation step
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                pbar.set_postfix({
                    'loss': f"{accumulated_loss:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
                accumulated_loss = 0
        
        
        # Compute training metrics
        if len(train_losses) > 0:
            train_loss = np.mean(train_losses)
            train_metrics = compute_metrics(train_targets_all, train_preds_all)
        else:
            print("No valid training samples in this epoch")
            continue
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds_all = []
        val_targets_all = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} - Validation"):
                if len(batch) == 0:
                    continue
                    
                for building in batch:
                    if building is None or len(building) == 0:
                        continue
                    
                    building_gpu = []
                    targets = []
                    
                    for floor_data in building:
                        floor_data_gpu = Data(
                            x=floor_data.x.to(device),
                            edge_index=floor_data.edge_index.to(device),
                            edge_attr=floor_data.edge_attr.to(device),
                            y=floor_data.y.to(device)
                        )
                        building_gpu.append(floor_data_gpu)
                        targets.append(floor_data.y.mean(dim=0))
                    
                    if len(building_gpu) == 0:
                        continue
                    
                    targets = torch.stack(targets).to(device)
                    
                    try:
                        predictions = model(building_gpu)
                        loss = loss_fn(predictions, targets)
                        
                        val_losses.append(loss.item())
                        val_preds_all.extend(predictions.cpu().numpy())
                        val_targets_all.extend(targets.cpu().numpy())
                        
                    except Exception as e:
                        print(f"\nError in validation: {e}")
                        continue
        
        # Compute validation metrics
        if len(val_losses) > 0:
            val_loss = np.mean(val_losses)
            val_metrics = compute_metrics(val_targets_all, val_preds_all)
        else:
            print("No valid validation samples in this epoch")
            continue
        
        # Print progress
        print(f"\n[Epoch {epoch}]")
        print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_metrics['MAE']:.4f}, R2: {train_metrics['R2']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_metrics['MAE']:.4f}, R2: {val_metrics['R2']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Store results
        results.append({
            "Epoch": epoch,
            "Train_Loss": train_loss,
            "Val_Loss": val_loss,
            "Train_MAE": train_metrics["MAE"],
            "Train_RMSE": train_metrics["RMSE"],
            "Train_R2": train_metrics["R2"],
            "Val_MAE": val_metrics["MAE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
            "Learning_Rate": optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            scheduler.step(val_loss)
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(save_path, "best_model.pt"))
            print(f"  [SAVED] New best model! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"[STOP] Early stopping at epoch {epoch}")
                break
        
        # Periodic garbage collection
        if epoch % 10 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Save results
    pd.DataFrame(results).to_excel(os.path.join(save_path, "training_log.xlsx"), index=False)
    print(f"\n[DONE] Training complete. Results saved to {save_path}")
    print(f"[BEST] Validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    print("="*60)
    print("GATv2 + GRU + Transformer Training (Updated)")
    print("="*60)
    print("Updates:")
    print("  - Residual connections in GATv2 layers")
    print("  - Shuffled data split")
    print("  - Learning rate: 0.0003 (reduced)")
    print("  - Batch size: 4 with gradient accumulation")
    print("  - CosineAnnealingWarmRestarts scheduler")
    print("="*60)
    train()