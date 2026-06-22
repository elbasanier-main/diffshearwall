import os
import json
import torch
import numpy as np
from torch_geometric.data import Data
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

INPUT_DIR = "generated_raw/output_generated_model_frindlydb_newx33O"
OUTPUT_DIR = "03. test folder/supergrap_19feat_correct_layout_gen_raw"
DEBUG_LOG = "edge_debug_log.txt"
pio.renderers.default = 'json'  # Server-safe

def parse_joint(joint_str):
    """Parse joint coordinate string"""
    return eval(joint_str) if isinstance(joint_str, str) else joint_str

def compute_wall_features(wall, elevation, shear_ratio, layout_x, layout_y):
    """Extract comprehensive features for a wall node"""
    # Parse wall corners
    joints = [parse_joint(wall[f"Elm Jt{i}"]) for i in range(1, 5)]
    j1, j2, j3, j4 = joints
    
    # Wall centroid
    centroid = np.mean(joints, axis=0)
    
    # Wall dimensions
    # Assuming rectangular walls: j1-j2 is bottom edge, j2-j3 is side edge
    width = np.linalg.norm(np.array(j2) - np.array(j1))
    height = np.linalg.norm(np.array(j3) - np.array(j2))
    
    # Wall orientation
    edge_vector = np.array(j2) - np.array(j1)
    angle = np.arctan2(edge_vector[1], edge_vector[0])
    
    # Determine if wall is aligned with X or Y axis
    is_x_aligned = abs(np.cos(angle)) > 0.9  # Close to 0 or 180 degrees
    is_y_aligned = abs(np.sin(angle)) > 0.9  # Close to 90 or 270 degrees
    
    # Wall area and aspect ratio
    area = width * height
    aspect_ratio = width / height if height > 0 else 1.0
    
    # Distance from building center (assuming center at origin)
    dist_from_center = np.linalg.norm(centroid[:2])
    
    # Wall position normalized by layout
    norm_x = centroid[0] / layout_x if layout_x > 0 else 0
    norm_y = centroid[1] / layout_y if layout_y > 0 else 0
    
    # Height ratio (current elevation / max expected elevation)
    height_ratio = elevation / 40.0  # Assuming max 40 units based on your data
    
    features = [
        elevation,              # 0: Floor elevation
        centroid[0],           # 1: Centroid X
        centroid[1],           # 2: Centroid Y
        centroid[2],           # 3: Centroid Z
        width,                 # 4: Wall width
        height,                # 5: Wall height
        area,                  # 6: Wall area
        aspect_ratio,          # 7: Aspect ratio
        angle,                 # 8: Wall orientation angle
        float(is_x_aligned),   # 9: X-alignment flag
        float(is_y_aligned),   # 10: Y-alignment flag
        dist_from_center,      # 11: Distance from center
        norm_x,                # 12: Normalized X position
        norm_y,                # 13: Normalized Y position
        height_ratio,          # 14: Height ratio
        shear_ratio,           # 15: Building shear wall ratio
        layout_x,              # 16: Building layout X
        layout_y,              # 17: Building layout Y
        float(wall.get("Element Name", 0))  # 18: Element ID
    ]
    
    return features, centroid, joints

def compute_edge_features(coord_a, coord_b, joints_a, joints_b):
    """Compute features for an edge between two walls"""
    # Distance between centroids
    dist = np.linalg.norm(coord_a - coord_b)
    
    # Relative position
    dx = coord_b[0] - coord_a[0]
    dy = coord_b[1] - coord_a[1]
    dz = coord_b[2] - coord_a[2]
    
    # Connection angle
    angle = np.arctan2(dy, dx)
    
    # Check if walls are aligned (parallel or perpendicular)
    edge_a = np.array(joints_a[1]) - np.array(joints_a[0])
    edge_b = np.array(joints_b[1]) - np.array(joints_b[0])
    
    # Normalize edge vectors
    edge_a_norm = edge_a / (np.linalg.norm(edge_a) + 1e-6)
    edge_b_norm = edge_b / (np.linalg.norm(edge_b) + 1e-6)
    
    # Dot product for alignment check
    dot_product = np.abs(np.dot(edge_a_norm[:2], edge_b_norm[:2]))
    is_parallel = dot_product > 0.9
    is_perpendicular = dot_product < 0.1
    
    # Vertical connection flag
    is_vertical = abs(dz) > 0.1
    
    edge_features = [
        dist,                    # 0: Distance between walls
        dx,                      # 1: X displacement
        dy,                      # 2: Y displacement
        dz,                      # 3: Z displacement (for vertical connections)
        angle,                   # 4: Connection angle
        float(is_parallel),      # 5: Parallel alignment
        float(is_perpendicular), # 6: Perpendicular alignment
        float(is_vertical),      # 7: Vertical connection flag
    ]
    
    return edge_features

def extract_layout_info_from_coords(file_data):
    """Compute layout_x and layout_y from actual joint coordinates.
    Avoids project name convention issues (e.g. 4x6 encodes Y_bays x X_bays).
    layout_x = max X joint coordinate on first floor
    layout_y = max Y joint coordinate on first floor
    """
    import ast
    xs, ys = [], []
    story = file_data["Story_detail"][0]
    for wall in story.get("Wall_Details", []):
        for jt in ["Elm Jt1", "Elm Jt2", "Elm Jt3", "Elm Jt4"]:
            coord = wall[jt]
            if isinstance(coord, str):
                coord = ast.literal_eval(coord)
            xs.append(float(coord[0]))
            ys.append(float(coord[1]))
    layout_x = max(xs) if xs else 18.0
    layout_y = max(ys) if ys else 18.0
    return layout_x, layout_y


def build_clean_grid_edges(floor_nodes, all_centroids, all_joints, grid_tol=3.0):
    """
    Build clean grid edges within a floor.
    
    Rules:
    1. Connect consecutive walls within each row (sorted by X)
    2. Connect consecutive walls within each column (sorted by Y)
    3. Connect each node to closest node in the NEXT row (up)
    4. Connect each node to closest node in the NEXT column (right)
    
    Result: ~3-4 edges per node (scalable to 700+ nodes)
    """
    edges = []
    edge_features = []
    edge_set = set()
    
    if len(floor_nodes) < 2:
        return edges, edge_features
    
    # Get coordinates for floor nodes
    coords = {node: (all_centroids[node][0], all_centroids[node][1]) for node in floor_nodes}
    
    def add_edge(node_a, node_b):
        """Add bidirectional edge if not exists"""
        edge_key = (min(node_a, node_b), max(node_a, node_b))
        if edge_key not in edge_set:
            edge_set.add(edge_key)
            edges.append([node_a, node_b])
            edges.append([node_b, node_a])
            
            edge_feat = compute_edge_features(
                all_centroids[node_a], all_centroids[node_b],
                all_joints[node_a], all_joints[node_b]
            )
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)
    
    # Cluster Y values into rows
    y_vals = sorted(set(coords[n][1] for n in floor_nodes))
    row_centers = []
    used = set()
    for y in y_vals:
        if y in used:
            continue
        cluster = [yv for yv in y_vals if abs(yv - y) < grid_tol]
        for yv in cluster:
            used.add(yv)
        row_centers.append(np.mean(cluster))
    row_centers = sorted(row_centers)
    
    # Cluster X values into columns
    x_vals = sorted(set(coords[n][0] for n in floor_nodes))
    col_centers = []
    used = set()
    for x in x_vals:
        if x in used:
            continue
        cluster = [xv for xv in x_vals if abs(xv - x) < grid_tol]
        for xv in cluster:
            used.add(xv)
        col_centers.append(np.mean(cluster))
    col_centers = sorted(col_centers)
    
    # Assign nodes to rows and columns
    rows = defaultdict(list)  # row_idx -> [(node, x)]
    cols = defaultdict(list)  # col_idx -> [(node, y)]
    
    for node in floor_nodes:
        x, y = coords[node]
        for ri, rc in enumerate(row_centers):
            if abs(y - rc) < grid_tol:
                rows[ri].append((node, x))
                break
        for ci, cc in enumerate(col_centers):
            if abs(x - cc) < grid_tol:
                cols[ci].append((node, y))
                break
    
    # 1. Connect consecutive in rows (sorted by X)
    for ri, node_list in rows.items():
        sorted_nodes = sorted(node_list, key=lambda x: x[1])
        for k in range(len(sorted_nodes) - 1):
            add_edge(sorted_nodes[k][0], sorted_nodes[k+1][0])
    
    # 2. Connect consecutive in columns (sorted by Y)
    for ci, node_list in cols.items():
        sorted_nodes = sorted(node_list, key=lambda x: x[1])
        for k in range(len(sorted_nodes) - 1):
            add_edge(sorted_nodes[k][0], sorted_nodes[k+1][0])
    
    # 3. Connect each node to closest in NEXT row (up)
    for ri in range(len(row_centers) - 1):
        current_nodes = [node for node, _ in rows[ri]]
        next_nodes = [node for node, _ in rows[ri + 1]]
        
        for node in current_nodes:
            if next_nodes:
                x = coords[node][0]
                closest = min(next_nodes, key=lambda n: abs(coords[n][0] - x))
                add_edge(node, closest)
    
    # 4. Connect each node to closest in NEXT column (right)
    for ci in range(len(col_centers) - 1):
        current_nodes = [node for node, _ in cols[ci]]
        next_nodes = [node for node, _ in cols[ci + 1]]
        
        for node in current_nodes:
            if next_nodes:
                y = coords[node][1]
                closest = min(next_nodes, key=lambda n: abs(coords[n][1] - y))
                add_edge(node, closest)
    
    return edges, edge_features


def build_enhanced_graph(json_path, grid_tol=3.0, vertical_thresh=4.0):
    """Build graph with clean grid edges"""
    with open(json_path) as f:
        data = json.load(f)
    
    file_data = data["File_details"][0]
    shear_ratio = file_data.get("Shear_wall_ratio", 0.0)
    layout_x, layout_y = extract_layout_info_from_coords(file_data)
    
    # Storage for graph components
    node_features = []
    node_targets = []
    story_ids = []
    all_centroids = []
    all_joints = []
    node_to_floor = {}
    floor_node_groups = []
    
    node_id = 0
    
    # Process each floor
    for floor_idx, story in enumerate(file_data["Story_detail"]):
        elevation = story["Elevation"]
        x_dir = story.get("X-Dir", 0.0)
        y_dir = story.get("Y-Dir", 0.0)
        
        floor_nodes = []
        
        # Process walls in this floor
        for wall in story.get("Wall_Details", []):
            try:
                features, centroid, joints = compute_wall_features(
                    wall, elevation, shear_ratio, layout_x, layout_y
                )
                
                node_features.append(features)
                node_targets.append([x_dir, y_dir])
                story_ids.append(elevation)
                all_centroids.append(centroid)
                all_joints.append(joints)
                
                node_to_floor[node_id] = floor_idx
                floor_nodes.append(node_id)
                node_id += 1
                
            except Exception as e:
                print(f"Error processing wall: {e}")
                continue
        
        floor_node_groups.append(floor_nodes)
    
    # Build edges
    edges = []
    edge_features = []
    
    # 1. Within-floor connections using clean grid logic
    for floor_nodes in floor_node_groups:
        floor_edges, floor_edge_feats = build_clean_grid_edges(
            floor_nodes, all_centroids, all_joints, grid_tol
        )
        edges.extend(floor_edges)
        edge_features.extend(floor_edge_feats)
    
    # 2. Between-floor connections (vertical)
    for i in range(len(floor_node_groups) - 1):
        upper_floor = floor_node_groups[i]
        lower_floor = floor_node_groups[i + 1]
        
        for node_a in upper_floor:
            coord_a = all_centroids[node_a]
            joints_a = all_joints[node_a]
            
            # Find closest node in floor below
            min_dist = float('inf')
            best_node_b = None
            
            for node_b in lower_floor:
                coord_b = all_centroids[node_b]
                # Only consider XY distance for vertical connections
                xy_dist = np.linalg.norm([
                    coord_a[0] - coord_b[0],
                    coord_a[1] - coord_b[1]
                ])
                
                if xy_dist < min_dist:
                    min_dist = xy_dist
                    best_node_b = node_b
            
            # Connect to closest node if within threshold
            if best_node_b is not None and min_dist < vertical_thresh:
                coord_b = all_centroids[best_node_b]
                joints_b = all_joints[best_node_b]
                
                edges.append([node_a, best_node_b])
                edges.append([best_node_b, node_a])
                
                edge_feat = compute_edge_features(
                    coord_a, coord_b, joints_a, joints_b
                )
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
    
    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else None
    y = torch.tensor(node_targets, dtype=torch.float)
    story_id = torch.tensor(story_ids, dtype=torch.float)
    
    # Create Data object
    graph_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        story_id=story_id
    )
    
    return graph_data, all_centroids, edges

def visualize_enhanced_graph(centroids, edges, story_ids, save_path=None):
    """Create enhanced visualization of the graph"""
    x = [c[0] for c in centroids]
    y = [c[1] for c in centroids]
    z = [c[2] for c in centroids]
    
    # Separate edges by type
    horizontal_edges = []
    vertical_edges = []
    
    for i in range(0, len(edges), 2):  # Skip reverse edges
        a, b = edges[i]
        if abs(centroids[a][2] - centroids[b][2]) < 0.1:
            horizontal_edges.append((a, b))
        else:
            vertical_edges.append((a, b))
    
    # Create edge traces
    def create_edge_trace(edge_list, color, name):
        edge_x, edge_y, edge_z = [], [], []
        for a, b in edge_list:
            edge_x.extend([centroids[a][0], centroids[b][0], None])
            edge_y.extend([centroids[a][1], centroids[b][1], None])
            edge_z.extend([centroids[a][2], centroids[b][2], None])
        
        return go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color=color, width=2),
            name=name
        )
    
    fig = go.Figure()
    
    # Add edges
    if horizontal_edges:
        fig.add_trace(create_edge_trace(horizontal_edges, 'blue', 'Horizontal Connections'))
    if vertical_edges:
        fig.add_trace(create_edge_trace(vertical_edges, 'red', 'Vertical Connections'))
    
    # Add nodes
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        marker=dict(
            size=8,
            color=z,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Elevation")
        ),
        text=[f"Floor {int(sid)}" for sid in story_ids],
        textposition="top center",
        name='Wall Nodes'
    ))
    
    fig.update_layout(
        title="Enhanced Building Graph Structure (Clean Grid Edges)",
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Elevation',
            aspectmode='data'
        ),
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()
    
    return fig

def process_json_files():
    """Process all JSON files with progress tracking"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json") and f != "transformation_config.json"]
    
    if not json_files:
        print(f"No JSON files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    print(f"Using CLEAN GRID edge logic (~3-4 edges per node)")
    
    statistics = []
    
    for filename in tqdm(json_files, desc="Processing files"):
        json_path = os.path.join(INPUT_DIR, filename)
        try:
            # Build graph with clean grid edges
            graph, centroids, edges = build_enhanced_graph(json_path)
            
            # Save graph
            output_path = os.path.join(OUTPUT_DIR, f"supergraph_{filename.replace('.json', '.pt')}")
            torch.save(graph, output_path)
            
            # Collect statistics
            num_nodes = graph.x.size(0)
            num_edges = graph.edge_index.size(1) // 2  # Bidirectional
            num_floors = len(graph.story_id.unique())
            
            stats = {
                'filename': filename,
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'num_floors': num_floors,
                'nodes_per_floor': num_nodes / num_floors,
                'edges_per_node': (num_edges * 2) / num_nodes if num_nodes > 0 else 0,
                'edge_density': num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
            }
            statistics.append(stats)
            
            pass  # Skip visualization on server
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save statistics
    if statistics:
        df_stats = pd.DataFrame(statistics)
        stats_path = os.path.join(OUTPUT_DIR, "conversion_statistics.csv")
        df_stats.to_csv(stats_path, index=False)
        
        print("\n" + "="*50)
        print("CONVERSION SUMMARY")
        print("="*50)
        print(f"Total files processed: {len(statistics)}")
        print(f"Average nodes per graph: {df_stats['num_nodes'].mean():.1f}")
        print(f"Average edges per graph: {df_stats['num_edges'].mean():.1f}")
        print(f"Average edges per node: {df_stats['edges_per_node'].mean():.2f}")
        print(f"Average edge density: {df_stats['edge_density'].mean():.4f}")
        print(f"\nStatistics saved to: {stats_path}")

def verify_graph_file(pt_file_path):
    """Verify the converted graph file"""
    data = torch.load(pt_file_path)
    print(f"\nVerifying {os.path.basename(pt_file_path)}:")
    print(f"Node features shape: {data.x.shape}")
    print(f"Number of edges: {data.edge_index.shape}")
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        print(f"Edge features shape: {data.edge_attr.shape}")
    print(f"Targets shape: {data.y.shape}")
    print(f"Unique floors: {data.story_id.unique().tolist()}")

if __name__ == "__main__":
    print("Enhanced Graph Converter with Clean Grid Edges")
    print("=" * 50)
    print("Edge rules:")
    print("  1. Consecutive in row (by X)")
    print("  2. Consecutive in column (by Y)")
    print("  3. Each node -> closest in next row")
    print("  4. Each node -> closest in next column")
    print("  5. Vertical edges between floors")
    print("=" * 50)
    process_json_files()
    
    # Verify a sample file
    sample_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pt')][:1]
    if sample_files:
        verify_graph_file(os.path.join(OUTPUT_DIR, sample_files[0]))
