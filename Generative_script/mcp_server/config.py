"""
mcp_server/config.py - Path configuration
"""
from pathlib import Path

# Project root: ~/CGVAE_RL_Tran (where python -m disc_diffusion.run runs)
PROJECT_ROOT = Path("/home/elbasanier/CGVAE_RL_Tran")

# Diffusion model
DIFFUSION_CKPT = PROJECT_ROOT / "disc_diffusion" / "outputs" / "checkpoints_v3" / "best_diffusion.pt"
MASKS_PATH = PROJECT_ROOT / "disc_diffusion" / "masks.pt"

# Drift model (18 features)
DRIFT_CKPT = PROJECT_ROOT / "disc_diffusion" / "drift" / "best_model_18feat.pt"
GRAPH_CONVERTER = PROJECT_ROOT / "disc_diffusion" / "drift" / "O1graph_converter_updated_v2.py"

# ETABS template
ETABS_TEMPLATE = PROJECT_ROOT / "disc_diffusion" / "drift" / "3x6_between_10.xlsx"

# Output
OUTPUT_DIR = PROJECT_ROOT / "disc_diffusion" / "outputs" / "generated"

# Image settings
MAX_IMAGE_WIDTH = 600
PNG_OPTIMIZE = True

# Generation defaults
DEFAULT_FLOORS = 10
DEFAULT_CANDIDATES = 5
MAX_CANDIDATES = 20
GENERATION_TIMEOUT = 600

# No plan type restriction - let run.py handle mask availability
VALID_PLAN_TYPES = [
    (3, 3), (4, 3), (4, 4), (5, 5), (2, 4), (2, 7), (2, 10),
    (3, 7), (3, 10), (6, 6), (3, 6), (4, 6), (4, 2), (6, 2),
    (7, 3), (8, 2), (8, 3),
]
