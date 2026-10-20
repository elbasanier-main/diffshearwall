# Shear Wall MCP Server

MCP server for the Shear Wall Layout Generation pipeline.  
Wraps `run.py generate` to expose layout generation via Claude.

## Directory Structure

```
/home/elbasanier/CGVAE_RL_Tran/disc_diffusion/
├── run.py                          # Existing generation pipeline
├── disc_diffusion/                 # Existing model code
├── mcp_server/                     # THIS FOLDER
│   ├── shearwall_mcp.py            # MCP server (main)
│   ├── config.py                   # Path configuration
│   ├── requirements.txt            # Python dependencies
│   └── README.md                   # This file
```

## Setup on jhl705SuperServer

### 1. Copy files to server

```bash
# On server: create directory
mkdir -p /home/elbasanier/CGVAE_RL_Tran/disc_diffusion/mcp_server

# Copy the 3 files into mcp_server/
```

### 2. Install dependencies

```bash
pip install -r mcp_server/requirements.txt --break-system-packages
```

### 3. Update config.py

Edit `mcp_server/config.py` to match your actual checkpoint paths:

```python
# Verify these paths exist on the server:
DIFFUSION_CKPT = ...   # best.pt from diffusion training
MASKS_PATH = ...        # masks.pt from build_masks
DRIFT_CKPT = ...        # best_model.pt from GATv2 training
```

### 4. Verify setup

```bash
cd /home/elbasanier/CGVAE_RL_Tran/disc_diffusion
python mcp_server/shearwall_mcp.py --help
```

## Usage

### Option A: stdio (Claude Desktop / Claude Code)

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "shearwall": {
      "command": "python",
      "args": [
        "/home/elbasanier/CGVAE_RL_Tran/disc_diffusion/mcp_server/shearwall_mcp.py"
      ],
      "cwd": "/home/elbasanier/CGVAE_RL_Tran/disc_diffusion"
    }
  }
}
```

For remote server access via SSH tunnel:

```bash
# On local machine: forward port
ssh -L 8000:localhost:8000 elbasanier@jhl705SuperServer
```

### Option B: HTTP (remote access)

```bash
cd /home/elbasanier/CGVAE_RL_Tran/disc_diffusion
python mcp_server/shearwall_mcp.py --transport http --port 8000
```

## Available Tools

| Tool | Description |
|---|---|
| `shearwall_generate_layout` | Generate & rank shear wall layouts (returns images + metrics) |
| `shearwall_get_plan_images` | Retrieve images from a previous generation |
| `shearwall_list_available_plans` | List valid plan types and model status |

## Example Conversation

```
User: "Generate 5 candidates for a 4x3 building, 10 floors"

Claude: [calls shearwall_generate_layout(lx=4, ly=3, floors=10, num_candidates=5)]
        → Shows floor plan images + drift values + ranking table
```
