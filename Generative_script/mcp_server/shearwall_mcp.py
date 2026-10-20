"""
mcp_server/shearwall_mcp.py
============================
MCP server for the Shear Wall Layout Generation pipeline.

Usage (stdio):  python shearwall_mcp.py
Usage (HTTP):   python shearwall_mcp.py --transport http --port 8000
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import subprocess
import sys
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP, Image
from pydantic import BaseModel, Field, ConfigDict, field_validator

from config import (
    PROJECT_ROOT,
    DIFFUSION_CKPT,
    MASKS_PATH,
    DRIFT_CKPT,
    GRAPH_CONVERTER,
    OUTPUT_DIR,
    ETABS_TEMPLATE,
    MAX_IMAGE_WIDTH,
    DEFAULT_FLOORS,
    DEFAULT_CANDIDATES,
    MAX_CANDIDATES,
    GENERATION_TIMEOUT,
    VALID_PLAN_TYPES,
)

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("shearwall_mcp")

# ---------------------------------------------------------------------------
_port = 8000
for i, arg in enumerate(sys.argv):
    if arg == "--port" and i + 1 < len(sys.argv):
        _port = int(sys.argv[i + 1])

mcp = FastMCP("shearwall_mcp", host="0.0.0.0", port=_port)

# Image server port (MCP on 8000, images on 8001)
IMAGE_PORT = 8001


# ===========================================================================
# Helpers
# ===========================================================================

def _compress_png(png_path: Path, max_width: int = MAX_IMAGE_WIDTH) -> bytes:
    """Resize and compress PNG for MCP transport."""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(png_path)
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except ImportError:
        return png_path.read_bytes()


def _find_latest_output(lx: int, ly: int) -> Optional[Path]:
    """Find most recent output folder for a plan type."""
    matches = sorted(OUTPUT_DIR.glob(f"plan_{lx}x{ly}_*"))
    return matches[-1] if matches else None


def _image_url(folder_name: str, filename: str) -> str:
    """Build URL to access an image via the image server."""
    return f"http://localhost:{IMAGE_PORT}/{folder_name}/{filename}"


def _build_generate_command(lx, ly, floors, shear_ratio, num_candidates, save_etabs):
    """Build CLI command for run.py generate."""
    cmd = [
        sys.executable, "-m", "disc_diffusion.run", "generate",
        "--masks", str(MASKS_PATH),
        "--ckpt", str(DIFFUSION_CKPT),
        "--etabs_template", str(ETABS_TEMPLATE),
        "--etabs_template", str(ETABS_TEMPLATE),
        "--lx", str(lx), "--ly", str(ly),
        "--floors", str(floors),
        "--shear", str(shear_ratio),
        "--candidates", str(num_candidates),
        "--output_dir", str(OUTPUT_DIR),
        "--save_before_json",
    ]
    if DRIFT_CKPT.exists():
        cmd.extend(["--drift_checkpoint", str(DRIFT_CKPT)])
    if GRAPH_CONVERTER.exists():
        cmd.extend(["--graph_converter", str(GRAPH_CONVERTER)])
    # Always export ETABS xlsx
    cmd.append("--save_etabs_xlsx")
    if ETABS_TEMPLATE.exists():
        cmd.extend(["--etabs_template", str(ETABS_TEMPLATE)])
    return cmd


def _slim_result(result_data: dict) -> dict:
    """Extract only essential metrics (no wall coords)."""
    return {
        "layout_x": result_data.get("layout_x"),
        "layout_y": result_data.get("layout_y"),
        "num_floors": result_data.get("num_floors"),
        "shear_ratio": result_data.get("shear_ratio"),
        "x_drift_mean": result_data.get("x_drift_mean"),
        "y_drift_mean": result_data.get("y_drift_mean"),
        "drift_pred_per_floor": result_data.get("drift_pred_per_floor"),
    }


def _start_image_server():
    """Start a simple HTTP server to serve generated images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    handler = partial(SimpleHTTPRequestHandler, directory=str(OUTPUT_DIR))
    server = HTTPServer(("0.0.0.0", IMAGE_PORT), handler)
    logger.info("Image server running on http://0.0.0.0:%d", IMAGE_PORT)
    server.serve_forever()


# ===========================================================================
# Input models
# ===========================================================================

class GenerateLayoutInput(BaseModel):
    """Input for shear wall layout generation."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    lx: int = Field(..., description="Number of bays in X direction. In plan AxB, lx=A (first number)", ge=2, le=99)
    ly: int = Field(..., description="Number of bays in Y direction. In plan AxB, ly=B (second number)", ge=2, le=99)
    floors: int = Field(default=DEFAULT_FLOORS, description="Number of stories", ge=1, le=20)
    shear_ratio: float = Field(default=0.0, description="Target shear wall ratio (0=auto)", ge=0.0, le=1.0)
    num_candidates: int = Field(default=DEFAULT_CANDIDATES, description="Candidates to generate", ge=1, le=MAX_CANDIDATES)
    save_etabs: bool = Field(default=False, description="Export ETABS xlsx")

    @field_validator("num_candidates")
    @classmethod
    def cap_candidates(cls, v: int) -> int:
        return min(v, MAX_CANDIDATES)


class GetPlanImagesInput(BaseModel):
    """Input for retrieving images from a previous generation."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    lx: int = Field(..., description="Number of bays in X direction. In plan AxB, lx=A (first number)", ge=2, le=99)
    ly: int = Field(..., description="Number of bays in Y direction. In plan AxB, ly=B (second number)", ge=2, le=99)
    rank: Optional[int] = Field(default=None, description="Specific rank (None = best + table)", ge=1, le=MAX_CANDIDATES)


# ===========================================================================
# Tools
# ===========================================================================

@mcp.tool(
    name="shearwall_generate_layout",
    annotations={
        "title": "Generate Shear Wall Layout",
        "readOnlyHint": False, "destructiveHint": False,
        "idempotentHint": False, "openWorldHint": False,
    },
)
async def shearwall_generate_layout(params: GenerateLayoutInput) -> list:
    """
    Generate shear wall layout candidates using discrete diffusion,
    rank by predicted drift, return best plan image URL + metrics.

    Pipeline: diffusion generate -> GATv2 drift predict -> rank -> best PNG
    """
    # No plan type restriction - run.py handles mask availability

    missing = []
    if not DIFFUSION_CKPT.exists():
        missing.append(f"Diffusion checkpoint: {DIFFUSION_CKPT}")
    if not MASKS_PATH.exists():
        missing.append(f"Masks: {MASKS_PATH}")
    if missing:
        return [f"Error: Missing files: {missing}"]

    cmd = _build_generate_command(
        params.lx, params.ly, params.floors,
        params.shear_ratio, params.num_candidates, params.save_etabs,
    )
    logger.info("Running: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=GENERATION_TIMEOUT,
        )
        if proc.returncode != 0:
            return [f"Generation failed (code {proc.returncode}): {proc.stderr[-1000:]}"]
    except subprocess.TimeoutExpired:
        return [f"Generation timed out after {GENERATION_TIMEOUT}s."]

    out_dir = _find_latest_output(params.lx, params.ly)
    if out_dir is None:
        return ["Error: Output directory not found."]

    # Read result
    result_data = {}
    rp = out_dir / "result.json"
    if rp.exists():
        with open(rp) as f:
            result_data = json.load(f)

    folder = out_dir.name
    response = []

    # 1. Text summary
    response.append(json.dumps({
        "status": "success",
        "plan_type": f"{params.lx}x{params.ly}",
        "output_dir": str(out_dir),
        **_slim_result(result_data),
    }, indent=2, default=str))

    # 2. Image URLs (viewable via SSH tunnel on port 8001)
    best_png = out_dir / "rank01_BEST.png"
    if best_png.exists():
        url = _image_url(folder, "rank01_BEST.png")
        response.append(f"Best candidate (Rank #1): {url}")

    table_png = out_dir / "comparison_table.png"
    if table_png.exists():
        url = _image_url(folder, "comparison_table.png")
        response.append(f"Comparison table: {url}")

    # 3. List all available images
    pngs = sorted(out_dir.glob("rank*.png"))
    if pngs:
        img_list = "\n".join(f"  - {_image_url(folder, p.name)}" for p in pngs)
        response.append(f"All candidates:\n{img_list}")

    return response


@mcp.tool(
    name="shearwall_get_plan_images",
    annotations={
        "title": "Get Plan Images",
        "readOnlyHint": True, "destructiveHint": False,
        "idempotentHint": True, "openWorldHint": False,
    },
)
async def shearwall_get_plan_images(params: GetPlanImagesInput) -> list:
    """
    Retrieve floor plan image URLs from the most recent generation.
    Images are accessible via http://localhost:8001/...
    """
    out_dir = _find_latest_output(params.lx, params.ly)
    if out_dir is None:
        return [f"No results for plan {params.lx}x{params.ly}. Run generate first."]

    folder = out_dir.name

    if params.rank is not None:
        fname = "rank01_BEST.png" if params.rank == 1 else f"rank{params.rank:02d}.png"
        png_path = out_dir / fname
        if not png_path.exists():
            available = [p.name for p in out_dir.glob("rank*.png")]
            return [f"Rank {params.rank} not found. Available: {available}"]
        return [f"Plan {params.lx}x{params.ly} Rank #{params.rank}: {_image_url(folder, fname)}"]

    # Default: metrics + all image URLs
    result_data = {}
    rp = out_dir / "result.json"
    if rp.exists():
        with open(rp) as f:
            result_data = json.load(f)

    slim = _slim_result(result_data)
    response = [f"Plan {params.lx}x{params.ly}: X-drift={slim.get('x_drift_mean')}, Y-drift={slim.get('y_drift_mean')}"]

    best_png = out_dir / "rank01_BEST.png"
    if best_png.exists():
        response.append(f"Best plan: {_image_url(folder, 'rank01_BEST.png')}")

    table_png = out_dir / "comparison_table.png"
    if table_png.exists():
        response.append(f"Comparison: {_image_url(folder, 'comparison_table.png')}")

    return response


@mcp.tool(
    name="shearwall_list_available_plans",
    annotations={
        "title": "List Available Plan Types",
        "readOnlyHint": True, "destructiveHint": False,
        "idempotentHint": True, "openWorldHint": False,
    },
)
async def shearwall_list_available_plans() -> str:
    """List valid plan types, model status, and existing generations."""
    plans = []
    for lx, ly in VALID_PLAN_TYPES:
        plans.append({
            "plan_type": f"{lx}x{ly}", "lx": lx, "ly": ly,
            "footprint": f"{lx*6}m x {ly*6}m",
            "max_wall_slots": (ly+1)*lx*3 + (lx+1)*ly*3,
        })

    existing = []
    if OUTPUT_DIR.exists():
        for d in sorted(OUTPUT_DIR.iterdir()):
            if d.is_dir() and d.name.startswith("plan_"):
                rj = d / "result.json"
                if rj.exists():
                    with open(rj) as f:
                        r = json.load(f)
                    existing.append({
                        "folder": d.name,
                        "plan": f"{r.get('layout_x','?')}x{r.get('layout_y','?')}",
                        "x_drift": r.get("x_drift_mean"),
                        "y_drift": r.get("y_drift_mean"),
                    })

    return json.dumps({
        "valid_plan_types": plans, "total": len(plans),
        "existing_generations": existing,
        "models_ready": {
            "diffusion_model": DIFFUSION_CKPT.exists(),
            "masks": MASKS_PATH.exists(),
            "drift_model": DRIFT_CKPT.exists(),
            "graph_converter": GRAPH_CONVERTER.exists(),
        },
    }, indent=2)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Shear Wall MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "http":
        # Start image server in background thread
        img_thread = threading.Thread(target=_start_image_server, daemon=True)
        img_thread.start()

        import uvicorn
        logger.info("Starting MCP server on HTTP port %d", args.port)
        app = mcp.streamable_http_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        logger.info("Starting MCP server on stdio")
        mcp.run()


if __name__ == "__main__":
    main()