"""MCP server: lets AI assistants and agents (Claude Desktop, Claude Code, any MCP client)
call the classifier as a tool.

Run:  python mcp_server.py            (stdio transport)
Claude Desktop config:
  {"mcpServers": {"chest-ct": {"command": "python", "args": ["/path/to/repo/mcp_server.py"]}}}
"""

import base64
import json
import os
from pathlib import Path

import yaml
from mcp.server.mcpserver import MCPServer

from cnnClassifier.pipeline.prediction import PredictionPipeline

ROOT = Path(__file__).parent
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())

mcp = MCPServer("chest-ct-classifier")
_pipeline: PredictionPipeline | None = None


def pipeline() -> PredictionPipeline:
    global _pipeline
    if _pipeline is None:
        model_path = os.getenv("MODEL_PATH", str(ROOT / "model" / "model.h5"))
        _pipeline = PredictionPipeline(model_path, PARAMS["REVIEW_THRESHOLD"])
    return _pipeline


@mcp.tool()
def classify_ct_scan(image_path: str, save_heatmap: bool = True) -> dict:
    """Classify a chest CT slice (PNG/JPEG) as adenocarcinoma or normal.

    Returns the label, confidence, per-class probabilities and whether the result
    should go to a human expert. With save_heatmap, a Grad-CAM overlay is written
    next to the image as <name>_gradcam.png. Research demo only, not for diagnosis.
    """
    path = Path(image_path).expanduser()
    if not path.is_file():
        return {"error": f"File not found: {path}"}

    result = pipeline().predict(path.read_bytes(), explain=save_heatmap)
    heatmap = result.pop("heatmap_png", None)
    if heatmap:
        out = path.with_name(f"{path.stem}_gradcam.png")
        out.write_bytes(base64.b64decode(heatmap))
        result["heatmap_path"] = str(out)
    result["disclaimer"] = "Research model trained on a small public dataset; not a medical device."
    return result


@mcp.tool()
def model_card() -> dict:
    """Held-out test metrics, data split and limitations of the served model."""
    evaluation = ROOT / "reports" / "evaluation.json"
    split = ROOT / "reports" / "data_split.json"
    return {
        "architecture": "VGG16 (ImageNet, frozen) + global average pooling, dropout, softmax head",
        "input": "224x224 RGB",
        "classes": ["Adenocarcinoma Cancer", "Normal"],
        "test_metrics": json.loads(evaluation.read_text()) if evaluation.exists() else None,
        "data_split": json.loads(split.read_text()) if split.exists() else None,
        "review_threshold": PARAMS["REVIEW_THRESHOLD"],
        "limitations": (ROOT / "MODEL_CARD.md").read_text() if (ROOT / "MODEL_CARD.md").exists() else "",
    }


if __name__ == "__main__":
    mcp.run()
