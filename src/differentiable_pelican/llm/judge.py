from __future__ import annotations

import base64
import json
from pathlib import Path
from textwrap import dedent

from pydantic import BaseModel

from differentiable_pelican.llm.client import llm_call_json
from differentiable_pelican.optimizer import OptimizationMetrics

_SUMMARY_KEYS = ("mse", "edge", "ssim", "perimeter", "degeneracy", "canvas", "total")


def summarize_metrics(metrics: OptimizationMetrics) -> dict[str, object]:
    """
    Build a compact summary of optimization metrics for the LLM judge.
    Avoids sending per-step loss_history (which can be 500+ entries).
    """
    history = metrics.get("loss_history", [])
    summary: dict[str, object] = {
        "steps_completed": metrics.get("steps_completed", 0),
        "resolution": metrics.get("resolution", 0),
        "final_loss": metrics.get("final_loss", 0.0),
    }
    if history:
        summary["initial_loss"] = {k: round(history[0][k], 6) for k in _SUMMARY_KEYS if k in history[0]}
        summary["final_loss_breakdown"] = {k: round(history[-1][k], 6) for k in _SUMMARY_KEYS if k in history[-1]}
    return summary


class JudgeFeedback(BaseModel):
    """
    Structured feedback from judge evaluating SVG output.
    Extends ImageValidation with SVG-aware critiques.
    """

    overall_quality: float  # 0-1 score
    resembles_pelican: bool
    geometric_accuracy: str  # Free-form description
    missing_features: list[str]
    topology_issues: list[str]
    suggestions: list[str]
    ready_for_refinement: bool
    similarity_to_target: float  # 0-1


def judge_svg(
    svg_path: Path,
    png_path: Path,
    target_path: Path,
    metrics: OptimizationMetrics | None = None,
) -> JudgeFeedback:
    """
    Use LLM to evaluate optimized SVG and provide structured feedback.

    Args:
        svg_path: Path to SVG file
        png_path: Path to rendered PNG
        target_path: Path to target image
        metrics: Optional optimization metrics

    Returns:
        Structured feedback for refinement
    """
    # Read SVG code
    svg_code = svg_path.read_text()

    # Encode images
    def encode_image(path: Path) -> tuple[str, str]:
        with path.open("rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            return data, "image/jpeg"
        else:
            return data, "image/png"

    png_b64, png_media = encode_image(png_path)
    target_b64, target_media = encode_image(target_path)

    # Build prompt
    prompt_text = dedent("""
        You are evaluating a computer-generated SVG drawing of a pelican.

        Below is:
        1. The rendered PNG output
        2. The target image (desired pelican)
        3. The SVG code

        Please provide a structured critique in JSON format with these fields:

        - overall_quality (float 0-1): Overall quality score
        - resembles_pelican (bool): Does it look like a pelican?
        - geometric_accuracy (str): Description of geometric accuracy issues
        - missing_features (list[str]): What pelican features are missing (e.g., "pouch", "long beak")
        - topology_issues (list[str]): Structural problems (e.g., "head too small", "body-head overlap wrong")
        - suggestions (list[str]): Specific suggestions for improvement
        - ready_for_refinement (bool): Is it good enough or needs major structural changes?
        - similarity_to_target (float 0-1): How similar to target image

        Focus on:
        - Pelican-specific anatomy: long beak, throat pouch, body shape
        - Relative proportions and positioning
        - Whether shapes capture the essential pelican structure
        - Comparison to the target image

        Respond with ONLY valid JSON, no other text.
        """).strip()

    if metrics:
        summary = summarize_metrics(metrics)
        prompt_text += f"\n\nOptimization metrics:\n{json.dumps(summary, indent=2)}"

    # Prepare content
    content_blocks = [
        {
            "type": "text",
            "text": "Rendered PNG output:",
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": png_media,
                "data": png_b64,
            },
        },
        {
            "type": "text",
            "text": "Target image:",
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": target_media,
                "data": target_b64,
            },
        },
        {
            "type": "text",
            "text": f"SVG Code:\n```svg\n{svg_code}\n```\n\n{prompt_text}",
        },
    ]

    response_json = llm_call_json(content_blocks, max_tokens=2048)
    return JudgeFeedback(**response_json)


## Tests


def test_summarize_metrics_excludes_loss_history():
    metrics: OptimizationMetrics = {
        "loss_history": [
            {"mse": 0.5, "edge": 0.1, "ssim": 0.3, "perimeter": 0.01, "degeneracy": 0.0, "canvas": 0.0, "total": 0.91},
            {"mse": 0.4, "edge": 0.08, "ssim": 0.25, "perimeter": 0.01, "degeneracy": 0.0, "canvas": 0.0, "total": 0.74},
            {"mse": 0.1, "edge": 0.02, "ssim": 0.05, "perimeter": 0.01, "degeneracy": 0.0, "canvas": 0.0, "total": 0.18},
        ],
        "final_loss": 0.18,
        "steps_completed": 3,
        "resolution": 128,
    }
    summary = summarize_metrics(metrics)
    assert "loss_history" not in summary
    assert summary["steps_completed"] == 3
    assert summary["resolution"] == 128
    assert summary["final_loss"] == 0.18
    assert summary["initial_loss"]["total"] == 0.91  # pyright: ignore[reportIndexIssue]
    assert summary["final_loss_breakdown"]["total"] == 0.18  # pyright: ignore[reportIndexIssue]
    # Summary should be compact (no per-step data)
    summary_json = json.dumps(summary)
    assert len(summary_json) < 500
