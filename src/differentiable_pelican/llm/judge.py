from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from textwrap import dedent

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent.parent / ".env.local")


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
    metrics: dict | None = None,
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
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")

    client = anthropic.Anthropic(api_key=api_key)

    # Read SVG code
    svg_code = svg_path.read_text()

    # Encode images
    def encode_image(path: Path) -> tuple[str, str]:
        with path.open("rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            return data, "image/jpeg"
        elif ext == ".png":
            return data, "image/png"
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
        prompt_text += f"\n\nOptimization metrics:\n{json.dumps(metrics, indent=2)}"

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

    # Call API
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[{"role": "user", "content": content_blocks}],
    )

    # Parse response
    response_text = response.content[0].text
    try:
        response_json = json.loads(response_text)
        return JudgeFeedback(**response_json)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(
            f"Failed to parse judge response as JSON: {e}\nResponse: {response_text}"
        ) from e
