from __future__ import annotations

from textwrap import dedent
from typing import Any

from pydantic import BaseModel

from differentiable_pelican.llm.client import llm_call_json
from differentiable_pelican.llm.judge import JudgeFeedback


class ShapeEdit(BaseModel):
    """
    Single edit action for a shape.

    LLM returns mixed types in changes/init_params: floats, strings ("+20%"),
    or [x, y] lists for triangle vertices, so we use Any for flexibility.
    """

    type: str  # "modify", "add", "remove"
    shape: str  # Shape name/identifier
    changes: dict[str, Any] | None = None  # For modify
    primitive: str | None = None  # For add: "circle", "ellipse", "triangle"
    init_params: dict[str, Any] | None = None  # For add


class ArchitectResponse(BaseModel):
    """
    Structured response from architect with proposed edits.
    """

    actions: list[ShapeEdit]
    rationale: str


def architect_edits(feedback: JudgeFeedback) -> ArchitectResponse:
    """
    Use LLM to propose geometric edits based on judge feedback.

    Args:
        feedback: Structured critique from judge

    Returns:
        Proposed edits to apply
    """
    feedback_str = feedback.model_dump_json(indent=2)

    prompt = dedent(f"""
        You are an AI architect designing geometric shapes for a pelican drawing.

        You received this feedback from a judge:
        {feedback_str}

        Your task: propose concrete geometric edits to improve the drawing.

        Available shape types:
        - circle: {{cx, cy, radius}} (all in [0,1] normalized coordinates)
        - ellipse: {{cx, cy, rx, ry, rotation}} (rotation in radians)
        - triangle: {{v0, v1, v2}} (three vertices, each [x, y] in [0,1])

        Available edit types:
        1. "modify": Change parameters of existing shape
           - Use percentage changes like "+20%" or absolute values
        2. "add": Add new shape
           - Provide primitive type and init_params
        3. "remove": Remove shape by name

        Current pelican structure (typical):
        - body: large ellipse
        - head: circle
        - beak: triangle
        - eye: small circle
        - wing: ellipse

        Respond with JSON in this format:
        {{
            "actions": [
                {{
                    "type": "modify",
                    "shape": "beak",
                    "changes": {{"length": "+20%", "rotation": "+0.1"}}
                }},
                {{
                    "type": "add",
                    "shape": "pouch",
                    "primitive": "ellipse",
                    "init_params": {{"cx": 0.5, "cy": 0.6, "rx": 0.08, "ry": 0.06, "rotation": 0.0}}
                }}
            ],
            "rationale": "Pelicans have distinctive throat pouches and longer beaks..."
        }}

        Respond with ONLY valid JSON, no other text.
        """).strip()

    content_blocks = [{"type": "text", "text": prompt}]
    response_json = llm_call_json(content_blocks, max_tokens=2048)
    return ArchitectResponse(**response_json)
