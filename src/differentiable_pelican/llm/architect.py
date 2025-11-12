from __future__ import annotations

import json
import os
from textwrap import dedent

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from differentiable_pelican.llm.judge import JudgeFeedback

load_dotenv()


class ShapeEdit(BaseModel):
    """
    Single edit action for a shape.
    """

    type: str  # "modify", "add", "remove"
    shape: str  # Shape name/identifier
    changes: dict[str, str | float] | None = None  # For modify
    primitive: str | None = None  # For add: "circle", "ellipse", "triangle"
    init_params: dict[str, float] | None = None  # For add


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
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")

    client = anthropic.Anthropic(api_key=api_key)

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

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract text from first text content block
    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text = block.text  # type: ignore[attr-defined]
            break

    if not response_text:
        raise ValueError(f"No text content in response: {response.content}")

    try:
        response_json = json.loads(response_text)
        return ArchitectResponse(**response_json)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(
            f"Failed to parse architect response as JSON: {e}\nResponse: {response_text}"
        ) from e
