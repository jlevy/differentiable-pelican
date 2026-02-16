from __future__ import annotations

from textwrap import dedent
from typing import Any

from pydantic import BaseModel

from differentiable_pelican.geometry import Circle, Ellipse, Shape, Triangle
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


def describe_scene(shapes: list[Shape], shape_names: list[str]) -> str:
    """
    Build a text description of the current scene for the architect prompt.
    Lists each shape's name, primitive type, and current parameter values.
    """
    lines: list[str] = []
    for name, shape in zip(shape_names, shapes, strict=True):
        if isinstance(shape, Circle):
            params = shape.get_params()
            lines.append(
                f'- "{name}" (circle): cx={params.cx.item():.3f}, cy={params.cy.item():.3f}, '
                f"radius={params.radius.item():.3f}, intensity={shape.intensity.item():.3f}"
            )
        elif isinstance(shape, Ellipse):
            params = shape.get_params()
            lines.append(
                f'- "{name}" (ellipse): cx={params.cx.item():.3f}, cy={params.cy.item():.3f}, '
                f"rx={params.rx.item():.3f}, ry={params.ry.item():.3f}, "
                f"rotation={params.rotation.item():.3f}, intensity={shape.intensity.item():.3f}"
            )
        elif isinstance(shape, Triangle):
            params = shape.get_params()
            v = params.vertices
            lines.append(
                f'- "{name}" (triangle): v0=[{v[0, 0].item():.3f}, {v[0, 1].item():.3f}], '
                f"v1=[{v[1, 0].item():.3f}, {v[1, 1].item():.3f}], "
                f"v2=[{v[2, 0].item():.3f}, {v[2, 1].item():.3f}], "
                f"intensity={shape.intensity.item():.3f}"
            )
    return "\n".join(lines)


def architect_edits(
    feedback: JudgeFeedback,
    shapes: list[Shape] | None = None,
    shape_names: list[str] | None = None,
) -> ArchitectResponse:
    """
    Use LLM to propose geometric edits based on judge feedback.
    When shapes and names are provided, the prompt includes the exact
    current scene state so the LLM can propose valid edits.
    """
    feedback_str = feedback.model_dump_json(indent=2)

    # Build scene description from actual shapes if available
    if shapes is not None and shape_names is not None:
        scene_block = describe_scene(shapes, shape_names)
    else:
        scene_block = "(no scene state available)"

    prompt = dedent(f"""
        You are an AI architect designing geometric shapes for a pelican drawing.

        You received this feedback from a judge:
        {feedback_str}

        Current scene ({len(shapes) if shapes else 0} shapes):
        {scene_block}

        Your task: propose concrete geometric edits to improve the drawing.

        Editable fields per primitive type:
        - circle: cx, cy, radius, intensity (all in [0,1] normalized coordinates)
        - ellipse: cx, cy, rx, ry, rotation (radians), intensity
        - triangle: v0, v1, v2 (each [x,y] in [0,1]), intensity

        Available edit types:
        1. "modify": Change parameters of an existing shape (use exact shape name from list above)
           - Use percentage changes like "+20%" or "-10%" or absolute values
        2. "add": Add a new shape with a descriptive name
           - Provide "primitive" (circle/ellipse/triangle) and "init_params"
        3. "remove": Remove a shape by its exact name

        Respond with JSON in this format:
        {{
            "actions": [
                {{
                    "type": "modify",
                    "shape": "<exact shape name from list above>",
                    "changes": {{"cx": "+5%", "ry": 0.08}}
                }},
                {{
                    "type": "add",
                    "shape": "pouch",
                    "primitive": "ellipse",
                    "init_params": {{"cx": 0.5, "cy": 0.6, "rx": 0.08, "ry": 0.06, "rotation": 0.0, "intensity": 0.2}}
                }}
            ],
            "rationale": "Brief explanation of why these edits improve the drawing."
        }}

        IMPORTANT: Use only shape names that exist in the current scene for modify/remove actions.
        Use only the editable fields listed above for each primitive type.

        Respond with ONLY valid JSON, no other text.
        """).strip()

    content_blocks = [{"type": "text", "text": prompt}]
    response_json = llm_call_json(content_blocks, max_tokens=2048)
    return ArchitectResponse(**response_json)


## Tests


def test_describe_scene_includes_all_shapes():
    import torch

    device = torch.device("cpu")
    shapes: list[Shape] = [
        Circle(cx=0.5, cy=0.3, radius=0.1, device=device, intensity=0.2),
        Ellipse(cx=0.5, cy=0.5, rx=0.3, ry=0.15, rotation=0.1, device=device, intensity=0.1),
        Triangle(v0=(0.3, 0.4), v1=(0.7, 0.4), v2=(0.5, 0.2), device=device, intensity=0.05),
    ]
    names = ["head", "body", "beak"]
    desc = describe_scene(shapes, names)
    assert '"head" (circle)' in desc
    assert '"body" (ellipse)' in desc
    assert '"beak" (triangle)' in desc
    assert "cx=" in desc
    assert "v0=" in desc
    assert "intensity=" in desc
