from __future__ import annotations

import re

import torch

from differentiable_pelican.geometry import (
    Circle,
    Ellipse,
    Shape,
    Triangle,
    inv_softplus,
    logit_param,
)
from differentiable_pelican.llm.architect import ShapeEdit


def parse_percentage(value_str: str | float, current: float) -> float:
    """
    Parse percentage change like "+20%" or "-10%", or absolute float value.
    """
    if isinstance(value_str, (int, float)):
        return float(value_str)

    match = re.match(r"([+-]?)(\d+(?:\.\d+)?)%", str(value_str))
    if match:
        sign = match.group(1)
        pct = float(match.group(2)) / 100.0
        if sign == "+":
            return current * (1.0 + pct)
        elif sign == "-":
            return current * (1.0 - pct)
        else:
            return current * pct
    else:
        return float(value_str)


def _apply_intensity_edit(edit: ShapeEdit, shape: Shape) -> None:
    """Apply intensity edit to any shape type."""
    if edit.changes and "intensity" in edit.changes:
        current_intensity = float(shape.intensity.item())
        new_intensity = parse_percentage(edit.changes["intensity"], current_intensity)
        new_intensity = max(0.01, min(0.99, new_intensity))
        shape.intensity_raw.data = torch.tensor(logit_param(new_intensity), device=shape.device)


def apply_edit_to_shape(edit: ShapeEdit, shape: Shape) -> None:
    """
    Apply a modify edit to an existing shape (in-place).
    """
    if edit.type != "modify" or not edit.changes:
        return

    # Apply intensity edit (works for all shape types)
    _apply_intensity_edit(edit, shape)

    # Get current parameters
    if isinstance(shape, Circle):
        params = shape.get_params()
        if "cx" in edit.changes:
            new_cx = parse_percentage(edit.changes["cx"], float(params.cx.item()))
            shape.cx_raw.data = torch.tensor(logit_param(new_cx), device=shape.device)
        if "cy" in edit.changes:
            new_cy = parse_percentage(edit.changes["cy"], float(params.cy.item()))
            shape.cy_raw.data = torch.tensor(logit_param(new_cy), device=shape.device)
        if "radius" in edit.changes:
            new_r = parse_percentage(edit.changes["radius"], float(params.radius.item()))
            shape.radius_raw.data = torch.tensor([inv_softplus(new_r)], device=shape.device)

    elif isinstance(shape, Ellipse):
        params = shape.get_params()
        if "cx" in edit.changes:
            new_cx = parse_percentage(edit.changes["cx"], float(params.cx.item()))
            shape.cx_raw.data = torch.tensor(logit_param(new_cx), device=shape.device)
        if "cy" in edit.changes:
            new_cy = parse_percentage(edit.changes["cy"], float(params.cy.item()))
            shape.cy_raw.data = torch.tensor(logit_param(new_cy), device=shape.device)
        if "rx" in edit.changes:
            new_rx = parse_percentage(edit.changes["rx"], float(params.rx.item()))
            shape.rx_raw.data = torch.tensor([inv_softplus(new_rx)], device=shape.device)
        if "ry" in edit.changes:
            new_ry = parse_percentage(edit.changes["ry"], float(params.ry.item()))
            shape.ry_raw.data = torch.tensor([inv_softplus(new_ry)], device=shape.device)
        if "rotation" in edit.changes:
            new_rot = parse_percentage(edit.changes["rotation"], float(params.rotation.item()))
            shape.rotation_raw.data = torch.tensor([new_rot], device=shape.device)

    elif isinstance(shape, Triangle):
        # Handle vertex modifications: values can be [x, y] lists or individual coords
        for vname, raw_param in [("v0", shape.v0_raw), ("v1", shape.v1_raw), ("v2", shape.v2_raw)]:
            if vname in edit.changes:
                val = edit.changes[vname]
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    new_x = max(0.01, min(0.99, float(val[0])))
                    new_y = max(0.01, min(0.99, float(val[1])))
                    raw_param.data = torch.tensor(
                        [logit_param(new_x), logit_param(new_y)], device=shape.device
                    )


def create_shape_from_edit(edit: ShapeEdit, device: torch.device) -> Shape | None:
    """
    Create a new shape from an add edit.
    """
    if edit.type != "add" or not edit.init_params:
        return None

    params = edit.init_params
    intensity = params.get("intensity", 0.0)

    if edit.primitive == "circle":
        return Circle(
            cx=params.get("cx", 0.5),
            cy=params.get("cy", 0.5),
            radius=params.get("radius", 0.1),
            device=device,
            intensity=intensity,
        )
    elif edit.primitive == "ellipse":
        return Ellipse(
            cx=params.get("cx", 0.5),
            cy=params.get("cy", 0.5),
            rx=params.get("rx", 0.15),
            ry=params.get("ry", 0.1),
            rotation=params.get("rotation", 0.0),
            device=device,
            intensity=intensity,
        )
    elif edit.primitive == "triangle":
        v0 = params.get("v0", [0.3, 0.3])
        v1 = params.get("v1", [0.7, 0.3])
        v2 = params.get("v2", [0.5, 0.7])
        if not isinstance(v0, (list, tuple)):
            v0 = [0.3, 0.3]
        if not isinstance(v1, (list, tuple)):
            v1 = [0.7, 0.3]
        if not isinstance(v2, (list, tuple)):
            v2 = [0.5, 0.7]
        return Triangle(
            v0=(float(v0[0]), float(v0[1])),
            v1=(float(v1[0]), float(v1[1])),
            v2=(float(v2[0]), float(v2[1])),
            device=device,
            intensity=intensity,
        )
    else:
        return None


def parse_edits(
    edits: list[ShapeEdit],
    shapes: list[Shape],
    shape_names: list[str],
) -> tuple[list[Shape], list[str]]:
    """
    Apply edits to shape list.

    Args:
        edits: List of edit actions
        shapes: Current list of shapes
        shape_names: Names corresponding to shapes

    Returns:
        new_shapes: Updated list of shapes
        new_names: Updated list of names
    """
    device = shapes[0].device if shapes else torch.device("cpu")

    # Build name -> shape mapping
    shape_map = {name: shape for name, shape in zip(shape_names, shapes, strict=True)}

    for edit in edits:
        if edit.type == "modify":
            if edit.shape in shape_map:
                apply_edit_to_shape(edit, shape_map[edit.shape])

        elif edit.type == "add":
            new_shape = create_shape_from_edit(edit, device)
            if new_shape:
                shape_map[edit.shape] = new_shape

        elif edit.type == "remove":
            if edit.shape in shape_map:
                del shape_map[edit.shape]

    # Rebuild lists
    new_names = list(shape_map.keys())
    new_shapes = list(shape_map.values())

    return new_shapes, new_names
