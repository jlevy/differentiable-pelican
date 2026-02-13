from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path

from differentiable_pelican.geometry import Circle, Ellipse, Shape, Triangle


def shapes_to_svg(
    shapes: Sequence[Shape],
    width: int,
    height: int,
    output_path: Path,
) -> None:
    """
    Export shapes to SVG file.

    Args:
        shapes: List of Shape objects
        width: SVG viewBox width (typically same as render resolution)
        height: SVG viewBox height
        output_path: Path to save SVG file
    """
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
    ]
    svg_parts.append(f'  <rect width="{width}" height="{height}" fill="white"/>')

    for i, shape in enumerate(shapes):
        svg_element = _shape_to_svg_element(shape, width, height)
        svg_parts.append(f"  <!-- Shape {i}: {type(shape).__name__} -->")
        svg_parts.append(f"  {svg_element}")

    svg_parts.append("</svg>")

    svg_content = "\n".join(svg_parts)
    output_path.write_text(svg_content)


def _intensity_to_fill(shape: Shape) -> str:
    """Convert a shape's intensity parameter to an SVG fill color string."""
    val = int(float(shape.intensity.item()) * 255)
    val = max(0, min(255, val))
    return f"rgb({val},{val},{val})"


def _shape_to_svg_element(shape: Shape, width: int, height: int) -> str:
    """
    Convert a single shape to SVG element string.
    """
    fill = _intensity_to_fill(shape)

    if isinstance(shape, Circle):
        params = shape.get_params()
        cx = float(params.cx.item()) * width
        cy = float(params.cy.item()) * height
        r = float(params.radius.item()) * width
        return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}"/>'

    elif isinstance(shape, Ellipse):
        params = shape.get_params()
        cx = float(params.cx.item()) * width
        cy = float(params.cy.item()) * height
        rx = float(params.rx.item()) * width
        ry = float(params.ry.item()) * height
        rotation_deg = float(params.rotation.item()) * 180.0 / math.pi

        transform = f"rotate({rotation_deg:.2f} {cx:.2f} {cy:.2f})"
        return f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" fill="{fill}" transform="{transform}"/>'

    elif isinstance(shape, Triangle):
        params = shape.get_params()
        vertices = params.vertices
        v0 = vertices[0]
        v1 = vertices[1]
        v2 = vertices[2]

        x0, y0 = float(v0[0].item()) * width, float(v0[1].item()) * height
        x1, y1 = float(v1[0].item()) * width, float(v1[1].item()) * height
        x2, y2 = float(v2[0].item()) * width, float(v2[1].item()) * height

        points = f"{x0:.2f},{y0:.2f} {x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f}"
        return f'<polygon points="{points}" fill="{fill}"/>'

    else:
        raise ValueError(f"Unknown shape type: {type(shape)}")


def composite_stages_svg(
    stage_svg_paths: Sequence[tuple[Path, str, str]],
    output_path: Path,
) -> None:
    """
    Composite multiple stage SVGs into a single side-by-side SVG with labels.

    Each stage is rendered as a nested <svg> element with its own viewBox,
    so the output scales cleanly at any display size.

    Args:
        stage_svg_paths: List of (svg_path, label, sublabel) tuples.
            label: short name shown below the image (e.g. "Round 8")
            sublabel: detail line (e.g. "17 shapes")
        output_path: Where to write the composite SVG.
    """
    n = len(stage_svg_paths)
    if n == 0:
        return

    img_size = 128
    gap = 4
    margin = 4
    label_gap = 3
    label_h = 12
    sublabel_h = 10

    total_w = margin * 2 + n * img_size + (n - 1) * gap
    total_h = margin + img_size + label_gap + label_h + 1 + sublabel_h + margin

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}">']
    lines.append(f'  <rect width="{total_w}" height="{total_h}" fill="white"/>')

    font = "system-ui, Helvetica, Arial, sans-serif"

    for i, (svg_path, label, sublabel) in enumerate(stage_svg_paths):
        x = margin + i * (img_size + gap)
        y = margin

        # Extract inner content (strip outer <svg> and </svg> tags, comments)
        raw = svg_path.read_text()
        inner = re.sub(r"<svg[^>]*>", "", raw, count=1)
        inner = inner.rsplit("</svg>", 1)[0]
        inner = re.sub(r"<!--.*?-->", "", inner)
        inner = inner.strip()

        # Border
        lines.append(
            f'  <rect x="{x - 0.5}" y="{y - 0.5}" width="{img_size + 1}" '
            f'height="{img_size + 1}" rx="1" fill="none" stroke="#ddd" stroke-width="0.5"/>'
        )

        # Nested SVG
        lines.append(
            f'  <svg x="{x}" y="{y}" width="{img_size}" height="{img_size}" '
            f'viewBox="0 0 128 128">'
        )
        lines.append(f"    {inner}")
        lines.append("  </svg>")

        # Labels
        cx = x + img_size / 2
        ly = margin + img_size + label_gap + label_h
        lines.append(
            f'  <text x="{cx}" y="{ly}" text-anchor="middle" '
            f'font-family="{font}" font-size="9" font-weight="500" fill="#444">'
            f"{label}</text>"
        )
        sy = ly + 1 + sublabel_h
        lines.append(
            f'  <text x="{cx}" y="{sy}" text-anchor="middle" '
            f'font-family="{font}" font-size="7.5" fill="#888">'
            f"{sublabel}</text>"
        )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines))


## Tests


def test_export_svg_creates_file(tmp_path: Path):
    import torch

    from differentiable_pelican.geometry import Circle

    device = torch.device("cpu")
    circle = Circle(cx=0.5, cy=0.5, radius=0.2, device=device)

    output_path = tmp_path / "test.svg"
    shapes_to_svg([circle], width=100, height=100, output_path=output_path)

    assert output_path.exists()


def test_export_svg_valid_xml(tmp_path: Path):
    import xml.etree.ElementTree as ET

    import torch

    from differentiable_pelican.geometry import Circle

    device = torch.device("cpu")
    circle = Circle(cx=0.5, cy=0.5, radius=0.2, device=device)

    output_path = tmp_path / "test.svg"
    shapes_to_svg([circle], width=100, height=100, output_path=output_path)

    # Parse as XML
    tree = ET.parse(output_path)
    root = tree.getroot()

    # Check it's an SVG element
    assert "svg" in root.tag


def test_export_svg_contains_shapes(tmp_path: Path):
    import torch

    from differentiable_pelican.geometry import Circle, Ellipse, Triangle

    device = torch.device("cpu")
    shapes = [
        Circle(cx=0.5, cy=0.5, radius=0.1, device=device),
        Ellipse(cx=0.3, cy=0.3, rx=0.15, ry=0.1, rotation=0.0, device=device),
        Triangle(v0=(0.2, 0.2), v1=(0.4, 0.2), v2=(0.3, 0.4), device=device),
    ]

    output_path = tmp_path / "test.svg"
    shapes_to_svg(shapes, width=100, height=100, output_path=output_path)

    content = output_path.read_text()
    assert "<circle" in content
    assert "<ellipse" in content
    assert "<polygon" in content


def test_svg_viewbox_correct(tmp_path: Path):
    import torch

    from differentiable_pelican.geometry import Circle

    device = torch.device("cpu")
    circle = Circle(cx=0.5, cy=0.5, radius=0.2, device=device)

    output_path = tmp_path / "test.svg"
    shapes_to_svg([circle], width=128, height=256, output_path=output_path)

    content = output_path.read_text()
    assert 'viewBox="0 0 128 256"' in content
