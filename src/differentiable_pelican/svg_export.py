from __future__ import annotations

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
        rotation_deg = float(params.rotation.item()) * 180.0 / 3.14159

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


## Tests


def test_export_svg_creates_file(tmp_path: Path):
    """
    Test that SVG export creates a file.
    """
    import torch

    from differentiable_pelican.geometry import Circle

    device = torch.device("cpu")
    circle = Circle(cx=0.5, cy=0.5, radius=0.2, device=device)

    output_path = tmp_path / "test.svg"
    shapes_to_svg([circle], width=100, height=100, output_path=output_path)

    assert output_path.exists()


def test_export_svg_valid_xml(tmp_path: Path):
    """
    Test that exported SVG is valid XML.
    """
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
    """
    Test that SVG contains expected shape elements.
    """
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
    """
    Test that SVG viewBox matches dimensions.
    """
    import torch

    from differentiable_pelican.geometry import Circle

    device = torch.device("cpu")
    circle = Circle(cx=0.5, cy=0.5, radius=0.2, device=device)

    output_path = tmp_path / "test.svg"
    shapes_to_svg([circle], width=128, height=256, output_path=output_path)

    content = output_path.read_text()
    assert 'viewBox="0 0 128 256"' in content
