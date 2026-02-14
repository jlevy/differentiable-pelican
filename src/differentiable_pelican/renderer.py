from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from differentiable_pelican.geometry import Shape
from differentiable_pelican.sdf import coverage_from_sdf


def make_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Create normalized coordinate grid for rendering.

    Returns:
        Tensor of shape [H, W, 2] with (x, y) coordinates in [0, 1]
    """
    # Create pixel center coordinates: (i + 0.5) / N for i in [0, N)
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height

    # Create meshgrid
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Stack to [H, W, 2] with (x, y) convention
    grid = torch.stack([xx, yy], dim=-1)

    return grid


def render(
    shapes: list[Shape],
    height: int,
    width: int,
    tau: float,
    device: torch.device,
    grid: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Render shapes to a grayscale image using soft SDF rasterization.

    Args:
        shapes: List of Shape objects (rendered back-to-front)
        height: Image height in pixels
        width: Image width in pixels
        tau: Softness parameter in pixel units
        device: Device to render on
        grid: Optional precomputed grid (for efficiency)

    Returns:
        Grayscale image tensor of shape [H, W] with values in [0, 1]
        1.0 = white background, 0.0 = black foreground
    """
    if grid is None:
        grid = make_grid(height, width, device)

    # Start with white background
    out = torch.ones(height, width, device=device)

    # Composite shapes using alpha-over (painter's algorithm)
    for shape in shapes:
        # Compute SDF
        sdf = shape.sdf(grid)

        # Convert to coverage
        coverage = coverage_from_sdf(sdf, tau)

        # Alpha-over: out = (1 - alpha) * out + alpha * color
        # Use shape's optimizable intensity (0.0 = black, 1.0 = white)
        color = shape.intensity
        out = (1 - coverage) * out + coverage * color

    return out


def render_to_numpy(
    shapes: list[Shape],
    height: int,
    width: int,
    tau: float,
    device: torch.device,
) -> np.ndarray:
    """
    Render shapes and convert to numpy-compatible format.

    Returns:
        Numpy array of shape [H, W] with values in [0, 255] as uint8
    """
    rendered = render(shapes, height, width, tau, device)
    # Convert to uint8 range
    rendered_np = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
    return rendered_np


def save_render(
    shapes: list[Shape],
    height: int,
    width: int,
    tau: float,
    device: torch.device,
    path: str,
) -> None:
    """
    Render shapes and save to PNG file.
    """
    rendered_np = render_to_numpy(shapes, height, width, tau, device)
    img = Image.fromarray(rendered_np, mode="L")
    img.save(path)


## Tests


def test_make_grid_shape():
    device = torch.device("cpu")
    grid = make_grid(64, 128, device)
    assert grid.shape == (64, 128, 2)


def test_make_grid_normalized():
    device = torch.device("cpu")
    grid = make_grid(64, 64, device)
    assert torch.all((grid >= 0) & (grid <= 1))


def test_make_grid_pixel_centers():
    device = torch.device("cpu")
    grid = make_grid(64, 64, device)
    # Pixel centers: first pixel at 0.5/64, last at 63.5/64
    expected_first = 0.5 / 64
    expected_last = 63.5 / 64
    assert torch.allclose(grid[0, 0], torch.tensor([expected_first, expected_first], device=device))
    assert torch.allclose(grid[63, 63], torch.tensor([expected_last, expected_last], device=device))
    # No coordinate should be exactly 0.0 or 1.0
    assert grid.min() > 0.0
    assert grid.max() < 1.0


def test_render_single_circle():
    from differentiable_pelican.geometry import Circle

    device = torch.device("cpu")
    circle = Circle(cx=0.5, cy=0.5, radius=0.2, device=device)
    # Tau in normalized coordinates: 1.0 pixel / 64 pixels
    tau = 1.0 / 64.0
    rendered = render([circle], height=64, width=64, tau=tau, device=device)

    assert rendered.shape == (64, 64)
    # Background should be close to white (1.0)
    assert rendered[0, 0] > 0.9
    # Center should be close to black (0.0)
    assert rendered[32, 32] < 0.1


def test_render_composite_shapes():
    from differentiable_pelican.geometry import Circle

    device = torch.device("cpu")
    circle1 = Circle(cx=0.3, cy=0.5, radius=0.1, device=device)
    circle2 = Circle(cx=0.7, cy=0.5, radius=0.1, device=device)
    tau = 1.0 / 64.0
    rendered = render([circle1, circle2], height=64, width=64, tau=tau, device=device)

    # Both circles should create dark regions
    # Left circle center
    assert rendered[32, 19] < 0.2
    # Right circle center
    assert rendered[32, 44] < 0.2
    # Between circles should be lighter
    assert rendered[32, 32] > 0.5


def test_render_deterministic():
    from differentiable_pelican.geometry import Circle

    device = torch.device("cpu")
    circle = Circle(cx=0.5, cy=0.5, radius=0.15, device=device)
    tau = 1.0 / 64.0

    rendered1 = render([circle], height=64, width=64, tau=tau, device=device)
    rendered2 = render([circle], height=64, width=64, tau=tau, device=device)

    assert torch.allclose(rendered1, rendered2)
