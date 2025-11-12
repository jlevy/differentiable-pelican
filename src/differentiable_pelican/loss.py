from __future__ import annotations

from collections.abc import Sequence

import torch

from differentiable_pelican.geometry import Circle, Ellipse, Shape, Triangle


def mse_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error between rendered and target images.
    """
    return torch.mean((rendered - target) ** 2)


def perimeter_prior(shapes: Sequence[Shape]) -> torch.Tensor:
    """
    Penalize large shapes to encourage compactness.

    For circles and ellipses, approximate perimeter.
    For triangles, sum edge lengths.
    """
    total = torch.tensor(0.0, device=shapes[0].device)

    for shape in shapes:
        if isinstance(shape, Circle):
            params = shape.get_params()
            # Perimeter = 2 * pi * r
            perimeter = 2 * 3.14159 * params.radius
            total = total + perimeter
        elif isinstance(shape, Ellipse):
            params = shape.get_params()
            # Approximate perimeter for ellipse (Ramanujan's formula)
            a, b = params.rx, params.ry
            h = ((a - b) ** 2) / ((a + b) ** 2)
            perimeter = 3.14159 * (a + b) * (1 + 3 * h / (10 + torch.sqrt(4 - 3 * h)))
            total = total + perimeter
        elif isinstance(shape, Triangle):
            params = shape.get_params()
            v = params.vertices
            # Sum of edge lengths
            edge_lengths = (
                torch.norm(v[1] - v[0]) + torch.norm(v[2] - v[1]) + torch.norm(v[0] - v[2])
            )
            total = total + edge_lengths

    return total


def triangle_degeneracy_penalty(
    shapes: Sequence[Shape], min_edge_length: float = 0.01
) -> torch.Tensor:
    """
    Penalize degenerate triangles with very small edges.
    """
    penalty = torch.tensor(0.0, device=shapes[0].device)

    for shape in shapes:
        if isinstance(shape, Triangle):
            params = shape.get_params()
            v = params.vertices
            edge1 = torch.norm(v[1] - v[0])
            edge2 = torch.norm(v[2] - v[1])
            edge3 = torch.norm(v[0] - v[2])

            # Penalize if any edge is below minimum
            penalty = penalty + torch.relu(min_edge_length - edge1)
            penalty = penalty + torch.relu(min_edge_length - edge2)
            penalty = penalty + torch.relu(min_edge_length - edge3)

    return penalty


def on_canvas_penalty(shapes: Sequence[Shape], margin: float = 0.05) -> torch.Tensor:
    """
    Penalize shapes that go outside [0, 1] bounds with margin.
    """
    penalty = torch.tensor(0.0, device=shapes[0].device)

    for shape in shapes:
        if isinstance(shape, Circle):
            params = shape.get_params()
            # Check if center + radius goes outside
            penalty = penalty + torch.relu(params.cx + params.radius - (1.0 - margin))
            penalty = penalty + torch.relu(margin - (params.cx - params.radius))
            penalty = penalty + torch.relu(params.cy + params.radius - (1.0 - margin))
            penalty = penalty + torch.relu(margin - (params.cy - params.radius))
        elif isinstance(shape, Ellipse):
            params = shape.get_params()
            # Approximate with max radius
            max_r = torch.maximum(params.rx, params.ry)
            penalty = penalty + torch.relu(params.cx + max_r - (1.0 - margin))
            penalty = penalty + torch.relu(margin - (params.cx - max_r))
            penalty = penalty + torch.relu(params.cy + max_r - (1.0 - margin))
            penalty = penalty + torch.relu(margin - (params.cy - max_r))
        elif isinstance(shape, Triangle):
            params = shape.get_params()
            v = params.vertices
            # Check each vertex
            for i in range(3):
                penalty = penalty + torch.relu(v[i, 0] - (1.0 - margin))
                penalty = penalty + torch.relu(margin - v[i, 0])
                penalty = penalty + torch.relu(v[i, 1] - (1.0 - margin))
                penalty = penalty + torch.relu(margin - v[i, 1])

    return penalty


def total_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    shapes: Sequence[Shape],
    perimeter_weight: float = 0.001,
    degeneracy_weight: float = 0.1,
    canvas_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute total loss with all components.

    Returns:
        total_loss: Combined loss tensor
        breakdown: Dictionary of individual loss components
    """
    loss_mse = mse_loss(rendered, target)
    loss_perimeter = perimeter_prior(shapes)
    loss_degeneracy = triangle_degeneracy_penalty(shapes)
    loss_canvas = on_canvas_penalty(shapes)

    total = (
        loss_mse
        + perimeter_weight * loss_perimeter
        + degeneracy_weight * loss_degeneracy
        + canvas_weight * loss_canvas
    )

    breakdown = {
        "mse": float(loss_mse.item()),
        "perimeter": float(loss_perimeter.item()),
        "degeneracy": float(loss_degeneracy.item()),
        "canvas": float(loss_canvas.item()),
        "total": float(total.item()),
    }

    return total, breakdown


## Tests


def test_mse_loss_zero_for_identical():
    """
    Test that MSE is zero for identical images.
    """
    img = torch.rand(64, 64)
    loss = mse_loss(img, img)
    assert torch.abs(loss) < 1e-6


def test_mse_loss_positive_for_different():
    """
    Test that MSE is positive for different images.
    """
    img1 = torch.zeros(64, 64)
    img2 = torch.ones(64, 64)
    loss = mse_loss(img1, img2)
    assert loss > 0


def test_mse_loss_has_gradient():
    """
    Test that MSE loss backpropagates.
    """
    img1 = torch.rand(64, 64, requires_grad=True)
    img2 = torch.rand(64, 64)
    loss = mse_loss(img1, img2)
    loss.backward()
    assert img1.grad is not None


def test_perimeter_prior_larger_for_big_shapes():
    """
    Test that larger shapes have higher perimeter prior.
    """
    device = torch.device("cpu")
    small_circle = Circle(cx=0.5, cy=0.5, radius=0.1, device=device)
    large_circle = Circle(cx=0.5, cy=0.5, radius=0.3, device=device)

    prior_small = perimeter_prior([small_circle])
    prior_large = perimeter_prior([large_circle])

    assert prior_large > prior_small


def test_perimeter_prior_gradient_nonzero():
    """
    Test that perimeter prior has gradients.
    """
    device = torch.device("cpu")
    circle = Circle(cx=0.5, cy=0.5, radius=0.2, device=device)
    prior = perimeter_prior([circle])
    prior.backward()
    assert circle.radius_raw.grad is not None


def test_total_loss_combines_components():
    """
    Test that total loss includes all components.
    """
    device = torch.device("cpu")
    rendered = torch.rand(64, 64)
    target = torch.rand(64, 64)
    shapes = [Circle(cx=0.5, cy=0.5, radius=0.2, device=device)]

    loss, breakdown = total_loss(rendered, target, shapes)

    assert "mse" in breakdown
    assert "perimeter" in breakdown
    assert "total" in breakdown
    assert breakdown["total"] > breakdown["mse"]
