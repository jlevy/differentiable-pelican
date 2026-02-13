from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from differentiable_pelican.geometry import Circle, Ellipse, Shape, Triangle


def mse_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error between rendered and target images.
    """
    return torch.mean((rendered - target) ** 2)


def edge_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Edge-aware loss using Sobel gradient magnitude.

    Compares edge maps of rendered and target images to encourage
    matching sharp boundaries and contours.
    """
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=rendered.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=rendered.device)

    # Reshape for conv2d: [1, 1, 3, 3]
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    # Reshape images: [1, 1, H, W]
    r = rendered.unsqueeze(0).unsqueeze(0)
    t = target.unsqueeze(0).unsqueeze(0)

    # Compute gradients
    r_gx = torch.nn.functional.conv2d(r, sobel_x, padding=1)
    r_gy = torch.nn.functional.conv2d(r, sobel_y, padding=1)
    t_gx = torch.nn.functional.conv2d(t, sobel_x, padding=1)
    t_gy = torch.nn.functional.conv2d(t, sobel_y, padding=1)

    # Gradient magnitude
    r_mag = torch.sqrt(r_gx ** 2 + r_gy ** 2 + 1e-8)
    t_mag = torch.sqrt(t_gx ** 2 + t_gy ** 2 + 1e-8)

    return torch.mean((r_mag - t_mag) ** 2)


def ssim_loss(
    rendered: torch.Tensor, target: torch.Tensor, window_size: int = 7
) -> torch.Tensor:
    """
    Structural similarity loss (1 - SSIM).

    Lower is better. Uses a Gaussian window for local statistics.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=rendered.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    window = g.unsqueeze(0) * g.unsqueeze(1)  # Outer product
    window = window / window.sum()
    window = window.view(1, 1, window_size, window_size)

    # Reshape
    r = rendered.unsqueeze(0).unsqueeze(0)
    t = target.unsqueeze(0).unsqueeze(0)
    pad = window_size // 2

    # Local means
    mu_r = torch.nn.functional.conv2d(r, window, padding=pad)
    mu_t = torch.nn.functional.conv2d(t, window, padding=pad)

    mu_r_sq = mu_r ** 2
    mu_t_sq = mu_t ** 2
    mu_rt = mu_r * mu_t

    # Local variances
    sigma_r_sq = torch.nn.functional.conv2d(r * r, window, padding=pad) - mu_r_sq
    sigma_t_sq = torch.nn.functional.conv2d(t * t, window, padding=pad) - mu_t_sq
    sigma_rt = torch.nn.functional.conv2d(r * t, window, padding=pad) - mu_rt

    ssim_map = ((2 * mu_rt + C1) * (2 * sigma_rt + C2)) / (
        (mu_r_sq + mu_t_sq + C1) * (sigma_r_sq + sigma_t_sq + C2)
    )

    return 1.0 - ssim_map.mean()


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
            perimeter = 2 * math.pi * params.radius
            total = total + perimeter
        elif isinstance(shape, Ellipse):
            params = shape.get_params()
            # Approximate perimeter for ellipse (Ramanujan's formula)
            a, b = params.rx, params.ry
            h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-8)
            perimeter = math.pi * (a + b) * (1 + 3 * h / (10 + torch.sqrt(4 - 3 * h)))
            total = total + perimeter
        elif isinstance(shape, Triangle):
            params = shape.get_params()
            v = params.vertices
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
            penalty = penalty + torch.relu(params.cx + params.radius - (1.0 - margin))
            penalty = penalty + torch.relu(margin - (params.cx - params.radius))
            penalty = penalty + torch.relu(params.cy + params.radius - (1.0 - margin))
            penalty = penalty + torch.relu(margin - (params.cy - params.radius))
        elif isinstance(shape, Ellipse):
            params = shape.get_params()
            max_r = torch.maximum(params.rx, params.ry)
            penalty = penalty + torch.relu(params.cx + max_r - (1.0 - margin))
            penalty = penalty + torch.relu(margin - (params.cx - max_r))
            penalty = penalty + torch.relu(params.cy + max_r - (1.0 - margin))
            penalty = penalty + torch.relu(margin - (params.cy - max_r))
        elif isinstance(shape, Triangle):
            params = shape.get_params()
            v = params.vertices
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
    edge_weight: float = 0.1,
    ssim_weight: float = 0.05,
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
    loss_edge = edge_loss(rendered, target) if edge_weight > 0 else torch.tensor(0.0)
    loss_ssim = ssim_loss(rendered, target) if ssim_weight > 0 else torch.tensor(0.0)

    total = (
        loss_mse
        + perimeter_weight * loss_perimeter
        + degeneracy_weight * loss_degeneracy
        + canvas_weight * loss_canvas
        + edge_weight * loss_edge
        + ssim_weight * loss_ssim
    )

    breakdown = {
        "mse": float(loss_mse.item()),
        "edge": float(loss_edge.item()),
        "ssim": float(loss_ssim.item()),
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


def test_edge_loss_zero_for_identical():
    """
    Test that edge loss is zero for identical images.
    """
    img = torch.rand(64, 64)
    loss = edge_loss(img, img)
    assert loss < 1e-5


def test_edge_loss_detects_differences():
    """
    Test that edge loss is higher when edges differ.
    """
    img1 = torch.zeros(64, 64)
    img1[20:40, 20:40] = 1.0  # Sharp box
    img2 = torch.zeros(64, 64)
    loss = edge_loss(img1, img2)
    assert loss > 0.001


def test_ssim_loss_zero_for_identical():
    """
    Test that SSIM loss is near zero for identical images.
    """
    img = torch.rand(64, 64)
    loss = ssim_loss(img, img)
    assert loss < 0.01


def test_ssim_loss_high_for_different():
    """
    Test that SSIM loss is high for very different images.
    """
    img1 = torch.zeros(64, 64)
    img2 = torch.ones(64, 64)
    loss = ssim_loss(img1, img2)
    assert loss > 0.5


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

    _, breakdown = total_loss(rendered, target, shapes)

    assert "mse" in breakdown
    assert "edge" in breakdown
    assert "ssim" in breakdown
    assert "perimeter" in breakdown
    assert "total" in breakdown
    assert breakdown["total"] > breakdown["mse"]
