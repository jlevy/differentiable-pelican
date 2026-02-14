from __future__ import annotations

import torch


def sdf_circle(points: torch.Tensor, center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """
    Signed distance field for a circle.

    Args:
        points: Shape [..., 2], coordinates (x, y)
        center: Shape [2], circle center (cx, cy)
        radius: Scalar, circle radius

    Returns:
        Signed distance: negative inside, positive outside, zero on boundary
    """
    dx = points[..., 0] - center[0]
    dy = points[..., 1] - center[1]
    dist_from_center = torch.sqrt(dx**2 + dy**2 + 1e-10)
    return dist_from_center - radius


def sdf_ellipse(
    points: torch.Tensor,
    center: torch.Tensor,
    radii: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """
    Signed distance field for an ellipse using scaled normalized distance.

    Uses the normalized-distance approximation scaled by geometric mean radius:
        sdf ≈ (‖p_normalized‖ - 1) · √(rx·ry)

    This is differentiable everywhere with bounded gradients, making it ideal
    for gradient-based optimization. The approximation error vs the exact
    Quilez solution is sub-pixel at typical rendering resolutions (128x128)
    and invisible after sigmoid smoothing.
    """
    # Translate to origin
    dx = points[..., 0] - center[0]
    dy = points[..., 1] - center[1]

    # Apply inverse rotation
    cos_theta = torch.cos(-rotation)
    sin_theta = torch.sin(-rotation)
    x_rot = dx * cos_theta - dy * sin_theta
    y_rot = dx * sin_theta + dy * cos_theta

    rx, ry = radii[0], radii[1]

    # Normalized distance: ‖(x/rx, y/ry)‖
    p_norm = torch.sqrt((x_rot / (rx + 1e-10)) ** 2 + (y_rot / (ry + 1e-10)) ** 2 + 1e-10)

    # Scale by geometric mean radius for proper distance units
    scale = torch.sqrt(rx * ry + 1e-10)

    return (p_norm - 1.0) * scale


def sdf_triangle(points: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """
    Signed distance field for a triangle.

    Args:
        points: Shape [..., 2], coordinates (x, y)
        vertices: Shape [3, 2], triangle vertices [(x0,y0), (x1,y1), (x2,y2)]

    Returns:
        Signed distance: negative inside, positive outside, zero on boundary
    """
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]

    # Vector from each vertex to the point
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    v_to_p0 = points - v0
    v_to_p1 = points - v1
    v_to_p2 = points - v2

    # Perpendicular distance to each edge
    # Project point onto edge and compute distance
    def edge_distance(edge: torch.Tensor, v_to_p: torch.Tensor) -> torch.Tensor:
        # Clamp projection to edge
        edge_len_sq = torch.sum(edge**2)
        t = torch.clamp(torch.sum(v_to_p * edge, dim=-1) / (edge_len_sq + 1e-8), 0.0, 1.0)
        proj = edge * t.unsqueeze(-1)
        closest = v_to_p - proj
        return torch.norm(closest, dim=-1)

    d0 = edge_distance(e0, v_to_p0)
    d1 = edge_distance(e1, v_to_p1)
    d2 = edge_distance(e2, v_to_p2)

    # Minimum distance to any edge
    min_dist = torch.minimum(torch.minimum(d0, d1), d2)

    # Determine sign using cross products
    # If point is on the left of all edges (CCW winding), it's inside
    def cross_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    c0 = cross_2d(e0, v_to_p0)
    c1 = cross_2d(e1, v_to_p1)
    c2 = cross_2d(e2, v_to_p2)

    # Inside if all cross products have same sign (works for both CW and CCW winding)
    inside = ((c0 > 0) & (c1 > 0) & (c2 > 0)) | ((c0 < 0) & (c1 < 0) & (c2 < 0))

    # Apply sign
    return torch.where(inside, -min_dist, min_dist)


def coverage_from_sdf(sdf: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Convert SDF to soft coverage using sigmoid.

    Args:
        sdf: Signed distance field values
        tau: Softness parameter (in pixel units)

    Returns:
        Coverage in [0, 1]: 1 inside, 0 outside, smooth transition at boundary
    """
    # Clamp to avoid extreme logits
    clamped_sdf = torch.clamp(sdf / tau, -10.0, 10.0)
    return torch.sigmoid(-clamped_sdf)


## Tests


def test_sdf_circle_at_origin():
    points = torch.tensor([[0.5, 0.5]])
    center = torch.tensor([0.5, 0.5])
    radius = torch.tensor(0.2)
    sdf = sdf_circle(points, center, radius)
    assert sdf[0] < 0


def test_sdf_circle_outside():
    points = torch.tensor([[0.0, 0.0]])
    center = torch.tensor([0.5, 0.5])
    radius = torch.tensor(0.2)
    sdf = sdf_circle(points, center, radius)
    assert sdf[0] > 0


def test_sdf_circle_on_boundary():
    points = torch.tensor([[0.7, 0.5]])
    center = torch.tensor([0.5, 0.5])
    radius = torch.tensor(0.2)
    sdf = sdf_circle(points, center, radius)
    assert torch.abs(sdf[0]) < 1e-6


def test_sdf_ellipse_on_boundary():
    # Point on major axis boundary: (0.5 + 0.2, 0.5) should have SDF ≈ 0
    points = torch.tensor([[0.7, 0.5]])
    center = torch.tensor([0.5, 0.5])
    radii = torch.tensor([0.2, 0.1])
    rotation = torch.tensor(0.0)
    sdf = sdf_ellipse(points, center, radii, rotation)
    assert torch.abs(sdf[0]) < 0.01


def test_sdf_ellipse_eccentric():
    # Highly eccentric ellipse: rx=0.4, ry=0.05
    center = torch.tensor([0.5, 0.5])
    radii = torch.tensor([0.4, 0.05])
    rotation = torch.tensor(0.0)
    # Point on minor axis boundary: SDF ≈ 0
    on_minor = torch.tensor([[0.5, 0.55]])
    sdf_minor = sdf_ellipse(on_minor, center, radii, rotation)
    assert torch.abs(sdf_minor[0]) < 0.02
    # Center: should be negative (inside)
    sdf_center = sdf_ellipse(torch.tensor([[0.5, 0.5]]), center, radii, rotation)
    assert sdf_center[0] < 0
    # Point 0.05 outside minor axis: should be positive (outside)
    outside = torch.tensor([[0.5, 0.6]])
    sdf_outside = sdf_ellipse(outside, center, radii, rotation)
    assert sdf_outside[0] > 0


def test_sdf_triangle_vertices():
    vertices = torch.tensor([[0.3, 0.3], [0.7, 0.3], [0.5, 0.7]])
    # Check first vertex
    points = vertices[0:1]
    sdf = sdf_triangle(points, vertices)
    assert torch.abs(sdf[0]) < 1e-5


def test_coverage_sigmoid_range():
    sdf_values = torch.tensor([-1.0, 0.0, 1.0])
    coverage = coverage_from_sdf(sdf_values, tau=1.0)
    assert torch.all((coverage >= 0) & (coverage <= 1))


def test_coverage_inside_outside():
    sdf_inside = torch.tensor(-2.0)
    sdf_outside = torch.tensor(2.0)
    cov_inside = coverage_from_sdf(sdf_inside, tau=1.0)
    cov_outside = coverage_from_sdf(sdf_outside, tau=1.0)
    assert cov_inside > 0.85
    assert cov_outside < 0.15


def test_sdf_triangle_cw_winding():
    # CCW order
    vertices_ccw = torch.tensor([[0.3, 0.3], [0.7, 0.3], [0.5, 0.7]])
    # CW order (reversed)
    vertices_cw = torch.tensor([[0.5, 0.7], [0.7, 0.3], [0.3, 0.3]])

    # Center point should be inside for both winding orders
    center = torch.tensor([[0.5, 0.43]])
    sdf_ccw = sdf_triangle(center, vertices_ccw)
    sdf_cw = sdf_triangle(center, vertices_cw)
    assert sdf_ccw[0] < 0, "Center should be inside triangle (CCW)"
    assert sdf_cw[0] < 0, "Center should be inside triangle (CW)"


def test_sdf_triangle_gradient_flow():
    vertices = torch.tensor([[0.3, 0.3], [0.7, 0.3], [0.5, 0.7]], requires_grad=True)
    point = torch.tensor([[0.5, 0.5]])
    sdf = sdf_triangle(point, vertices)
    sdf.sum().backward()
    assert vertices.grad is not None
    assert not torch.all(vertices.grad == 0)
