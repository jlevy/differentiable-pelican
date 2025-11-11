from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CircleParams:
    """
    Circle parameters derived from unconstrained tensors.
    """

    cx: torch.Tensor  # [0, 1]
    cy: torch.Tensor  # [0, 1]
    radius: torch.Tensor  # > 0


@dataclass
class EllipseParams:
    """
    Ellipse parameters derived from unconstrained tensors.
    """

    cx: torch.Tensor  # [0, 1]
    cy: torch.Tensor  # [0, 1]
    rx: torch.Tensor  # > 0
    ry: torch.Tensor  # > 0
    rotation: torch.Tensor  # radians


@dataclass
class TriangleParams:
    """
    Triangle parameters derived from unconstrained tensors.
    """

    vertices: torch.Tensor  # [3, 2], each vertex in [0, 1]^2


class Shape(nn.Module):
    """
    Base class for parameterized shapes.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def get_params(self):
        """
        Return constrained parameters.
        """
        raise NotImplementedError

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance field at given points.
        """
        raise NotImplementedError


class Circle(Shape):
    """
    Differentiable circle with constrained parameters.
    """

    def __init__(
        self,
        cx: float,
        cy: float,
        radius: float,
        device: torch.device,
    ):
        super().__init__(device)
        # Store unconstrained parameters
        # Use logit for [0,1] values and inv_softplus for positive values
        self.cx_raw = nn.Parameter(torch.tensor(self._logit(cx), device=device))
        self.cy_raw = nn.Parameter(torch.tensor(self._logit(cy), device=device))
        self.radius_raw = nn.Parameter(torch.tensor([self._inv_softplus(radius)], device=device))

    @staticmethod
    def _logit(x: float, eps: float = 1e-6) -> float:
        """
        Inverse sigmoid.
        """
        x = max(eps, min(1 - eps, x))
        return float(torch.logit(torch.tensor(x)).item())

    @staticmethod
    def _inv_softplus(x: float, eps: float = 1e-6) -> float:
        """
        Inverse softplus: softplus(y) = x => y = log(exp(x) - 1)
        """
        x = max(eps, x)
        return float(torch.log(torch.exp(torch.tensor(x)) - 1.0).item())

    def get_params(self) -> CircleParams:
        """
        Derive constrained parameters from unconstrained.
        """
        cx = torch.sigmoid(self.cx_raw)
        cy = torch.sigmoid(self.cy_raw)
        radius = torch.nn.functional.softplus(self.radius_raw)
        return CircleParams(cx=cx, cy=cy, radius=radius)

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance field.
        """
        from differentiable_pelican.sdf import sdf_circle

        params = self.get_params()
        center = torch.stack([params.cx, params.cy])
        return sdf_circle(points, center, params.radius)


class Ellipse(Shape):
    """
    Differentiable ellipse with constrained parameters.
    """

    def __init__(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        rotation: float,
        device: torch.device,
    ):
        super().__init__(device)
        self.cx_raw = nn.Parameter(torch.tensor(Circle._logit(cx), device=device))
        self.cy_raw = nn.Parameter(torch.tensor(Circle._logit(cy), device=device))
        self.rx_raw = nn.Parameter(torch.tensor([Circle._inv_softplus(rx)], device=device))
        self.ry_raw = nn.Parameter(torch.tensor([Circle._inv_softplus(ry)], device=device))
        self.rotation_raw = nn.Parameter(torch.tensor([rotation], device=device))

    def get_params(self) -> EllipseParams:
        """
        Derive constrained parameters.
        """
        cx = torch.sigmoid(self.cx_raw)
        cy = torch.sigmoid(self.cy_raw)
        rx = torch.nn.functional.softplus(self.rx_raw)
        ry = torch.nn.functional.softplus(self.ry_raw)
        rotation = self.rotation_raw  # Rotation is unconstrained
        return EllipseParams(cx=cx, cy=cy, rx=rx, ry=ry, rotation=rotation)

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance field.
        """
        from differentiable_pelican.sdf import sdf_ellipse

        params = self.get_params()
        center = torch.stack([params.cx, params.cy])
        radii = torch.stack([params.rx, params.ry])
        return sdf_ellipse(points, center, radii, params.rotation)


class Triangle(Shape):
    """
    Differentiable triangle with constrained parameters.
    """

    def __init__(
        self,
        v0: tuple[float, float],
        v1: tuple[float, float],
        v2: tuple[float, float],
        device: torch.device,
    ):
        super().__init__(device)
        # Each vertex coordinate is constrained to [0, 1] via sigmoid
        self.v0_raw = nn.Parameter(
            torch.tensor([Circle._logit(v0[0]), Circle._logit(v0[1])], device=device)
        )
        self.v1_raw = nn.Parameter(
            torch.tensor([Circle._logit(v1[0]), Circle._logit(v1[1])], device=device)
        )
        self.v2_raw = nn.Parameter(
            torch.tensor([Circle._logit(v2[0]), Circle._logit(v2[1])], device=device)
        )

    def get_params(self) -> TriangleParams:
        """
        Derive constrained parameters.
        """
        v0 = torch.sigmoid(self.v0_raw)
        v1 = torch.sigmoid(self.v1_raw)
        v2 = torch.sigmoid(self.v2_raw)
        vertices = torch.stack([v0, v1, v2])
        return TriangleParams(vertices=vertices)

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance field.
        """
        from differentiable_pelican.sdf import sdf_triangle

        params = self.get_params()
        return sdf_triangle(points, params.vertices)


def create_initial_pelican(device: torch.device) -> list[Shape]:
    """
    Create hard-coded initial pelican geometry.

    Structure:
    - Body: large ellipse
    - Head: circle above body
    - Beak: triangle pointing right
    - Eye: small circle on head
    - Wing: ellipse on body (optional)
    """
    shapes: list[Shape] = [
        # Body: ellipse in lower center
        Ellipse(
            cx=0.5,
            cy=0.6,
            rx=0.15,
            ry=0.25,
            rotation=0.0,
            device=device,
        ),
        # Head: circle above body
        Circle(
            cx=0.5,
            cy=0.35,
            radius=0.12,
            device=device,
        ),
        # Beak: triangle pointing right
        Triangle(
            v0=(0.58, 0.32),
            v1=(0.58, 0.38),
            v2=(0.75, 0.35),
            device=device,
        ),
        # Eye: small circle
        Circle(
            cx=0.52,
            cy=0.33,
            radius=0.02,
            device=device,
        ),
        # Wing: ellipse on body
        Ellipse(
            cx=0.48,
            cy=0.6,
            rx=0.08,
            ry=0.15,
            rotation=-0.3,
            device=device,
        ),
    ]

    return shapes


## Tests


def test_circle_params_in_range():
    """
    Test that circle parameters are within valid ranges.
    """
    device = torch.device("cpu")
    circle = Circle(cx=0.5, cy=0.5, radius=0.1, device=device)
    params = circle.get_params()
    assert 0 <= params.cx <= 1
    assert 0 <= params.cy <= 1
    assert params.radius > 0


def test_ellipse_params_in_range():
    """
    Test that ellipse parameters are within valid ranges.
    """
    device = torch.device("cpu")
    ellipse = Ellipse(cx=0.5, cy=0.5, rx=0.2, ry=0.1, rotation=0.5, device=device)
    params = ellipse.get_params()
    assert 0 <= params.cx <= 1
    assert 0 <= params.cy <= 1
    assert params.rx > 0
    assert params.ry > 0


def test_triangle_vertices_in_range():
    """
    Test that triangle vertices are within [0, 1].
    """
    device = torch.device("cpu")
    triangle = Triangle(v0=(0.3, 0.3), v1=(0.7, 0.3), v2=(0.5, 0.7), device=device)
    params = triangle.get_params()
    assert torch.all((params.vertices >= 0) & (params.vertices <= 1))


def test_create_initial_pelican():
    """
    Test that initial pelican geometry can be created.
    """
    device = torch.device("cpu")
    shapes = create_initial_pelican(device)
    assert len(shapes) == 5
    assert isinstance(shapes[0], Ellipse)
    assert isinstance(shapes[1], Circle)
    assert isinstance(shapes[2], Triangle)
