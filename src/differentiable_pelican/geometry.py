from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from typing_extensions import override


def logit_param(x: float, eps: float = 1e-6) -> float:
    """
    Inverse sigmoid for constraining parameters to [0,1].
    """
    x = max(eps, min(1 - eps, x))
    return float(torch.logit(torch.tensor(x)).item())


def inv_softplus(x: float, eps: float = 1e-6) -> float:
    """
    Inverse softplus: softplus(y) = x => y = log(exp(x) - 1).
    Used for constraining parameters to be positive.
    """
    x = max(eps, x)
    return float(torch.log(torch.exp(torch.tensor(x)) - 1.0).item())


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

    Each shape has an optimizable intensity parameter (grayscale value)
    that controls how dark the shape renders. This allows the optimizer
    to match varying tones in the target image.
    """

    device: torch.device
    intensity_raw: nn.Parameter

    def __init__(self, device: torch.device, intensity: float = 0.0):
        super().__init__()
        self.device = device
        # Intensity in [0, 1] via sigmoid. 0.0 = black, 1.0 = white.
        # Default 0.0 (maps to ~0.5 via sigmoid, but logit(0.0) -> -inf,
        # so we use the raw value directly and let sigmoid constrain it)
        self.intensity_raw = nn.Parameter(
            torch.tensor(logit_param(max(0.01, min(0.99, intensity))), device=device)
        )

    @property
    def intensity(self) -> torch.Tensor:
        """Get the constrained intensity value in [0, 1]."""
        return torch.sigmoid(self.intensity_raw)

    def get_params(self):
        """
        Return constrained parameters.
        """
        raise NotImplementedError

    def sdf(self, _points: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportUnusedParameter]
        """
        Compute signed distance field at given points.
        """
        raise NotImplementedError


class Circle(Shape):
    """
    Differentiable circle with constrained parameters.
    """

    cx_raw: nn.Parameter
    cy_raw: nn.Parameter
    radius_raw: nn.Parameter

    def __init__(
        self,
        cx: float,
        cy: float,
        radius: float,
        device: torch.device,
        intensity: float = 0.0,
    ):
        super().__init__(device, intensity=intensity)
        # Store unconstrained parameters
        # Use logit for [0,1] values and inv_softplus for positive values
        self.cx_raw = nn.Parameter(torch.tensor(logit_param(cx), device=device))
        self.cy_raw = nn.Parameter(torch.tensor(logit_param(cy), device=device))
        self.radius_raw = nn.Parameter(torch.tensor([inv_softplus(radius)], device=device))

    @override
    def get_params(self) -> CircleParams:
        """
        Derive constrained parameters from unconstrained.
        """
        cx = torch.sigmoid(self.cx_raw)
        cy = torch.sigmoid(self.cy_raw)
        radius = torch.nn.functional.softplus(self.radius_raw)
        return CircleParams(cx=cx, cy=cy, radius=radius)

    @override
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

    cx_raw: nn.Parameter
    cy_raw: nn.Parameter
    rx_raw: nn.Parameter
    ry_raw: nn.Parameter
    rotation_raw: nn.Parameter

    def __init__(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        rotation: float,
        device: torch.device,
        intensity: float = 0.0,
    ):
        super().__init__(device, intensity=intensity)
        self.cx_raw = nn.Parameter(torch.tensor(logit_param(cx), device=device))
        self.cy_raw = nn.Parameter(torch.tensor(logit_param(cy), device=device))
        self.rx_raw = nn.Parameter(torch.tensor([inv_softplus(rx)], device=device))
        self.ry_raw = nn.Parameter(torch.tensor([inv_softplus(ry)], device=device))
        self.rotation_raw = nn.Parameter(torch.tensor([rotation], device=device))

    @override
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

    @override
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

    v0_raw: nn.Parameter
    v1_raw: nn.Parameter
    v2_raw: nn.Parameter

    def __init__(
        self,
        v0: tuple[float, float],
        v1: tuple[float, float],
        v2: tuple[float, float],
        device: torch.device,
        intensity: float = 0.0,
    ):
        super().__init__(device, intensity=intensity)
        # Each vertex coordinate is constrained to [0, 1] via sigmoid
        self.v0_raw = nn.Parameter(
            torch.tensor([logit_param(v0[0]), logit_param(v0[1])], device=device)
        )
        self.v1_raw = nn.Parameter(
            torch.tensor([logit_param(v1[0]), logit_param(v1[1])], device=device)
        )
        self.v2_raw = nn.Parameter(
            torch.tensor([logit_param(v2[0]), logit_param(v2[1])], device=device)
        )

    @override
    def get_params(self) -> TriangleParams:
        """
        Derive constrained parameters.
        """
        v0 = torch.sigmoid(self.v0_raw)
        v1 = torch.sigmoid(self.v1_raw)
        v2 = torch.sigmoid(self.v2_raw)
        vertices = torch.stack([v0, v1, v2])
        return TriangleParams(vertices=vertices)

    @override
    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance field.
        """
        from differentiable_pelican.sdf import sdf_triangle

        params = self.get_params()
        return sdf_triangle(points, params.vertices)


def create_initial_pelican(device: torch.device) -> tuple[list[Shape], list[str]]:
    """
    Create hard-coded initial pelican geometry matching the target image
    (side view, facing right).

    Returns:
        shapes: List of Shape objects
        names: Names for each shape (used by refinement loop)
    """
    shapes: list[Shape] = [
        # Body: large ellipse, light gray (pelican body is pale)
        Ellipse(
            cx=0.42, cy=0.55, rx=0.22, ry=0.28, rotation=-0.3,
            device=device, intensity=0.35,
        ),
        # Neck: tall narrow ellipse, light gray
        Ellipse(
            cx=0.52, cy=0.35, rx=0.06, ry=0.15, rotation=-0.2,
            device=device, intensity=0.40,
        ),
        # Head: circle at top right, light gray
        Circle(
            cx=0.58, cy=0.18, radius=0.08,
            device=device, intensity=0.35,
        ),
        # Beak upper: triangle pointing right, darker (olive/green in real pelican)
        Triangle(
            v0=(0.62, 0.15), v1=(0.62, 0.22), v2=(0.88, 0.20),
            device=device, intensity=0.25,
        ),
        # Beak lower / pouch: triangle below beak, darker
        Triangle(
            v0=(0.62, 0.22), v1=(0.88, 0.20), v2=(0.65, 0.28),
            device=device, intensity=0.30,
        ),
        # Wing: ellipse overlaying body, medium gray with texture
        Ellipse(
            cx=0.38, cy=0.50, rx=0.18, ry=0.15, rotation=-0.4,
            device=device, intensity=0.30,
        ),
        # Tail: small triangle at back, darker
        Triangle(
            v0=(0.18, 0.52), v1=(0.25, 0.48), v2=(0.12, 0.60),
            device=device, intensity=0.20,
        ),
        # Eye: tiny circle on head, very dark
        Circle(
            cx=0.60, cy=0.16, radius=0.015,
            device=device, intensity=0.05,
        ),
        # Feet: small ellipse at bottom, dark green
        Ellipse(
            cx=0.45, cy=0.88, rx=0.06, ry=0.04, rotation=0.0,
            device=device, intensity=0.15,
        ),
    ]

    names = [
        "body", "neck", "head", "beak_upper", "beak_lower",
        "wing", "tail", "eye", "feet",
    ]

    return shapes, names


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
    shapes, names = create_initial_pelican(device)
    assert len(shapes) == 9
    assert len(names) == 9
    assert isinstance(shapes[0], Ellipse)  # body
    assert "body" in names
    assert "head" in names
    assert "beak_upper" in names
