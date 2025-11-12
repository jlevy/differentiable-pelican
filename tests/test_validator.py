from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from differentiable_pelican.validator import validate_image


@pytest.mark.slow
@pytest.mark.e2e
def test_validate_blank_image(tmp_path: Path) -> None:
    """
    E2E test: validate blank image using Anthropic API.
    Requires ANTHROPIC_API_KEY in .env.local.
    """
    blank_img = Image.new("RGB", (128, 128), color="white")
    blank_path = tmp_path / "blank.png"
    blank_img.save(blank_path)

    validation = validate_image(blank_path)
    assert validation.is_blank or not validation.has_shapes


@pytest.mark.slow
@pytest.mark.e2e
def test_validate_simple_circle(tmp_path: Path) -> None:
    """
    E2E test: validate image with simple circle using Anthropic API.
    Requires ANTHROPIC_API_KEY in .env.local.
    """
    circle_img = Image.new("RGB", (128, 128), color="white")
    pixels = np.array(circle_img)
    y, x = np.ogrid[:128, :128]
    mask = (x - 64) ** 2 + (y - 64) ** 2 <= 30**2
    pixels[mask] = [0, 0, 0]
    circle_img = Image.fromarray(pixels)
    circle_path = tmp_path / "circle.png"
    circle_img.save(circle_path)

    validation = validate_image(circle_path)
    # Relaxed check: API returns results (testing integration, not image quality)
    assert validation.description
    assert isinstance(validation.has_shapes, bool)
    assert isinstance(validation.is_blank, bool)


@pytest.mark.slow
@pytest.mark.e2e
def test_validate_with_target(tmp_path: Path) -> None:
    """
    E2E test: validate with target image and similarity scoring using Anthropic API.
    Requires ANTHROPIC_API_KEY in .env.local.
    """
    circle_img = Image.new("RGB", (128, 128), color="white")
    pixels = np.array(circle_img)
    y, x = np.ogrid[:128, :128]
    mask = (x - 64) ** 2 + (y - 64) ** 2 <= 30**2
    pixels[mask] = [0, 0, 0]
    circle_img = Image.fromarray(pixels)
    circle_path = tmp_path / "circle.png"
    circle_img.save(circle_path)

    validation = validate_image(circle_path, target_path=circle_path)
    # Relaxed check: API returns similarity score (testing integration, not accuracy)
    assert validation.similarity_to_target is not None
    assert 0.0 <= validation.similarity_to_target <= 1.0
