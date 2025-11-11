from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from differentiable_pelican.validator import validate_image


@pytest.mark.slow
def test_validate_blank_image(tmp_path: Path) -> None:
    """
    Test validation of a blank image. Will fail clearly if API key is not set.
    """
    blank_img = Image.new("RGB", (128, 128), color="white")
    blank_path = tmp_path / "blank.png"
    blank_img.save(blank_path)

    validation = validate_image(blank_path)
    assert validation.is_blank or not validation.has_shapes


@pytest.mark.slow
def test_validate_simple_circle(tmp_path: Path) -> None:
    """
    Test validation of an image with a simple circle.
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
    assert validation.has_shapes
    assert not validation.is_blank


@pytest.mark.slow
def test_validate_with_target(tmp_path: Path) -> None:
    """
    Test validation with a target image. Verifies similarity scoring.
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
    assert validation.similarity_to_target is not None
    assert validation.similarity_to_target > 0.8
