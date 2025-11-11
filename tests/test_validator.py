from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from differentiable_pelican.validator import ImageValidation, validate_image


def has_api_key() -> bool:
    """
    Check if Anthropic API key is available.
    """
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env.local")
    return os.getenv("ANTHROPIC_API_KEY") is not None


def test_validation_schema_parsing():
    """
    Test that ImageValidation schema works correctly.
    """
    data = {
        "is_blank": False,
        "has_shapes": True,
        "shapes_recognizable": True,
        "resembles_pelican": False,
        "on_canvas": True,
        "description": "A simple black circle on white background",
        "issues": [],
    }
    validation = ImageValidation(**data)
    assert validation.is_blank is False
    assert validation.has_shapes is True
    assert validation.similarity_to_target is None


def test_validation_schema_with_similarity():
    """
    Test that ImageValidation handles optional similarity score.
    """
    data = {
        "is_blank": False,
        "has_shapes": True,
        "shapes_recognizable": True,
        "resembles_pelican": True,
        "on_canvas": True,
        "description": "A pelican-like shape",
        "issues": [],
        "similarity_to_target": 0.75,
    }
    validation = ImageValidation(**data)
    assert validation.similarity_to_target == 0.75


@pytest.fixture
def temp_images(tmp_path: Path):
    """
    Create temporary test images.
    """
    # Blank white image
    blank_img = Image.new("RGB", (128, 128), color="white")
    blank_path = tmp_path / "blank.png"
    blank_img.save(blank_path)

    # Image with a simple black circle
    circle_img = Image.new("RGB", (128, 128), color="white")
    pixels = np.array(circle_img)
    y, x = np.ogrid[:128, :128]
    mask = (x - 64) ** 2 + (y - 64) ** 2 <= 30**2
    pixels[mask] = [0, 0, 0]
    circle_img = Image.fromarray(pixels)
    circle_path = tmp_path / "circle.png"
    circle_img.save(circle_path)

    return {"blank": blank_path, "circle": circle_path}


@pytest.mark.skipif(not has_api_key(), reason="No API key")
@pytest.mark.slow
def test_validate_blank_image(temp_images):
    """
    Test validation of a blank image.
    """
    validation = validate_image(temp_images["blank"])
    assert validation.is_blank or not validation.has_shapes


@pytest.mark.skipif(not has_api_key(), reason="No API key")
@pytest.mark.slow
def test_validate_simple_circle(temp_images):
    """
    Test validation of an image with a simple circle.
    """
    validation = validate_image(temp_images["circle"])
    assert validation.has_shapes
    assert not validation.is_blank


@pytest.mark.skipif(not has_api_key(), reason="No API key")
@pytest.mark.slow
def test_validate_with_target(temp_images):
    """
    Test validation with a target image provided.
    """
    validation = validate_image(temp_images["circle"], target_path=temp_images["circle"])
    assert validation.similarity_to_target is not None
    assert validation.similarity_to_target > 0.8
