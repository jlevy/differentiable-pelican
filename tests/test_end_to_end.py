from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from differentiable_pelican.geometry import create_initial_pelican
from differentiable_pelican.optimizer import load_target_image, optimize
from differentiable_pelican.renderer import save_render
from differentiable_pelican.svg_export import shapes_to_svg
from differentiable_pelican.validator import validate_image


@pytest.mark.slow
def test_full_optimization_pipeline(tmp_path: Path) -> None:
    """
    End-to-end test: optimize shapes to match target, verify loss reduction and outputs.
    """
    device = torch.device("cpu")
    resolution = 64
    steps = 20

    shapes = create_initial_pelican(device)
    target_path = Path("images/pelican-drawing-1.jpg")

    if not target_path.exists():
        pytest.skip("Target image not found")

    target = load_target_image(target_path, resolution, device)

    output_dir = tmp_path / "optimization"
    output_dir.mkdir()

    metrics = optimize(
        shapes,
        target,
        resolution,
        steps,
        lr=0.02,
        save_every=None,
        output_dir=output_dir,
    )

    # Verify optimization reduced loss
    assert len(metrics["loss_history"]) == steps
    initial_loss = metrics["loss_history"][0]["total"]
    final_loss = metrics["final_loss"]
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} -> {final_loss}"

    # Save and verify outputs
    tau = 0.5 / resolution
    png_path = output_dir / "pelican_optimized.png"
    save_render(shapes, resolution, resolution, tau, device, str(png_path))
    assert png_path.exists()

    svg_path = output_dir / "pelican_optimized.svg"
    shapes_to_svg(shapes, resolution, resolution, svg_path)
    assert svg_path.exists()

    svg_content = svg_path.read_text()
    assert "<svg" in svg_content
    assert "</svg>" in svg_content
    assert "circle" in svg_content or "ellipse" in svg_content or "polygon" in svg_content


@pytest.mark.slow
def test_pelican_optimization_with_validation(tmp_path: Path) -> None:
    """
    Full pipeline with LLM validation. Verifies optimized output resembles a pelican.
    Will fail clearly if ANTHROPIC_API_KEY is not set.
    """
    device = torch.device("cpu")
    resolution = 128
    steps = 50

    shapes = create_initial_pelican(device)
    target_path = Path("images/pelican-drawing-1.jpg")

    if not target_path.exists():
        pytest.skip("Target image not found")

    target = load_target_image(target_path, resolution, device)

    output_dir = tmp_path / "validation"
    output_dir.mkdir()

    # Optimize
    metrics = optimize(
        shapes,
        target,
        resolution,
        steps,
        lr=0.02,
        save_every=None,
        output_dir=output_dir,
    )

    # Verify significant loss reduction
    loss_reduction = (metrics["loss_history"][0]["total"] - metrics["final_loss"]) / metrics[
        "loss_history"
    ][0]["total"]
    assert loss_reduction > 0.1, f"Should reduce loss by at least 10%, got {loss_reduction*100:.1f}%"

    # Save PNG
    tau = 0.5 / resolution
    png_path = output_dir / "pelican_optimized.png"
    save_render(shapes, resolution, resolution, tau, device, str(png_path))

    # Validate with LLM
    validation = validate_image(png_path, target_path=target_path)

    # Check validation results
    assert not validation.is_blank, "Rendered image should not be blank"
    assert validation.has_shapes, "Rendered image should contain visible shapes"
    assert validation.on_canvas, "Shapes should be within canvas bounds"

    # Should resemble a pelican or at least have recognizable shapes
    assert (
        validation.shapes_recognizable or validation.resembles_pelican
    ), "Output should have recognizable shapes or resemble a pelican"

    # Similarity should be reasonable
    if validation.similarity_to_target is not None:
        assert (
            validation.similarity_to_target > 0.1
        ), f"Should have some similarity to target, got {validation.similarity_to_target}"


@pytest.mark.slow
def test_higher_resolution_optimization(tmp_path: Path) -> None:
    """
    Test at 256x256 resolution. Verifies performance and frame saving.
    """
    device = torch.device("cpu")
    resolution = 256
    steps = 30

    shapes = create_initial_pelican(device)
    target_path = Path("images/pelican-drawing-1.jpg")

    if not target_path.exists():
        pytest.skip("Target image not found")

    target = load_target_image(target_path, resolution, device)

    output_dir = tmp_path / "high_res"
    output_dir.mkdir()

    metrics = optimize(
        shapes,
        target,
        resolution,
        steps,
        lr=0.02,
        save_every=10,
        output_dir=output_dir,
    )

    # Verify optimization completed
    assert metrics["steps_completed"] == steps
    assert len(metrics["loss_history"]) == steps

    # Verify frames were saved
    frames_dir = output_dir / "frames"
    assert frames_dir.exists()
    frame_files = list(frames_dir.glob("frame_*.png"))
    expected_frames = (steps // 10) + 1
    assert len(frame_files) == expected_frames

    # Save and verify final output
    tau = 0.5 / resolution
    png_path = output_dir / "pelican_optimized.png"
    save_render(shapes, resolution, resolution, tau, device, str(png_path))
    assert png_path.exists()

    img = Image.open(png_path)
    assert img.size == (resolution, resolution)

    # Verify meaningful loss reduction
    initial_loss = metrics["loss_history"][0]["total"]
    final_loss = metrics["final_loss"]
    loss_reduction_pct = (initial_loss - final_loss) / initial_loss * 100
    assert (
        loss_reduction_pct > 5
    ), f"Should reduce loss by at least 5% at high res, got {loss_reduction_pct:.1f}%"
