from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image

from differentiable_pelican.geometry import Shape
from differentiable_pelican.loss import total_loss
from differentiable_pelican.renderer import make_grid, render


class OptimizationMetrics(TypedDict):
    """Metrics returned from optimization."""

    loss_history: list[dict[str, float]]
    final_loss: float
    steps_completed: int
    resolution: int


# Type for progress callback: (step, total_steps, loss_breakdown) -> None
ProgressCallback = Callable[[int, int, dict[str, float]], None]


def load_target_image(path: Path, resolution: int, device: torch.device) -> torch.Tensor:
    """
    Load and preprocess target image.
    """
    img = Image.open(path).convert("L")  # Grayscale
    img = img.resize((resolution, resolution))
    # Convert to tensor, normalize to [0, 1]
    tensor = torch.from_numpy(np.array(img)).float() / 255.0
    return tensor.to(device)


def anneal_tau(step: int, total_steps: int, tau_start: float, tau_end: float) -> float:
    """
    Exponentially anneal softness parameter.
    """
    if total_steps == 0:
        return tau_end
    progress = step / total_steps
    return tau_start * (tau_end / tau_start) ** progress


def optimize(
    shapes: list[Shape],
    target: torch.Tensor,
    resolution: int,
    steps: int,
    lr: float = 0.02,
    tau_start: float | None = None,
    tau_end: float | None = None,
    save_every: int | None = None,
    output_dir: Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> OptimizationMetrics:
    """
    Optimize shapes to match target image using gradient descent.

    Args:
        shapes: List of Shape objects to optimize
        target: Target image tensor [H, W] in [0, 1]
        resolution: Image resolution (H = W = resolution)
        steps: Number of optimization steps
        lr: Learning rate
        tau_start: Starting softness (default: 1.0 pixel)
        tau_end: Ending softness (default: 0.2 pixels)
        save_every: Save intermediate frames every N steps (None = don't save)
        output_dir: Directory to save frames (required if save_every is set)
        progress_callback: Optional callback for progress updates

    Returns:
        Metrics dictionary with loss history and final parameters
    """
    device = shapes[0].device

    # Default tau values in normalized coordinates
    if tau_start is None:
        tau_start = 1.0 / resolution
    if tau_end is None:
        tau_end = 0.2 / resolution

    # Collect parameters from all shapes
    params = []
    for shape in shapes:
        params.extend(list(shape.parameters()))

    # Create optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(params, lr=lr)

    # Learning rate scheduler: warm up then cosine decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr / 20
    )

    # Precompute grid
    grid = make_grid(resolution, resolution, device)

    # Metrics tracking
    loss_history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_params: dict[int, dict[str, torch.Tensor]] | None = None

    # Setup output directory if saving frames
    frames_dir: Path | None = None
    if save_every and output_dir:
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for step in range(steps):
        # Anneal tau
        tau = anneal_tau(step, steps, tau_start, tau_end)

        # Forward pass
        rendered = render(shapes, resolution, resolution, tau, device, grid=grid)

        # Compute loss
        loss, breakdown = total_loss(rendered, target, shapes)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        # Update
        optimizer.step()
        scheduler.step()

        # Track metrics
        loss_history.append(breakdown)

        # Save best parameters
        if breakdown["total"] < best_loss:
            best_loss = breakdown["total"]
            best_params = {
                i: {k: v.detach().clone() for k, v in shape.state_dict().items()}
                for i, shape in enumerate(shapes)
            }

        # Progress callback
        if progress_callback is not None:
            progress_callback(step, steps, breakdown)

        # Save intermediate frames
        if (
            save_every
            and output_dir
            and frames_dir
            and (step % save_every == 0 or step == steps - 1)
        ):
            with torch.no_grad():
                rendered_np = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(rendered_np, mode="L")
                img.save(frames_dir / f"frame_{step:04d}.png")

        # Check for NaN
        if torch.isnan(loss):
            print(f"Warning: NaN loss at step {step}, stopping")
            break

    # Restore best parameters
    if best_params:
        for i, shape in enumerate(shapes):
            shape.load_state_dict(best_params[i])

    metrics: OptimizationMetrics = {
        "loss_history": loss_history,
        "final_loss": best_loss,
        "steps_completed": step + 1,
        "resolution": resolution,
    }

    return metrics


## Tests


def test_anneal_tau():
    """
    Test that tau anneals correctly.
    """
    tau_0 = anneal_tau(0, 100, 1.0, 0.2)
    tau_mid = anneal_tau(50, 100, 1.0, 0.2)
    tau_end = anneal_tau(100, 100, 1.0, 0.2)

    assert abs(tau_0 - 1.0) < 1e-6
    assert abs(tau_end - 0.2) < 1e-6
    assert 0.2 < tau_mid < 1.0


def test_anneal_tau_zero_steps():
    """
    Test that tau handles zero steps gracefully.
    """
    tau = anneal_tau(0, 0, 1.0, 0.2)
    assert abs(tau - 0.2) < 1e-6


def test_optimize_reduces_loss_simple_case():
    """
    Test that optimization reduces loss for a simple case.
    """
    from differentiable_pelican.geometry import Circle

    device = torch.device("cpu")
    resolution = 32

    # Create target: circle at center
    target_circle = Circle(cx=0.5, cy=0.5, radius=0.2, device=device)
    tau = 1.0 / resolution
    target = render([target_circle], resolution, resolution, tau, device).detach()

    # Create initial: slightly offset circle
    init_circle = Circle(cx=0.45, cy=0.45, radius=0.18, device=device)

    # Optimize
    metrics = optimize([init_circle], target, resolution, steps=20, lr=0.05)

    # Check that loss decreased
    initial_loss = metrics["loss_history"][0]["total"]
    final_loss = metrics["final_loss"]

    assert final_loss < initial_loss


def test_optimize_with_progress_callback():
    """
    Test that progress callback is invoked during optimization.
    """
    from differentiable_pelican.geometry import Circle

    device = torch.device("cpu")
    resolution = 32

    target_circle = Circle(cx=0.5, cy=0.5, radius=0.2, device=device)
    tau = 1.0 / resolution
    target = render([target_circle], resolution, resolution, tau, device).detach()

    init_circle = Circle(cx=0.45, cy=0.45, radius=0.18, device=device)

    callback_calls: list[int] = []

    def on_progress(step: int, total: int, breakdown: dict[str, float]) -> None:
        callback_calls.append(step)

    optimize(
        [init_circle], target, resolution, steps=10, lr=0.05,
        progress_callback=on_progress,
    )

    assert len(callback_calls) == 10
    assert callback_calls == list(range(10))
