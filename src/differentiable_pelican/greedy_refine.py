from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import TypedDict

import torch
from rich.console import Console

from differentiable_pelican.geometry import Circle, Ellipse, Shape, Triangle
from differentiable_pelican.loss import total_loss
from differentiable_pelican.optimizer import load_target_image, optimize
from differentiable_pelican.renderer import make_grid, render, save_render
from differentiable_pelican.svg_export import shapes_to_svg

console = Console()

SHAPE_TYPES = ["circle", "ellipse", "triangle"]


class GreedyRoundRecord(TypedDict, total=False):
    """Record from a single greedy round."""

    round: int
    action: str  # "add", "reject"
    shape_type: str
    shape_name: str
    loss_before: float
    loss_after: float
    num_shapes: int
    accepted: bool


class GreedyResult(TypedDict):
    """Results from greedy refinement."""

    rounds_completed: int
    shapes_added: int
    initial_loss: float
    final_loss: float
    history: list[GreedyRoundRecord]
    final_shapes: int


def create_random_shape(
    shape_type: str,
    device: torch.device,
    scale: float = 1.0,
) -> Shape:
    """
    Create a random shape with random position, size, and intensity.

    The shape is placed randomly within the image bounds. Gradient descent
    will pull it to the optimal position during optimization.

    Args:
        shape_type: "circle", "ellipse", or "triangle"
        device: torch device
        scale: Size multiplier (1.0 = default size, 0.5 = half, 2.0 = double)
    """
    cx = random.uniform(0.15, 0.85)
    cy = random.uniform(0.15, 0.85)
    # Bias toward darker values since pelican is dark on white background
    intensity = random.uniform(0.05, 0.55)

    base_size = 0.08 * scale

    if shape_type == "circle":
        radius = base_size * random.uniform(0.5, 1.5)
        return Circle(cx=cx, cy=cy, radius=radius, device=device, intensity=intensity)

    elif shape_type == "ellipse":
        rx = base_size * random.uniform(0.5, 2.0)
        ry = base_size * random.uniform(0.3, 1.5)
        rotation = random.uniform(-1.5, 1.5)
        return Ellipse(cx=cx, cy=cy, rx=rx, ry=ry, rotation=rotation, device=device, intensity=intensity)

    elif shape_type == "triangle":
        spread = base_size * 2
        v0 = (
            max(0.05, min(0.95, cx + random.uniform(-spread, spread))),
            max(0.05, min(0.95, cy + random.uniform(-spread, spread))),
        )
        v1 = (
            max(0.05, min(0.95, cx + random.uniform(-spread, spread))),
            max(0.05, min(0.95, cy + random.uniform(-spread, spread))),
        )
        v2 = (
            max(0.05, min(0.95, cx + random.uniform(-spread, spread))),
            max(0.05, min(0.95, cy + random.uniform(-spread, spread))),
        )
        return Triangle(v0=v0, v1=v1, v2=v2, device=device, intensity=intensity)

    else:
        raise ValueError(f"Unknown shape type: {shape_type}")


def _evaluate_loss(
    shapes: list[Shape],
    target: torch.Tensor,
    resolution: int,
    device: torch.device,
) -> float:
    """Evaluate loss without gradient computation."""
    with torch.no_grad():
        grid = make_grid(resolution, resolution, device)
        tau = 0.5 / resolution
        rendered = render(shapes, resolution, resolution, tau, device, grid=grid)
        _, breakdown = total_loss(rendered, target, shapes)
        return breakdown["total"]


def _freeze_shapes(shapes: list[Shape]) -> None:
    """Set requires_grad=False on all parameters of the given shapes."""
    for shape in shapes:
        for p in shape.parameters():
            p.requires_grad_(False)


def _unfreeze_shapes(shapes: list[Shape]) -> None:
    """Set requires_grad=True on all parameters of the given shapes."""
    for shape in shapes:
        for p in shape.parameters():
            p.requires_grad_(True)


def greedy_refinement_loop(
    initial_shapes: list[Shape],
    shape_names: list[str],
    target_path: Path,
    resolution: int,
    output_dir: Path,
    max_shapes: int = 20,
    initial_steps: int = 500,
    settle_steps: int = 100,
    reoptimize_steps: int = 200,
    freeze_existing: bool = True,
    max_consecutive_failures: int = 5,
    scale: float = 1.0,
    seed: int = 42,
) -> GreedyResult:
    """
    Greedy shape-dropping refinement loop.

    Adds one shape at a time, lets gradient descent find its optimal placement,
    and keeps it only if it improves the loss. This builds up complexity
    gradually and rejects shapes that don't help.

    Args:
        initial_shapes: Starting geometry
        shape_names: Names for each shape
        target_path: Target image path
        resolution: Image resolution
        output_dir: Output directory
        max_shapes: Maximum total shapes (budget)
        initial_steps: Optimization steps for initial geometry
        settle_steps: Steps to optimize the new shape (frozen or unfrozen existing)
        reoptimize_steps: Steps for full re-optimization after accepting a shape
        freeze_existing: If True, only the new shape is optimized during settle phase
        max_consecutive_failures: Stop adding after this many consecutive rejections
        scale: Size multiplier for new shapes (1.0 = default)
        seed: Random seed for reproducibility

    Returns:
        Refinement results with history
    """
    random.seed(seed)
    device = initial_shapes[0].device
    target = load_target_image(target_path, resolution, device)

    shapes = list(initial_shapes)
    names = list(shape_names)
    history: list[GreedyRoundRecord] = []

    # Phase 1: Optimize initial geometry
    console.print(f"\n[bold cyan]Phase 1: Initial optimization ({initial_steps} steps)[/bold cyan]")
    console.print(f"  Shapes: {len(shapes)} ({', '.join(names)})")

    metrics = optimize(shapes, target, resolution, initial_steps, lr=0.02)
    initial_loss = metrics["final_loss"]
    console.print(f"  Initial loss: {initial_loss:.6f}")

    # Save initial state
    round_dir = output_dir / "round_00_initial"
    round_dir.mkdir(parents=True, exist_ok=True)
    tau = 0.5 / resolution
    save_render(shapes, resolution, resolution, tau, device, str(round_dir / "optimized.png"))
    shapes_to_svg(shapes, resolution, resolution, round_dir / "optimized.svg")

    # Phase 2: Greedy shape addition
    console.print(f"\n[bold cyan]Phase 2: Greedy shape addition (budget: {max_shapes})[/bold cyan]")
    if freeze_existing:
        console.print("  Mode: freeze existing shapes during settle phase")

    best_loss = initial_loss
    best_shapes = copy.deepcopy(shapes)
    best_names = list(names)
    consecutive_failures = 0
    shapes_added = 0
    round_num = 0
    type_idx = 0  # Cycle through shape types

    while len(shapes) < max_shapes and consecutive_failures < max_consecutive_failures:
        round_num += 1
        shape_type = SHAPE_TYPES[type_idx % len(SHAPE_TYPES)]
        type_idx += 1

        console.print(f"\n  [cyan]--- Round {round_num} ---[/cyan]")

        # Evaluate current loss
        loss_before = _evaluate_loss(shapes, target, resolution, device)

        # Create a random candidate shape
        candidate = create_random_shape(shape_type, device, scale=scale)
        candidate_name = f"{shape_type}_{round_num}"

        # Save state for rollback
        pre_state = copy.deepcopy(shapes)
        pre_names = list(names)

        # Add candidate
        shapes.append(candidate)
        names.append(candidate_name)
        console.print(f"  Trying {shape_type} (now {len(shapes)} shapes)")

        # Two-phase trial:
        # Phase A: Settle the new shape (freeze existing, optimize new shape only)
        # Phase B: Re-optimize all shapes together
        # Then decide: keep or discard based on overall result.

        # Phase A: Settle
        if freeze_existing:
            _freeze_shapes(shapes[:-1])

        console.print(f"  Phase A: Settling new shape ({settle_steps} steps, freeze={freeze_existing})")
        optimize(shapes, target, resolution, settle_steps, lr=0.02)

        if freeze_existing:
            _unfreeze_shapes(shapes[:-1])

        loss_after_settle = _evaluate_loss(shapes, target, resolution, device)
        console.print(f"  After settle: {loss_after_settle:.6f}")

        # Phase B: Re-optimize all shapes together
        if reoptimize_steps > 0:
            console.print(f"  Phase B: Re-optimizing all {len(shapes)} shapes ({reoptimize_steps} steps)")
            metrics = optimize(shapes, target, resolution, reoptimize_steps, lr=0.02)
            loss_after = metrics["final_loss"]
            console.print(f"  After re-optimize: {loss_after:.6f}")
        else:
            loss_after = loss_after_settle

        # Decision: keep or discard based on overall improvement
        improvement = loss_before - loss_after

        if loss_after < loss_before:
            # Shape helped! Accept it.
            consecutive_failures = 0
            shapes_added += 1
            console.print(
                f"  [green]Accepted[/green] {candidate_name}: "
                f"{loss_before:.6f} -> {loss_after:.6f} (improved {improvement:.6f})"
            )

            # Update best
            if loss_after < best_loss:
                best_loss = loss_after
                best_shapes = copy.deepcopy(shapes)
                best_names = list(names)

            # Save round output
            round_dir = output_dir / f"round_{round_num:02d}_accept_{shape_type}"
            round_dir.mkdir(parents=True, exist_ok=True)
            save_render(shapes, resolution, resolution, tau, device, str(round_dir / "optimized.png"))
            shapes_to_svg(shapes, resolution, resolution, round_dir / "optimized.svg")

            history.append(
                {
                    "round": round_num,
                    "action": "add",
                    "shape_type": shape_type,
                    "shape_name": candidate_name,
                    "loss_before": loss_before,
                    "loss_after": loss_after,
                    "num_shapes": len(shapes),
                    "accepted": True,
                }
            )

        else:
            # Shape didn't help after both phases. Discard and restore.
            consecutive_failures += 1
            shapes = pre_state
            names = pre_names
            console.print(
                f"  [yellow]Rejected[/yellow] {candidate_name}: "
                f"{loss_before:.6f} -> {loss_after:.6f} "
                f"(failures: {consecutive_failures}/{max_consecutive_failures})"
            )

            history.append(
                {
                    "round": round_num,
                    "action": "reject",
                    "shape_type": shape_type,
                    "shape_name": candidate_name,
                    "loss_before": loss_before,
                    "loss_after": loss_after,
                    "num_shapes": len(shapes),
                    "accepted": False,
                }
            )

    if consecutive_failures >= max_consecutive_failures:
        console.print(
            f"\n  [yellow]Stopping: {consecutive_failures} consecutive rejections[/yellow]"
        )

    # Restore best shapes
    shapes = best_shapes
    names = best_names

    # Save final outputs
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_png = final_dir / "pelican_final.png"
    final_svg = final_dir / "pelican_final.svg"
    save_render(shapes, resolution, resolution, tau, device, str(final_png))
    shapes_to_svg(shapes, resolution, resolution, final_svg)

    # Save history
    history_path = output_dir / "greedy_history.json"
    with history_path.open("w") as f:
        json.dump(
            {
                "initial_loss": initial_loss,
                "final_loss": best_loss,
                "shapes_added": shapes_added,
                "total_rounds": round_num,
                "final_shape_count": len(shapes),
                "freeze_existing": freeze_existing,
                "scale": scale,
                "history": history,
            },
            f,
            indent=2,
        )

    console.print("\n[green]Greedy refinement complete![/green]")
    console.print(f"  Added {shapes_added} shapes ({len(shapes)} total)")
    console.print(f"  Loss: {initial_loss:.6f} -> {best_loss:.6f}")
    console.print(f"  Final outputs: {final_dir}")

    return {
        "rounds_completed": round_num,
        "shapes_added": shapes_added,
        "initial_loss": initial_loss,
        "final_loss": best_loss,
        "history": history,
        "final_shapes": len(shapes),
    }
