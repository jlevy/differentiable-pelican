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
from differentiable_pelican.svg_export import composite_stages_svg, shapes_to_svg

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


def create_random_shapes(
    count: int,
    device: torch.device,
    scale: float = 1.0,
    seed: int | None = None,
) -> tuple[list[Shape], list[str]]:
    """
    Create `count` random shapes of mixed types, cycling through circle/ellipse/triangle.

    Returns (shapes, names) in the same format as `create_initial_pelican()`.
    """
    if seed is not None:
        random.seed(seed)

    shapes: list[Shape] = []
    names: list[str] = []
    for i in range(count):
        shape_type = SHAPE_TYPES[i % len(SHAPE_TYPES)]
        shape = create_random_shape(shape_type, device, scale=scale)
        shapes.append(shape)
        names.append(f"{shape_type}_{i}")
    return shapes, names


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
        return Ellipse(
            cx=cx, cy=cy, rx=rx, ry=ry, rotation=rotation, device=device, intensity=intensity
        )

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


def _generate_greedy_gif(output_dir: Path, final_png: Path) -> None:
    """Generate a GIF showing progression through greedy refinement rounds."""
    try:
        import imageio.v3 as iio  # pyright: ignore[reportMissingImports]

        # Collect frame PNGs: initial + each accepted round + final best
        frame_paths: list[Path] = []

        # Initial optimization frame
        initial_frame = output_dir / "round_00_initial" / "optimized.png"
        if initial_frame.exists():
            frame_paths.append(initial_frame)

        # Accepted round frames (sorted by round number)
        round_dirs = sorted(output_dir.glob("round_*_accept_*"))
        for rd in round_dirs:
            frame = rd / "optimized.png"
            if frame.exists():
                frame_paths.append(frame)

        # Final best-params frame (may differ from last round if best was earlier)
        if final_png.exists():
            frame_paths.append(final_png)

        if len(frame_paths) < 2:
            return

        images = [iio.imread(str(f)) for f in frame_paths]  # pyright: ignore[reportUnknownMemberType]

        # Convert grayscale to RGB if needed (match all frames to same format)
        import numpy as np

        processed = []
        for img in images:
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] == 4:
                img = img[:, :, :3]
            processed.append(img)

        # 400ms per frame, hold final frame 1200ms
        durations = [400] * (len(processed) - 1) + [1200]

        gif_path = output_dir / "greedy_refinement.gif"
        iio.imwrite(str(gif_path), processed, duration=durations, loop=0)  # pyright: ignore[reportUnknownMemberType]
        console.print(f"  -> Saved greedy GIF: {gif_path} ({len(processed)} frames)")
    except Exception as e:
        console.print(f"  [yellow]Warning: Could not create greedy GIF: {e}[/yellow]")


def _generate_composite_svg(
    output_dir: Path,
    history: list[GreedyRoundRecord],
    max_stages: int = 8,
) -> None:
    """Generate a composite SVG showing evenly-spaced pipeline stages."""
    # Collect all available stage SVGs
    stages: list[tuple[Path, str, str]] = []

    # Initial optimization
    initial_svg = output_dir / "round_00_initial" / "optimized.svg"
    if initial_svg.exists():
        stages.append((initial_svg, "Optimized", "9 shapes"))

    # Accepted rounds
    accepted = [r for r in history if r.get("accepted", False)]
    for rec in accepted:
        rnum = rec.get("round", 0)
        stype = rec.get("shape_type", "unknown")
        nshapes = rec.get("num_shapes", 0)
        svg = output_dir / f"round_{rnum:02d}_accept_{stype}" / "optimized.svg"
        if svg.exists():
            stages.append((svg, f"Round {rnum}", f"{nshapes} shapes"))

    if len(stages) < 2:
        return

    # Pick evenly-spaced subset if too many stages
    if len(stages) > max_stages:
        # Always include first and last; evenly sample the rest
        indices = [0]
        inner_count = max_stages - 2
        for i in range(inner_count):
            idx = int((i + 1) * (len(stages) - 1) / (inner_count + 1))
            indices.append(idx)
        indices.append(len(stages) - 1)
        stages = [stages[i] for i in indices]

    # Relabel the last stage as "Final"
    path, _, sublabel = stages[-1]
    stages[-1] = (path, "Final", sublabel)

    composite_path = output_dir / "pipeline_stages.svg"
    try:
        composite_stages_svg(stages, composite_path)
        console.print(f"  -> Saved composite SVG: {composite_path} ({len(stages)} stages)")
    except Exception as e:
        console.print(f"  [yellow]Warning: Could not create composite SVG: {e}[/yellow]")


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

        console.print(
            f"  Phase A: Settling new shape ({settle_steps} steps, freeze={freeze_existing})"
        )
        optimize(shapes, target, resolution, settle_steps, lr=0.02)

        if freeze_existing:
            _unfreeze_shapes(shapes[:-1])

        loss_after_settle = _evaluate_loss(shapes, target, resolution, device)
        console.print(f"  After settle: {loss_after_settle:.6f}")

        # Phase B: Re-optimize all shapes together
        if reoptimize_steps > 0:
            console.print(
                f"  Phase B: Re-optimizing all {len(shapes)} shapes ({reoptimize_steps} steps)"
            )
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
            save_render(
                shapes, resolution, resolution, tau, device, str(round_dir / "optimized.png")
            )
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

    # Generate GIF from round frames
    _generate_greedy_gif(output_dir, final_png)

    # Generate composite SVG showing pipeline stages
    _generate_composite_svg(output_dir, history)

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


## Tests


def test_create_random_shapes_count():
    device = torch.device("cpu")
    shapes, names = create_random_shapes(5, device, seed=42)
    assert len(shapes) == 5
    assert len(names) == 5


def test_create_random_shapes_types_cycle():
    device = torch.device("cpu")
    shapes, _names = create_random_shapes(6, device, seed=42)
    # Cycles through circle, ellipse, triangle
    assert isinstance(shapes[0], Circle)
    assert isinstance(shapes[1], Ellipse)
    assert isinstance(shapes[2], Triangle)
    assert isinstance(shapes[3], Circle)
    assert isinstance(shapes[4], Ellipse)
    assert isinstance(shapes[5], Triangle)


def test_create_random_shapes_reproducible():
    device = torch.device("cpu")
    shapes_a, names_a = create_random_shapes(3, device, seed=99)
    shapes_b, names_b = create_random_shapes(3, device, seed=99)
    assert names_a == names_b
    for a, b in zip(shapes_a, shapes_b):
        pa = a.get_params()
        pb = b.get_params()
        if isinstance(a, Circle):
            assert float(pa.cx) == float(pb.cx)
        elif isinstance(a, Ellipse):
            assert float(pa.cx) == float(pb.cx)


def test_create_random_shapes_empty():
    device = torch.device("cpu")
    shapes, names = create_random_shapes(0, device)
    assert shapes == []
    assert names == []
