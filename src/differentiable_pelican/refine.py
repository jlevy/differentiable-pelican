from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import torch
from rich.console import Console

from differentiable_pelican.geometry import Shape
from differentiable_pelican.llm.architect import architect_edits
from differentiable_pelican.llm.edit_parser import parse_edits
from differentiable_pelican.llm.judge import judge_svg
from differentiable_pelican.optimizer import OptimizationMetrics, load_target_image, optimize
from differentiable_pelican.renderer import save_render
from differentiable_pelican.svg_export import shapes_to_svg

console = Console()


class RefinementRoundRecord(TypedDict, total=False):
    """Record from a single refinement round. Fields vary by outcome."""

    round: int
    metrics: OptimizationMetrics
    feedback: dict[str, bool | float | str | list[str]]
    architect: dict[str, str | list[dict[str, str]]]
    num_shapes: int
    converged: bool
    error: str
    rolled_back: bool


class RefinementResult(TypedDict):
    """Results from multi-round refinement loop."""

    rounds_completed: int
    history: list[RefinementRoundRecord]
    final_shapes: int


def _save_shapes_state(shapes: list[Shape]) -> list[dict[str, torch.Tensor]]:
    """Save a deep copy of all shape parameters for rollback."""
    return [{k: v.detach().clone() for k, v in shape.state_dict().items()} for shape in shapes]


def _restore_shapes_state(shapes: list[Shape], state: list[dict[str, torch.Tensor]]) -> None:
    """Restore shape parameters from a saved state."""
    for shape, saved in zip(shapes, state, strict=True):
        shape.load_state_dict(saved)


def refinement_loop(
    initial_shapes: list[Shape],
    shape_names: list[str],
    target_path: Path,
    resolution: int,
    output_dir: Path,
    max_rounds: int = 5,
    steps_per_round: int = 500,
    convergence_threshold: float = 0.01,
    max_consecutive_failures: int = 2,
) -> RefinementResult:
    """
    Multi-round refinement loop with LLM feedback and rollback.

    Each round:
    1. Optimize shapes to match target (gradient descent)
    2. Judge evaluates result (LLM)
    3. Architect proposes structural edits (LLM)
    4. Apply edits to shapes
    5. If quality degrades, rollback to previous best

    Args:
        initial_shapes: Starting geometry
        shape_names: Names for each shape
        target_path: Target image path
        resolution: Image resolution
        output_dir: Output directory
        max_rounds: Maximum refinement rounds
        steps_per_round: Optimization steps per round
        convergence_threshold: Stop if improvement < threshold
        max_consecutive_failures: Rollback limit before stopping

    Returns:
        Refinement metrics and history
    """
    device = initial_shapes[0].device
    target = load_target_image(target_path, resolution, device)

    shapes = initial_shapes
    names = shape_names
    round_history: list[RefinementRoundRecord] = []
    previous_loss = float("inf")
    best_loss = float("inf")
    best_shapes_state: list[dict[str, torch.Tensor]] | None = None
    best_names: list[str] | None = None
    consecutive_failures = 0

    for round_num in range(max_rounds):
        round_dir = output_dir / f"round_{round_num:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[bold cyan]=== Round {round_num + 1}/{max_rounds} ===[/bold cyan]")
        console.print(f"  Shapes: {len(shapes)} ({', '.join(names)})")

        # Save state before this round for potential rollback
        pre_round_state = _save_shapes_state(shapes)
        pre_round_names = list(names)

        # Optimize current geometry
        console.print("  [cyan]Optimizing...[/cyan]")
        metrics = optimize(
            shapes,
            target,
            resolution,
            steps_per_round,
            lr=0.02,
            save_every=None,
            output_dir=None,
        )

        current_loss = metrics["final_loss"]
        console.print(f"  Loss: {current_loss:.6f} (prev: {previous_loss:.6f})")

        # Check if this round improved over previous
        if current_loss > previous_loss * 1.1 and best_shapes_state is not None:
            # Quality degraded significantly - rollback
            console.print("  [yellow]Quality degraded, rolling back...[/yellow]")
            _restore_shapes_state(shapes, pre_round_state)
            names = pre_round_names
            consecutive_failures += 1

            round_history.append(
                {
                    "round": round_num,
                    "metrics": metrics,
                    "rolled_back": True,
                    "num_shapes": len(shapes),
                }
            )

            if consecutive_failures >= max_consecutive_failures:
                console.print(
                    f"  [yellow]Stopping: {consecutive_failures} consecutive failures[/yellow]"
                )
                break
            continue
        else:
            consecutive_failures = 0

        # Update best state
        if current_loss < best_loss:
            best_loss = current_loss
            best_shapes_state = _save_shapes_state(shapes)
            best_names = list(names)

        # Save outputs
        tau = 0.5 / resolution
        png_path = round_dir / "optimized.png"
        svg_path = round_dir / "optimized.svg"
        save_render(shapes, resolution, resolution, tau, device, str(png_path))
        shapes_to_svg(shapes, resolution, resolution, svg_path)

        # Judge evaluation
        console.print("  [cyan]Evaluating with judge...[/cyan]")
        try:
            feedback = judge_svg(svg_path, png_path, target_path, metrics)

            # Save feedback
            feedback_path = round_dir / "feedback.json"
            with feedback_path.open("w") as f:
                json.dump(feedback.model_dump(), f, indent=2)

            console.print(f"  Quality: {feedback.overall_quality:.2f}")
            console.print(f"  Similarity: {feedback.similarity_to_target:.2f}")
            console.print(f"  Pelican: {'Yes' if feedback.resembles_pelican else 'No'}")

            # Check convergence
            loss_improvement = previous_loss - current_loss
            if (
                feedback.ready_for_refinement
                and abs(loss_improvement) < convergence_threshold
                and feedback.overall_quality > 0.7
            ):
                console.print("  [green]Converged! Quality is acceptable.[/green]")
                round_history.append(
                    {
                        "round": round_num,
                        "metrics": metrics,
                        "feedback": feedback.model_dump(),
                        "converged": True,
                        "num_shapes": len(shapes),
                    }
                )
                break

            # Architect proposes edits
            console.print("  [cyan]Generating edits with architect...[/cyan]")
            arch_response = architect_edits(feedback)

            # Save architect response
            arch_path = round_dir / "architect.json"
            with arch_path.open("w") as f:
                json.dump(arch_response.model_dump(), f, indent=2)

            console.print(f"  Proposed {len(arch_response.actions)} edits")
            for action in arch_response.actions:
                console.print(f"    - {action.type}: {action.shape}")

            # Apply edits
            console.print("  [cyan]Applying edits...[/cyan]")
            try:
                shapes, names = parse_edits(arch_response.actions, shapes, names)
                console.print(f"  -> Now {len(shapes)} shapes: {', '.join(names)}")
            except Exception as e:
                console.print(f"  [yellow]Edit application failed: {e}[/yellow]")

            # Record round
            round_history.append(
                {
                    "round": round_num,
                    "metrics": metrics,
                    "feedback": feedback.model_dump(),
                    "architect": arch_response.model_dump(),
                    "num_shapes": len(shapes),
                }
            )

            previous_loss = current_loss

        except Exception as e:
            console.print(f"  [red]Round {round_num} failed: {e}[/red]")
            round_history.append(
                {
                    "round": round_num,
                    "metrics": metrics,
                    "error": str(e),
                }
            )
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break

    # Restore best shapes if we have them and shape counts match
    if best_shapes_state is not None and best_names is not None:
        if len(shapes) == len(best_shapes_state):
            try:
                _restore_shapes_state(shapes, best_shapes_state)
                names = best_names
            except Exception as e:
                print(f"Warning: Could not restore best shapes: {e}")
        else:
            # Shape count changed (edits added/removed shapes). The current shapes
            # may not match the best state's topology, so we keep current shapes
            # but log the discrepancy.
            print(
                f"Note: Shape count changed ({len(best_shapes_state)} -> {len(shapes)}), "
                f"using current shapes for final output."
            )

    # Save final outputs
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    tau = 0.5 / resolution
    final_png = final_dir / "pelican_final.png"
    final_svg = final_dir / "pelican_final.svg"
    save_render(shapes, resolution, resolution, tau, device, str(final_png))
    shapes_to_svg(shapes, resolution, resolution, final_svg)

    # Save refinement history
    history_path = output_dir / "refinement_history.json"
    with history_path.open("w") as f:
        json.dump(
            {
                "rounds_completed": len(round_history),
                "max_rounds": max_rounds,
                "best_loss": best_loss,
                "history": round_history,
            },
            f,
            indent=2,
        )

    console.print(f"\n[green]Refinement complete! {len(round_history)} rounds[/green]")
    console.print(f"Best loss: {best_loss:.6f}")
    console.print(f"Final outputs: {final_dir}")

    return {
        "rounds_completed": len(round_history),
        "history": round_history,
        "final_shapes": len(shapes),
    }
