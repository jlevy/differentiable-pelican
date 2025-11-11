from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from differentiable_pelican.geometry import Shape
from differentiable_pelican.llm.architect import architect_edits
from differentiable_pelican.llm.edit_parser import parse_edits
from differentiable_pelican.llm.judge import judge_svg
from differentiable_pelican.optimizer import OptimizationMetrics, load_target_image, optimize
from differentiable_pelican.renderer import save_render
from differentiable_pelican.svg_export import shapes_to_svg


class RefinementRoundRecord(TypedDict, total=False):
    """Record from a single refinement round. Fields vary by outcome."""

    round: int
    metrics: OptimizationMetrics
    feedback: dict[str, bool | float | str | list[str]]
    architect: dict[str, str | list]
    num_shapes: int
    converged: bool
    error: str


class RefinementResult(TypedDict):
    """Results from multi-round refinement loop."""

    rounds_completed: int
    history: list[RefinementRoundRecord]
    final_shapes: int


def refinement_loop(
    initial_shapes: list[Shape],
    shape_names: list[str],
    target_path: Path,
    resolution: int,
    output_dir: Path,
    max_rounds: int = 5,
    steps_per_round: int = 500,
    convergence_threshold: float = 0.01,
) -> RefinementResult:
    """
    Multi-round refinement loop with LLM feedback.

    Args:
        initial_shapes: Starting geometry
        shape_names: Names for each shape
        target_path: Target image path
        resolution: Image resolution
        output_dir: Output directory
        max_rounds: Maximum refinement rounds
        steps_per_round: Optimization steps per round
        convergence_threshold: Stop if improvement < threshold

    Returns:
        Refinement metrics and history
    """
    device = initial_shapes[0].device
    target = load_target_image(target_path, resolution, device)

    shapes = initial_shapes
    names = shape_names
    round_history = []
    previous_loss = float("inf")

    for round_num in range(max_rounds):
        round_dir = output_dir / f"round_{round_num:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Round {round_num + 1}/{max_rounds} ===")

        # Optimize current geometry
        print("Optimizing...")
        metrics = optimize(
            shapes,
            target,
            resolution,
            steps_per_round,
            lr=0.02,
            save_every=None,  # Don't save intermediate frames for refinement
            output_dir=None,
        )

        # Save outputs
        tau = 0.5 / resolution
        png_path = round_dir / "optimized.png"
        svg_path = round_dir / "optimized.svg"
        save_render(shapes, resolution, resolution, tau, device, str(png_path))
        shapes_to_svg(shapes, resolution, resolution, svg_path)

        # Judge evaluation
        print("Evaluating with judge...")
        try:
            feedback = judge_svg(svg_path, png_path, target_path, metrics)

            # Save feedback
            feedback_path = round_dir / "feedback.json"
            with feedback_path.open("w") as f:
                json.dump(feedback.model_dump(), f, indent=2)

            print(f"Quality: {feedback.overall_quality:.2f}")
            print(f"Similarity: {feedback.similarity_to_target:.2f}")

            # Check convergence
            loss_improvement = previous_loss - metrics["final_loss"]
            if feedback.ready_for_refinement and loss_improvement < convergence_threshold:
                print("Converged! No major improvements needed.")
                round_history.append(
                    {
                        "round": round_num,
                        "metrics": metrics,
                        "feedback": feedback.model_dump(),
                        "converged": True,
                    }
                )
                break

            # Architect proposes edits
            print("Generating edits with architect...")
            arch_response = architect_edits(feedback)

            # Save architect response
            arch_path = round_dir / "architect.json"
            with arch_path.open("w") as f:
                json.dump(arch_response.model_dump(), f, indent=2)

            print(f"Proposed {len(arch_response.actions)} edits")
            print(f"Rationale: {arch_response.rationale[:100]}...")

            # Apply edits
            print("Applying edits...")
            shapes, names = parse_edits(arch_response.actions, shapes, names)

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

            previous_loss = metrics["final_loss"]

        except Exception as e:
            print(f"Warning: Round {round_num} failed with error: {e}")
            round_history.append(
                {
                    "round": round_num,
                    "metrics": metrics,
                    "error": str(e),
                }
            )
            break

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
                "history": round_history,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Refinement complete! {len(round_history)} rounds")
    print(f"Final outputs: {final_dir}")

    return {
        "rounds_completed": len(round_history),
        "history": round_history,
        "final_shapes": len(shapes),
    }
