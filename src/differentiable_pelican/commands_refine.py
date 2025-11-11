from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from differentiable_pelican.geometry import create_initial_pelican
from differentiable_pelican.refine import refinement_loop
from differentiable_pelican.utils import ensure_output_dir, pick_device, set_seed

console = Console()


def refine_command() -> None:
    """
    CLI command for full refinement loop (Phase 2C).

    Usage: pelican refine --target <path> [options]
    """
    parser = argparse.ArgumentParser(
        description="Full refinement loop with LLM judge and architect"
    )
    parser.add_argument("--target", type=Path, required=True, help="Path to target image")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/refine"),
        help="Output directory (default: out/refine)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Image resolution (default: 128x128)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Maximum refinement rounds (default: 3)",
    )
    parser.add_argument(
        "--steps-per-round",
        type=int,
        default=200,
        help="Optimization steps per round (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args(sys.argv[2:])

    if not args.target.exists():
        console.print(f"[red]Error: Target file not found: {args.target}[/red]")
        sys.exit(1)

    # Setup
    set_seed(args.seed)
    device = pick_device()
    output_dir = ensure_output_dir(args.output_dir)

    console.print(
        Panel.fit(
            f"[bold]Differentiable Pelican - Refinement Loop[/bold]\n\n"
            f"Target: {args.target}\n"
            f"Device: {device}\n"
            f"Resolution: {args.resolution}×{args.resolution}\n"
            f"Max Rounds: {args.rounds}\n"
            f"Steps/Round: {args.steps_per_round}\n"
            f"Output: {output_dir}",
            border_style="blue",
        )
    )

    # Create initial pelican
    console.print("\n[cyan]Creating initial pelican...[/cyan]")
    shapes = create_initial_pelican(device)
    shape_names = ["body", "head", "beak", "eye", "wing"]
    console.print(f"  ✓ Created {len(shapes)} shapes")

    # Run refinement loop
    try:
        result = refinement_loop(
            shapes,
            shape_names,
            args.target,
            args.resolution,
            output_dir,
            max_rounds=args.rounds,
            steps_per_round=args.steps_per_round,
        )

        console.print(f"\n[green]✓ Refinement complete![/green]")
        console.print(f"\nRounds completed: {result['rounds_completed']}")
        console.print(f"Final shape count: {result['final_shapes']}")
        console.print(f"\nOutputs saved to: {output_dir}")

    except Exception as e:
        console.print(f"[red]Error during refinement: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)
