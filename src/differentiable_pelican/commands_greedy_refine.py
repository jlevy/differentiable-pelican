from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from differentiable_pelican.geometry import create_initial_pelican
from differentiable_pelican.greedy_refine import greedy_refinement_loop
from differentiable_pelican.utils import ensure_output_dir, pick_device, set_seed

console = Console()


def greedy_refine_command() -> None:
    """
    CLI command for greedy shape-dropping refinement.

    Adds one shape at a time, lets gradient descent find optimal placement,
    and keeps only shapes that improve the loss.

    Usage: pelican greedy-refine --target <path> [options]
    """
    parser = argparse.ArgumentParser(
        description="Greedy shape-dropping refinement: add shapes one at a time"
    )
    parser.add_argument("--target", type=Path, required=True, help="Path to target image")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/greedy_refine"),
        help="Output directory (default: out/greedy_refine)",
    )
    parser.add_argument("--resolution", type=int, default=128, help="Image resolution (default: 128)")
    parser.add_argument("--max-shapes", type=int, default=20, help="Max total shapes (default: 20)")
    parser.add_argument(
        "--initial-steps", type=int, default=500, help="Initial optimization steps (default: 500)"
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=100,
        help="Steps to settle new shape (default: 100)",
    )
    parser.add_argument(
        "--reoptimize-steps",
        type=int,
        default=200,
        help="Steps to re-optimize all shapes after adding (default: 200)",
    )
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="Don't freeze existing shapes during settle phase",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Size multiplier for new shapes (default: 1.0)"
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=5,
        help="Stop after N consecutive rejections (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args(sys.argv[2:])

    if not args.target.exists():
        console.print(f"[red]Error: Target file not found: {args.target}[/red]")
        sys.exit(1)

    set_seed(args.seed)
    device = pick_device()
    output_dir = ensure_output_dir(args.output_dir)

    console.print(
        Panel.fit(
            f"[bold]Differentiable Pelican - Greedy Refinement[/bold]\n\n"
            f"Target: {args.target}\n"
            f"Device: {device}\n"
            f"Resolution: {args.resolution}x{args.resolution}\n"
            f"Max Shapes: {args.max_shapes}\n"
            f"Settle Steps: {args.settle_steps}\n"
            f"Re-optimize Steps: {args.reoptimize_steps}\n"
            f"Freeze Existing: {not args.no_freeze}\n"
            f"Scale: {args.scale}\n"
            f"Output: {output_dir}",
            border_style="blue",
        )
    )

    # Create initial pelican
    console.print("\n[cyan]Creating initial pelican...[/cyan]")
    shapes, names = create_initial_pelican(device)
    console.print(f"  -> Created {len(shapes)} shapes: {', '.join(names)}")

    result = greedy_refinement_loop(
        initial_shapes=shapes,
        shape_names=names,
        target_path=args.target,
        resolution=args.resolution,
        output_dir=output_dir,
        max_shapes=args.max_shapes,
        initial_steps=args.initial_steps,
        settle_steps=args.settle_steps,
        reoptimize_steps=args.reoptimize_steps,
        freeze_existing=not args.no_freeze,
        max_consecutive_failures=args.max_failures,
        scale=args.scale,
        seed=args.seed,
    )

    console.print("\n[green]Done![/green]")
    console.print(f"  Shapes added: {result['shapes_added']}")
    console.print(f"  Final shape count: {result['final_shapes']}")
    console.print(f"  Loss: {result['initial_loss']:.6f} -> {result['final_loss']:.6f}")
    console.print(f"\n  Outputs: {output_dir}")
