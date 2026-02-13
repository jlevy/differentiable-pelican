from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from differentiable_pelican.geometry import create_initial_pelican
from differentiable_pelican.optimizer import load_target_image, optimize
from differentiable_pelican.renderer import save_render
from differentiable_pelican.svg_export import shapes_to_svg
from differentiable_pelican.utils import ensure_output_dir, pick_device, set_seed

console = Console()


def optimize_command() -> None:
    """
    CLI command for optimization (Phase 1B/1C).

    Usage: pelican optimize --target <path> [options]
    """
    parser = argparse.ArgumentParser(description="Optimize geometry to match target image")
    parser.add_argument("--target", type=Path, required=True, help="Path to target image")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/optimize"),
        help="Output directory (default: out/optimize)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Image resolution (default: 128x128)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of optimization steps (default: 500)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        help="Learning rate (default: 0.02)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Save frames every N steps (default: 25, 0=disable)",
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
    resolution = args.resolution
    save_every = args.save_every if args.save_every > 0 else None

    console.print(
        Panel.fit(
            f"[bold]Differentiable Pelican - Optimize[/bold]\n\n"
            f"Target: {args.target}\n"
            f"Device: {device}\n"
            f"Resolution: {resolution}x{resolution}\n"
            f"Steps: {args.steps}\n"
            f"Learning Rate: {args.lr}\n"
            f"Output: {output_dir}",
            border_style="blue",
        )
    )

    # Load target
    console.print("\n[cyan]Loading target image...[/cyan]")
    target = load_target_image(args.target, resolution, device)
    console.print(f"  -> Loaded {args.target}")

    # Create initial pelican
    console.print("\n[cyan]Creating initial pelican geometry...[/cyan]")
    shapes, names = create_initial_pelican(device)
    console.print(f"  -> Created {len(shapes)} shapes: {', '.join(names)}")

    # Optimize with live progress bar
    console.print("\n[cyan]Optimizing...[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("loss: {task.fields[loss]:.4f}"),
        console=console,
    ) as progress:
        task = progress.add_task("Optimizing", total=args.steps, loss=0.0)

        def on_progress(step: int, _total: int, breakdown: dict[str, float]) -> None:
            progress.update(task, completed=step + 1, loss=breakdown["total"])

        metrics = optimize(
            shapes,
            target,
            resolution,
            args.steps,
            lr=args.lr,
            save_every=save_every,
            output_dir=output_dir,
            progress_callback=on_progress,
        )

    # Save final outputs
    console.print("\n[cyan]Saving outputs...[/cyan]")

    # Render final PNG
    tau = 0.5 / resolution
    png_path = output_dir / "pelican_optimized.png"
    save_render(shapes, resolution, resolution, tau, device, str(png_path))
    console.print(f"  -> Saved PNG: {png_path}")

    # Export SVG
    svg_path = output_dir / "pelican_optimized.svg"
    shapes_to_svg(shapes, resolution, resolution, svg_path)
    console.print(f"  -> Saved SVG: {svg_path}")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    console.print(f"  -> Saved metrics: {metrics_path}")

    # Generate animation if frames were saved
    if save_every:
        console.print("\n[cyan]Generating animation...[/cyan]")
        try:
            import imageio.v3 as iio

            frames_dir = output_dir / "frames"
            frame_files = sorted(frames_dir.glob("frame_*.png"))
            if frame_files:
                gif_path = output_dir / "optimization.gif"
                images = [iio.imread(str(f)) for f in frame_files]  # pyright: ignore[reportUnknownMemberType]
                iio.imwrite(str(gif_path), images, duration=250, loop=0)  # pyright: ignore[reportUnknownMemberType]
                console.print(f"  -> Saved animation: {gif_path}")
        except Exception as e:
            console.print(f"  [yellow]Warning: Could not create GIF: {e}[/yellow]")

    # Print final stats
    console.print("\n[green]Optimization complete![/green]")
    console.print(f"\nFinal Loss: {metrics['final_loss']:.6f}")
    console.print(f"Steps Completed: {metrics['steps_completed']}")
    console.print(f"\nOutputs:\n  PNG: {png_path}\n  SVG: {svg_path}\n  Metrics: {metrics_path}")
