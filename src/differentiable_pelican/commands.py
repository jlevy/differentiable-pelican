from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from differentiable_pelican.geometry import create_initial_pelican
from differentiable_pelican.renderer import save_render
from differentiable_pelican.svg_export import shapes_to_svg
from differentiable_pelican.utils import ensure_output_dir, pick_device

console = Console()


def render_test_command() -> None:
    """
    CLI command for test rendering (Phase 1A).

    Usage: pelican test-render [--output-dir <path>] [--resolution <size>]
    """
    parser = argparse.ArgumentParser(description="Render initial hard-coded pelican geometry")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/test_render"),
        help="Output directory (default: out/test_render)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Image resolution (default: 64x64)",
    )

    args = parser.parse_args(sys.argv[2:])

    # Setup
    device = pick_device()
    output_dir = ensure_output_dir(args.output_dir)
    resolution = args.resolution

    console.print(
        Panel.fit(
            f"[bold]Differentiable Pelican - Test Render[/bold]\n\n"
            f"Device: {device}\n"
            f"Resolution: {resolution}×{resolution}\n"
            f"Output: {output_dir}",
            border_style="blue",
        )
    )

    # Create initial pelican
    console.print("\n[cyan]Creating initial pelican geometry...[/cyan]")
    shapes, names = create_initial_pelican(device)
    console.print(f"  -> Created {len(shapes)} shapes: {', '.join(names)}")

    # Render to PNG
    console.print("\n[cyan]Rendering to PNG...[/cyan]")
    tau = 1.0 / resolution  # Tau in normalized coordinates
    png_path = output_dir / "pelican_test.png"
    save_render(shapes, resolution, resolution, tau, device, str(png_path))
    console.print(f"  ✓ Saved PNG: {png_path}")

    # Export to SVG
    console.print("\n[cyan]Exporting to SVG...[/cyan]")
    svg_path = output_dir / "pelican_test.svg"
    shapes_to_svg(shapes, resolution, resolution, svg_path)
    console.print(f"  ✓ Saved SVG: {svg_path}")

    console.print("\n[green]✓ Test render complete![/green]")
    console.print(f"\nOutputs:\n  PNG: {png_path}\n  SVG: {svg_path}")
