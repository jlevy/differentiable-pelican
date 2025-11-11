from __future__ import annotations

import sys

from rich.console import Console

console = Console()


def app() -> None:
    """
    Main CLI entry point.
    """
    if len(sys.argv) < 2:
        console.print("[red]Error: No command specified[/red]")
        console.print("\nAvailable commands:")
        console.print("  validate-image  - Validate a rendered image using LLM")
        console.print("  test-render     - Render initial hard-coded geometry")
        console.print("  optimize        - Optimize geometry to match target")
        console.print("  export          - Export parameters to SVG")
        console.print("  judge           - Evaluate optimized SVG")
        console.print("  refine          - Full refinement loop with LLM")
        sys.exit(1)

    command = sys.argv[1]

    if command == "validate-image":
        from differentiable_pelican.validator import validate_image_cli

        validate_image_cli()
    elif command == "test-render":
        from differentiable_pelican.commands import test_render_command

        test_render_command()
    elif command == "optimize":
        from differentiable_pelican.commands_optimize import optimize_command

        optimize_command()
    elif command == "export":
        console.print("[yellow]export command not yet implemented[/yellow]")
        sys.exit(1)
    elif command == "judge":
        from differentiable_pelican.commands_judge import judge_command

        judge_command()
    elif command == "refine":
        from differentiable_pelican.commands_refine import refine_command

        refine_command()
    else:
        console.print(f"[red]Error: Unknown command '{command}'[/red]")
        sys.exit(1)
