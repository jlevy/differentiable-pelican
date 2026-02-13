from __future__ import annotations

import sys

from rich.console import Console

_COMMANDS = {
    "validate-image": "Validate a rendered image using LLM",
    "test-render": "Render initial hard-coded geometry",
    "optimize": "Optimize geometry to match target",
    "judge": "Evaluate optimized SVG",
    "refine": "Full refinement loop with LLM",
    "greedy-refine": "Greedy shape-dropping refinement (no LLM)",
}

console = Console()
err_console = Console(stderr=True)


def _print_usage() -> None:
    console.print("Usage: pelican <command> [options]\n")
    console.print("Available commands:")
    for name, desc in _COMMANDS.items():
        console.print(f"  {name:<18s} {desc}")
    console.print("\nRun 'pelican <command> --help' for command-specific options.")


def app() -> None:
    """
    Main CLI entry point.
    """
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        _print_usage()
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    command = sys.argv[1]

    if command == "validate-image":
        from differentiable_pelican.validator import validate_image_cli

        validate_image_cli()
    elif command == "test-render":
        from differentiable_pelican.commands import render_test_command

        render_test_command()
    elif command == "optimize":
        from differentiable_pelican.commands_optimize import optimize_command

        optimize_command()
    elif command == "judge":
        from differentiable_pelican.commands_judge import judge_command

        judge_command()
    elif command == "refine":
        from differentiable_pelican.commands_refine import refine_command

        refine_command()
    elif command == "greedy-refine":
        from differentiable_pelican.commands_greedy_refine import greedy_refine_command

        greedy_refine_command()
    else:
        err_console.print(f"[red]Error: Unknown command '{command}'[/red]")
        _print_usage()
        sys.exit(1)
