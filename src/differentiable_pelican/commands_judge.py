from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from differentiable_pelican.llm.judge import judge_svg

console = Console()


def judge_command() -> None:
    """
    CLI command for judging optimized SVG (Phase 2A).

    Usage: pelican judge --svg <path> --png <path> --target <path>
    """
    parser = argparse.ArgumentParser(description="Evaluate optimized SVG with LLM judge")
    parser.add_argument("--svg", type=Path, required=True, help="Path to SVG file")
    parser.add_argument("--png", type=Path, required=True, help="Path to rendered PNG")
    parser.add_argument("--target", type=Path, required=True, help="Path to target image")
    parser.add_argument(
        "--metrics", type=Path, help="Optional path to metrics JSON from optimization"
    )
    parser.add_argument(
        "--output", type=Path, help="Optional path to save feedback JSON"
    )

    args = parser.parse_args(sys.argv[2:])

    # Check files exist
    for path, name in [(args.svg, "SVG"), (args.png, "PNG"), (args.target, "Target")]:
        if not path.exists():
            console.print(f"[red]Error: {name} file not found: {path}[/red]")
            sys.exit(1)

    console.print(
        Panel.fit(
            f"[bold]Differentiable Pelican - Judge[/bold]\n\n"
            f"SVG: {args.svg}\n"
            f"PNG: {args.png}\n"
            f"Target: {args.target}",
            border_style="blue",
        )
    )

    # Load metrics if provided
    metrics = None
    if args.metrics and args.metrics.exists():
        with args.metrics.open() as f:
            metrics = json.load(f)

    # Call judge
    console.print("\n[cyan]Evaluating with LLM judge...[/cyan]")
    try:
        feedback = judge_svg(args.svg, args.png, args.target, metrics)

        # Display feedback
        console.print("\n[green]✓ Evaluation complete![/green]\n")

        console.print(f"[bold]Overall Quality:[/bold] {feedback.overall_quality:.2f}")
        console.print(
            f"[bold]Resembles Pelican:[/bold] {'Yes' if feedback.resembles_pelican else 'No'}"
        )
        console.print(
            f"[bold]Similarity to Target:[/bold] {feedback.similarity_to_target:.2f}"
        )
        console.print(
            f"[bold]Ready for Refinement:[/bold] {'Yes' if feedback.ready_for_refinement else 'No'}"
        )

        if feedback.geometric_accuracy:
            console.print(f"\n[bold]Geometric Accuracy:[/bold]\n{feedback.geometric_accuracy}")

        if feedback.missing_features:
            console.print(f"\n[bold]Missing Features:[/bold]")
            for feature in feedback.missing_features:
                console.print(f"  • {feature}")

        if feedback.topology_issues:
            console.print(f"\n[bold]Topology Issues:[/bold]")
            for issue in feedback.topology_issues:
                console.print(f"  • {issue}")

        if feedback.suggestions:
            console.print(f"\n[bold]Suggestions:[/bold]")
            for suggestion in feedback.suggestions:
                console.print(f"  • {suggestion}")

        # Save feedback if requested
        if args.output:
            with args.output.open("w") as f:
                json.dump(feedback.model_dump(), f, indent=2)
            console.print(f"\n✓ Saved feedback: {args.output}")

    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")
        sys.exit(1)
