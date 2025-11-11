from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from textwrap import dedent

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console

console = Console()

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env.local")


class ImageValidation(BaseModel):
    """
    Structured validation result from LLM image analysis.
    """

    is_blank: bool
    has_shapes: bool
    shapes_recognizable: bool
    resembles_pelican: bool
    on_canvas: bool
    description: str
    issues: list[str]
    similarity_to_target: float | None = None


def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode image file to base64 for API transmission.
    """
    with image_path.open("rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: Path) -> str:
    """
    Determine media type from file extension.
    """
    ext = image_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    elif ext == ".gif":
        return "image/gif"
    else:
        return "image/png"


def validate_image(
    image_path: Path, target_path: Path | None = None, fix_suggestions: bool = False
) -> ImageValidation:
    """
    Use multimodal LLM to validate a rendered image.

    Checks:
    - Is the image blank (all white/black)?
    - Does it contain visible geometric shapes?
    - Are the shapes recognizable as a coherent object?
    - Do they resemble a pelican?
    - Are shapes within image bounds (on canvas)?

    If target_path is provided, also computes similarity score.
    If fix_suggestions is True, provides more detailed debugging feedback.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")

    client = anthropic.Anthropic(api_key=api_key)

    # Prepare images
    image_b64 = encode_image_to_base64(image_path)
    image_media_type = get_image_media_type(image_path)

    content_blocks: list[dict] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,
                "data": image_b64,
            },
        }
    ]

    if target_path:
        target_b64 = encode_image_to_base64(target_path)
        target_media_type = get_image_media_type(target_path)
        content_blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": target_media_type,
                    "data": target_b64,
                },
            }
        )

    # Build prompt
    if target_path:
        prompt_text = dedent("""
            Please analyze these two images. The first is a rendered output, the second is the target.

            Provide a structured analysis in JSON format with these fields:
            - is_blank (bool): Is the rendered image all white or all black with no visible content?
            - has_shapes (bool): Does the rendered image contain visible geometric shapes?
            - shapes_recognizable (bool): Do the shapes form a coherent, recognizable object?
            - resembles_pelican (bool): Does it look vaguely pelican-like?
            - on_canvas (bool): Are all shapes fully visible within the image bounds (not cut off)?
            - description (str): Brief description of what you see in the rendered image
            - issues (list[str]): List of specific problems or deficiencies
            - similarity_to_target (float): Similarity score from 0.0 to 1.0 comparing rendered to target

            Respond with ONLY valid JSON, no other text.
            """).strip()
    else:
        prompt_text = dedent("""
            Please analyze this rendered image.

            Provide a structured analysis in JSON format with these fields:
            - is_blank (bool): Is the image all white or all black with no visible content?
            - has_shapes (bool): Does the image contain visible geometric shapes?
            - shapes_recognizable (bool): Do the shapes form a coherent, recognizable object?
            - resembles_pelican (bool): Does it look vaguely pelican-like?
            - on_canvas (bool): Are all shapes fully visible within the image bounds (not cut off)?
            - description (str): Brief description of what you see
            - issues (list[str]): List of specific problems or deficiencies

            Respond with ONLY valid JSON, no other text.
            """).strip()

    if fix_suggestions:
        prompt_text += (
            "\n\nProvide detailed debugging suggestions in the 'issues' field to help fix problems."
        )

    content_blocks.append({"type": "text", "text": prompt_text})

    # Call API
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[{"role": "user", "content": content_blocks}],
    )

    # Parse response - extract text from first text content block
    response_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            response_text = block.text
            break

    if not response_text:
        raise ValueError(f"No text content in response: {response.content}")

    try:
        response_json = json.loads(response_text)
        return ImageValidation(**response_json)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(
            f"Failed to parse LLM response as JSON: {e}\nResponse: {response_text}"
        ) from e


def validate_image_cli() -> None:
    """
    CLI command for image validation.

    Usage: pelican validate-image --image <path> [--target <path>] [--fix-suggestions]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Validate a rendered image using LLM")
    parser.add_argument("--image", required=True, type=Path, help="Path to rendered image")
    parser.add_argument("--target", type=Path, help="Path to target image (optional)")
    parser.add_argument(
        "--fix-suggestions",
        action="store_true",
        help="Provide detailed debugging feedback",
    )

    args = parser.parse_args(sys.argv[2:])

    if not args.image.exists():
        console.print(f"[red]Error: Image file not found: {args.image}[/red]")
        sys.exit(1)

    if args.target and not args.target.exists():
        console.print(f"[red]Error: Target file not found: {args.target}[/red]")
        sys.exit(1)

    try:
        validation = validate_image(args.image, args.target, args.fix_suggestions)
        # Output as JSON for programmatic parsing
        print(validation.model_dump_json(indent=2))
    except Exception as e:
        console.print(f"[red]Error during validation: {e}[/red]")
        sys.exit(1)
