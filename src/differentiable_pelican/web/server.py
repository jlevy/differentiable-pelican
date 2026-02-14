from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue
from typing import TypedDict
from uuid import uuid4

import torch
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from differentiable_pelican.geometry import Shape, create_initial_pelican
from differentiable_pelican.greedy_refine import (
    SHAPE_TYPES,
    create_random_shape,
    create_random_shapes,
)
from differentiable_pelican.optimizer import load_target_image, optimize
from differentiable_pelican.svg_export import shapes_to_svg_string

STATIC_DIR = Path(__file__).parent / "static"


class Session(TypedDict, total=False):
    """In-memory state for an optimization session."""

    image_path: Path
    stop_event: threading.Event
    thread: threading.Thread | None
    queue: Queue[dict]
    running: bool


sessions: dict[str, Session] = {}
_temp_dir: Path | None = None


def _get_temp_dir() -> Path:
    global _temp_dir
    if _temp_dir is None:
        _temp_dir = Path(tempfile.mkdtemp(prefix="pelican_web_"))
    return _temp_dir


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:  # pyright: ignore[reportUnusedParameter]
    yield
    if _temp_dir is not None and _temp_dir.exists():
        shutil.rmtree(_temp_dir, ignore_errors=True)


app = FastAPI(title="Pelican SVG Optimizer", lifespan=_lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page frontend."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


@app.post("/api/upload")
async def upload(file: UploadFile) -> dict:
    """Accept an image upload, save to temp dir, return session_id."""
    session_id = str(uuid4())
    session_dir = _get_temp_dir() / session_id
    session_dir.mkdir(parents=True)

    image_path = session_dir / (file.filename or "target.png")
    content = await file.read()
    image_path.write_bytes(content)

    sessions[session_id] = {
        "image_path": image_path,
        "stop_event": threading.Event(),
        "thread": None,
        "queue": Queue(),
        "running": False,
    }
    return {"session_id": session_id}


def _run_optimize(
    session_id: str,
    shapes: list[Shape],
    target: torch.Tensor,
    resolution: int,
    steps: int,
    lr: float,
    stream_every: int,
) -> None:
    """Run optimization in a background thread, pushing SSE events to the queue."""
    session = sessions[session_id]
    q = session["queue"]
    stop_event = session["stop_event"]
    device = target.device

    def progress_callback(step: int, total_steps: int, loss_breakdown: dict[str, float]) -> None:
        if stop_event.is_set():
            raise KeyboardInterrupt("Optimization cancelled")

        if step % stream_every == 0 or step == total_steps - 1:
            svg_str = shapes_to_svg_string(shapes, resolution, resolution)
            q.put(
                {
                    "type": "progress",
                    "step": step,
                    "total_steps": total_steps,
                    "loss": loss_breakdown.get("total", 0.0),
                    "loss_breakdown": loss_breakdown,
                    "num_shapes": len(shapes),
                    "svg": svg_str,
                }
            )

    try:
        session["running"] = True
        metrics = optimize(
            shapes, target, resolution, steps, lr=lr, progress_callback=progress_callback
        )

        # Send final frame
        final_svg = shapes_to_svg_string(shapes, resolution, resolution)
        q.put(
            {
                "type": "complete",
                "step": steps,
                "total_steps": steps,
                "loss": metrics["final_loss"],
                "num_shapes": len(shapes),
                "svg": final_svg,
            }
        )
    except KeyboardInterrupt:
        q.put({"type": "complete", "step": -1, "total_steps": steps, "cancelled": True})
    except Exception as e:
        q.put({"type": "error", "message": str(e)})
    finally:
        session["running"] = False


def _run_greedy(
    session_id: str,
    initial_shapes: list[Shape],
    target: torch.Tensor,
    resolution: int,
    initial_steps: int,
    settle_steps: int,
    reoptimize_steps: int,
    max_shapes: int,
    scale: float,
    stream_every: int,
    lr: float,
) -> None:
    """Run greedy refinement in a background thread, pushing SSE events to the queue."""
    import copy
    import random

    session = sessions[session_id]
    q = session["queue"]
    stop_event = session["stop_event"]
    device = target.device

    shapes = list(initial_shapes)

    def make_progress_callback(
        step_offset: int, total: int, phase: str
    ) -> callable:  # pyright: ignore
        def cb(step: int, total_steps: int, loss_breakdown: dict[str, float]) -> None:
            if stop_event.is_set():
                raise KeyboardInterrupt("Optimization cancelled")
            global_step = step_offset + step
            if step % stream_every == 0 or step == total_steps - 1:
                svg_str = shapes_to_svg_string(shapes, resolution, resolution)
                q.put(
                    {
                        "type": "progress",
                        "step": global_step,
                        "total_steps": total,
                        "loss": loss_breakdown.get("total", 0.0),
                        "loss_breakdown": loss_breakdown,
                        "num_shapes": len(shapes),
                        "svg": svg_str,
                        "phase": phase,
                    }
                )

        return cb

    try:
        session["running"] = True
        total_estimated = initial_steps + max_shapes * (settle_steps + reoptimize_steps)

        # Phase 1: initial optimization
        optimize(
            shapes,
            target,
            resolution,
            initial_steps,
            lr=lr,
            progress_callback=make_progress_callback(0, total_estimated, "initial"),
        )

        # Phase 2: greedy addition
        step_offset = initial_steps
        consecutive_failures = 0
        type_idx = 0

        while len(shapes) < max_shapes and consecutive_failures < 5:
            if stop_event.is_set():
                break

            shape_type = SHAPE_TYPES[type_idx % len(SHAPE_TYPES)]
            type_idx += 1

            from differentiable_pelican.greedy_refine import _evaluate_loss

            loss_before = _evaluate_loss(shapes, target, resolution, device)

            candidate = create_random_shape(shape_type, device, scale=scale)
            pre_state = copy.deepcopy(shapes)
            shapes.append(candidate)

            # Settle
            optimize(
                shapes,
                target,
                resolution,
                settle_steps,
                lr=lr,
                progress_callback=make_progress_callback(
                    step_offset, total_estimated, "settle"
                ),
            )
            step_offset += settle_steps

            # Re-optimize
            optimize(
                shapes,
                target,
                resolution,
                reoptimize_steps,
                lr=lr,
                progress_callback=make_progress_callback(
                    step_offset, total_estimated, "reoptimize"
                ),
            )
            step_offset += reoptimize_steps

            loss_after = _evaluate_loss(shapes, target, resolution, device)

            if loss_after < loss_before:
                consecutive_failures = 0
                q.put(
                    {
                        "type": "shape_added",
                        "shape_type": shape_type,
                        "num_shapes": len(shapes),
                        "loss_before": loss_before,
                        "loss_after": loss_after,
                    }
                )
            else:
                consecutive_failures += 1
                shapes.clear()
                shapes.extend(pre_state)
                q.put(
                    {
                        "type": "shape_rejected",
                        "shape_type": shape_type,
                        "num_shapes": len(shapes),
                        "loss_before": loss_before,
                        "loss_after": loss_after,
                    }
                )

        final_svg = shapes_to_svg_string(shapes, resolution, resolution)
        q.put(
            {
                "type": "complete",
                "step": step_offset,
                "total_steps": step_offset,
                "num_shapes": len(shapes),
                "svg": final_svg,
            }
        )
    except KeyboardInterrupt:
        q.put({"type": "complete", "cancelled": True})
    except Exception as e:
        q.put({"type": "error", "message": str(e)})
    finally:
        session["running"] = False


@app.get("/api/stream/{session_id}")
async def stream(
    session_id: str,
    mode: str = "optimize",
    resolution: int = 128,
    steps: int = 500,
    lr: float = 0.02,
    stream_every: int = 10,
    initial_shapes_mode: str = "random_5",
    max_shapes: int = 30,
    initial_steps: int = 300,
    settle_steps: int = 80,
    reoptimize_steps: int = 150,
    scale: float = 1.0,
    seed: int = 42,
) -> StreamingResponse:
    """Start optimization and stream SVG frames as Server-Sent Events."""
    if session_id not in sessions:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'message': 'Unknown session'})}\n\n"]),
            media_type="text/event-stream",
        )

    session = sessions[session_id]

    # Cancel any existing run
    if session.get("running"):
        session["stop_event"].set()
        if session.get("thread"):
            session["thread"].join(timeout=5)  # pyright: ignore

    # Reset state
    session["stop_event"] = threading.Event()
    session["queue"] = Queue()
    session["running"] = False

    image_path = session["image_path"]
    device = torch.device("cpu")
    target = load_target_image(str(image_path), resolution, device)

    # Create initial shapes
    import random as _random

    _random.seed(seed)
    torch.manual_seed(seed)

    if initial_shapes_mode == "pelican":
        shapes, _names = create_initial_pelican(device)
    else:
        # Parse "random_N" pattern
        count = 5
        if "_" in initial_shapes_mode:
            try:
                count = int(initial_shapes_mode.split("_")[1])
            except (ValueError, IndexError):
                pass
        shapes, _names = create_random_shapes(count, device, seed=seed)

    if mode == "greedy":
        thread = threading.Thread(
            target=_run_greedy,
            args=(
                session_id,
                shapes,
                target,
                resolution,
                initial_steps,
                settle_steps,
                reoptimize_steps,
                max_shapes,
                scale,
                stream_every,
                lr,
            ),
            daemon=True,
        )
    else:
        thread = threading.Thread(
            target=_run_optimize,
            args=(session_id, shapes, target, resolution, steps, lr, stream_every),
            daemon=True,
        )

    session["thread"] = thread
    thread.start()

    def generate():  # pyright: ignore
        q = sessions[session_id]["queue"]
        while True:
            try:
                event = q.get(timeout=0.5)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("complete", "error"):
                    break
            except Empty:
                if not sessions[session_id].get("running") and q.empty():
                    break
                # Send keepalive
                yield ": keepalive\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/stop/{session_id}")
async def stop(session_id: str) -> dict:
    """Cancel a running optimization."""
    if session_id not in sessions:
        return {"error": "Unknown session"}

    session = sessions[session_id]
    session["stop_event"].set()
    return {"status": "stopping"}


@app.get("/api/status")
async def status() -> dict:
    """Return server status."""
    running_sessions = [sid for sid, s in sessions.items() if s.get("running")]
    return {
        "running": len(running_sessions) > 0,
        "sessions": len(sessions),
        "running_sessions": running_sessions,
    }


## Tests


def test_upload_creates_session():
    import asyncio

    from fastapi.testclient import TestClient

    client = TestClient(app)
    # Create a small test image
    import io

    from PIL import Image

    img = Image.new("L", (64, 64), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post("/api/upload", files={"file": ("test.png", buf, "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["session_id"] in sessions


def test_index_returns_html():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_status_endpoint():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "running" in data
    assert "sessions" in data


def test_stop_unknown_session():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/api/stop/nonexistent")
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
