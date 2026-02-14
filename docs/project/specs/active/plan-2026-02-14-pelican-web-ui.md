# Feature: Interactive Web UI for Differentiable Pelican

**Date:** 2026-02-14

**Author:** Joshua Levy

**Status:** Draft

## Overview

A minimal, modern web interface for the differentiable Pelican pipeline. A new `pelican serve`
command launches a local web server that opens in the browser. Users can drop any image onto
the page, configure optimization parameters, and watch the algorithm run in real time as SVG
shapes iteratively adapt to match the target image.

The core experience: drop an image, hit run, and watch shapes appear and morph step by step,
streaming intermediate SVG frames live to the browser.

## Goals

- Add a `pelican serve` command that starts a local web server and opens the browser
- Clean, minimal, modern UI -- plain HTML/CSS/JS, no frontend build step
- Drag-and-drop (or file chooser) for target image upload
- Configurable parameters: resolution, number of steps, learning rate, max shapes, etc.
- Live-streamed SVG animation as optimization runs, updating in real time
- Support both `optimize` (fixed shape set) and `greedy-refine` (incremental shape addition)
  modes from the web UI
- Shape initialization that starts with random shapes and visibly converges toward the target

## Non-Goals

- Multi-user or remote deployment (local single-user only)
- User accounts, authentication, or persistence across server restarts
- Mobile-optimized layout (desktop browser is fine)
- Editing individual shapes by hand in the browser (future work)
- LLM-guided refinement from the web UI (the `refine` command requires API keys and
  multi-round LLM calls; out of scope for v1)
- Color support (grayscale only, matching current pipeline)

## Background

The project is currently CLI-only, using Rich for terminal output. All rendering and
optimization happens synchronously, with results written to disk as PNG, SVG, GIF, and
JSON files. The `optimize` command runs a fixed number of gradient descent steps on an
initial set of 9 hardcoded shapes. The `greedy-refine` command adds shapes one at a time,
accepting or rejecting each based on loss improvement.

The optimization loop already supports a `progress_callback` parameter
(`optimizer.py:26, 60, 147-148`), which makes it straightforward to hook in real-time
streaming. The `shapes_to_svg` function (`svg_export.py:11-39`) generates complete SVG
strings that can be sent directly to the browser.

The design document (`pelican-plan.md:393`) lists "Simple web viewer with auto-refresh"
as a Phase 2 future enhancement, and line 1539 mentions "Interactive web viewer: Real-time
parameter tuning in browser" as a medium-term extension.

## Design

### Web Architecture

```
Browser (HTML/CSS/JS)              Python Server (FastAPI + uvicorn)
========================           =================================

 ┌──────────────────┐              ┌──────────────────┐
 │  Drop Zone /     │  POST        │  POST /api/upload │
 │  File Chooser    │ ─────────►   │  Save image to    │
 │                  │  multipart   │  temp dir, return  │
 └──────────────────┘  form data   │  session_id        │
                                   └──────────────────┘
 ┌──────────────────┐              ┌──────────────────┐
 │  Parameter Panel │  POST        │  POST /api/run    │
 │  [Run] button    │ ─────────►   │  Start optimize   │
 │                  │  JSON body   │  in background     │
 └──────────────────┘  + session   │  thread            │
                                   └──────────────────┘
 ┌──────────────────┐              ┌──────────────────┐
 │  SVG Display     │  SSE         │  GET /api/stream  │
 │  (live updating) │ ◄─────────   │  Server-Sent      │
 │                  │  text/       │  Events with SVG   │
 │  Progress bar    │  event-      │  + metrics each    │
 │  Loss display    │  stream      │  N steps           │
 │  Step counter    │              │                    │
 └──────────────────┘              └──────────────────┘
 ┌──────────────────┐              ┌──────────────────┐
 │  [Stop] button   │  POST        │  POST /api/stop   │
 │                  │ ─────────►   │  Cancel running    │
 │                  │              │  optimization      │
 └──────────────────┘              └──────────────────┘
```

#### Protocol: Server-Sent Events (SSE)

SSE is chosen over WebSocket because:
- Data flows in one direction (server to client) during optimization
- Simpler to implement -- just a streaming HTTP response
- Automatic reconnection built into the browser `EventSource` API
- No additional dependencies needed

Each SSE event contains a JSON payload:

```json
{
  "type": "progress",
  "step": 42,
  "total_steps": 500,
  "loss": 0.0523,
  "loss_breakdown": {"mse": 0.045, "edge": 0.003, "ssim": 0.002, ...},
  "num_shapes": 12,
  "svg": "<svg xmlns=\"http://www.w3.org/2000/svg\" ...>...</svg>"
}
```

Event types:
- `progress` -- intermediate step with SVG + metrics (sent every N steps)
- `shape_added` -- greedy mode: a new shape was accepted (includes shape info)
- `shape_rejected` -- greedy mode: a candidate shape was discarded
- `complete` -- optimization finished, final SVG + summary metrics
- `error` -- something went wrong

#### Image Upload Flow

1. User drops image or selects via file chooser
2. Browser reads the file client-side and shows a thumbnail preview
3. On "Run," the image is sent as `multipart/form-data` to `POST /api/upload`
4. Server saves to a temp directory, returns a `session_id` (UUID)
5. Browser opens SSE connection to `GET /api/stream/{session_id}?mode=greedy&resolution=128&...`
6. Server loads the image, creates initial shapes, and begins optimization
7. Every `stream_every` steps (default 10), the progress callback generates an SVG string
   and pushes it as an SSE event
8. Browser replaces the SVG element's `innerHTML` with the new SVG content

#### Concurrency Model

- FastAPI runs with uvicorn on a single worker
- Optimization runs in a background thread (not async -- PyTorch is CPU-bound)
- A threading `Event` flag supports cancellation: the progress callback checks it each step
- Only one optimization can run at a time (single-user local tool); a second request
  cancels the first
- Temp files are cleaned up on server shutdown

### Frontend Architecture

The frontend is a single HTML file with embedded CSS and JS -- no build step, no npm,
no framework. This keeps it dead simple to maintain alongside the Python package.

#### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Pelican SVG Optimizer                                      │
├─────────────────────────────┬───────────────────────────────┤
│                             │  Target Image                 │
│                             │  ┌───────────────────────┐    │
│   Generated SVG             │  │  (thumbnail preview)  │    │
│   ┌───────────────────────┐ │  └───────────────────────┘    │
│   │                       │ │                               │
│   │   (live SVG output)   │ │  Parameters                   │
│   │                       │ │  ┌───────────────────────┐    │
│   │                       │ │  │ Mode: [optimize ▾]    │    │
│   │                       │ │  │ Resolution: [128]     │    │
│   │                       │ │  │ Steps: ──●────── 500  │    │
│   │                       │ │  │ Learning rate: 0.02   │    │
│   │                       │ │  │ Max shapes: ──●── 30  │    │
│   │                       │ │  │ Stream every: 10      │    │
│   └───────────────────────┘ │  └───────────────────────┘    │
│                             │                               │
│   Step 42/500  Loss: 0.052  │  [▶ Run]  [■ Stop]           │
│   ████████░░░░░░░░  8.4%    │                               │
│                             │  ┌───────────────────────┐    │
│   Shapes: 12                │  │  Drop image here      │    │
│                             │  │  or click to browse    │    │
│                             │  └───────────────────────┘    │
├─────────────────────────────┴───────────────────────────────┤
│  Loss: 0.0523 | MSE: 0.045 | Shapes: 12 | Best: 0.048     │
└─────────────────────────────────────────────────────────────┘
```

#### Styling Approach

- **No CSS framework.** Use plain CSS with custom properties (variables) for theming.
- Modern defaults: `system-ui` font stack, subtle borders, generous whitespace.
- A muted, professional color palette (grays, one accent color).
- Responsive two-column layout with CSS grid; collapses to single column on narrow windows.
- Minimal visual noise: no gradients, no shadows, no rounded-everything. Flat and clean.
- Interactive elements styled with `:hover`, `:active`, `:disabled` states.
- Sliders styled with accent color via `accent-color` CSS property.
- Inspired by the Shadcn/ui aesthetic: understated, typographic, functional.

#### JavaScript

Vanilla JS, no framework. Key functions:

- `handleDrop(event)` / `handleFileSelect(event)` -- read image, show preview
- `startOptimization()` -- POST upload, then open `EventSource` for SSE stream
- `stopOptimization()` -- POST stop, close EventSource
- `onSSEMessage(event)` -- parse JSON, update SVG display, progress bar, metrics
- `updateParams()` -- read slider/input values into a params object

The SVG update is a simple `innerHTML` replacement on a container `<div>`. Since each
SSE event contains a complete SVG document, no incremental DOM patching is needed.

### Backend Components

#### New Files

| File | Purpose |
|------|---------|
| `src/differentiable_pelican/web/__init__.py` | Web subpackage |
| `src/differentiable_pelican/web/server.py` | FastAPI app, routes, SSE streaming |
| `src/differentiable_pelican/web/static/index.html` | Single-file frontend (HTML + CSS + JS) |
| `src/differentiable_pelican/commands_serve.py` | `pelican serve` CLI command |

#### Modified Files

| File | Change |
|------|--------|
| `cli.py` | Add `serve` command to dispatcher |
| `svg_export.py` | Add `shapes_to_svg_string()` that returns SVG as a string instead of writing to file |
| `pyproject.toml` | Add optional `[web]` dependency group for `fastapi` and `uvicorn` |

#### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Serve the static HTML page |
| `POST` | `/api/upload` | Accept image file, return `{session_id}` |
| `GET` | `/api/stream/{session_id}` | SSE stream of optimization progress |
| `POST` | `/api/stop/{session_id}` | Cancel running optimization |
| `GET` | `/api/status` | Server status (is optimization running?) |

#### Dependencies

Add as an optional dependency group so the core package doesn't require web deps:

```toml
[project.optional-dependencies]
web = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
]
```

Install with: `pip install differentiable-pelican[web]` or `uv pip install -e ".[web]"`

#### Server Implementation Sketch

```python
# web/server.py (conceptual outline)

app = FastAPI()

# In-memory session state
sessions: dict[str, Session] = {}

@app.post("/api/upload")
async def upload(file: UploadFile):
    session_id = str(uuid4())
    # Save file to tempdir
    # Return {"session_id": session_id}

@app.get("/api/stream/{session_id}")
async def stream(session_id: str, mode: str, resolution: int, ...):
    # Validate session exists
    # Create initial shapes (random or pelican)
    # Start optimization in background thread
    # Return StreamingResponse with SSE events

    def generate():
        # The progress_callback pushes events to a queue
        # This generator yields from the queue as SSE-formatted text
        while not done:
            event = queue.get()
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Shape Initialization Strategy

For the web UI, we support two initialization approaches:

1. **Random scatter** (default for web): Start with N random shapes of mixed types
   (circles, ellipses, triangles) scattered across the canvas at various sizes and
   intensities. As optimization runs, they visibly slide, resize, and reshape to match
   the target. This produces a compelling animation where order emerges from chaos.

2. **Pelican preset**: Use the hardcoded 9-shape pelican geometry from
   `create_initial_pelican()`. Useful for the pelican target image specifically.

In greedy mode, the random-scatter initialization is especially engaging: start with
just 1-3 random shapes, watch them optimize, then see new shapes appear one at a time,
each finding its place. With `stream_every` set low (e.g., every 5 steps), the user sees
smooth animation of shapes morphing into position.

### Parameter Controls

Parameters exposed in the UI, with defaults that work well for interactive use:

| Parameter | Type | Default | Range | Notes |
|-----------|------|---------|-------|-------|
| Mode | dropdown | `greedy` | `optimize`, `greedy` | Which pipeline to run |
| Resolution | number | 128 | 64-256 | Higher = slower but more detail |
| Steps (optimize) | slider | 500 | 100-2000 | Total optimization steps |
| Learning rate | number | 0.02 | 0.001-0.1 | Adam optimizer LR |
| Max shapes (greedy) | slider | 30 | 5-100 | Shape budget for greedy mode |
| Initial steps (greedy) | number | 300 | 100-1000 | Steps for initial shape set |
| Settle steps (greedy) | number | 80 | 20-500 | Steps for new shape settling |
| Re-optimize steps (greedy) | number | 150 | 50-500 | Steps for full re-optimization |
| Stream every | slider | 10 | 1-50 | Send SVG every N steps |
| Initial shapes | dropdown | `random (5)` | `random (3)`, `random (5)`, `random (10)`, `pelican (9)` | Starting shape configuration |
| Scale (greedy) | number | 1.0 | 0.5-2.0 | Size multiplier for new shapes |
| Seed | number | 42 | 0-9999 | Random seed for reproducibility |

Advanced parameters (learning rate, settle steps, etc.) are collapsed by default behind
a "Show advanced" toggle.

## Implementation Plan

### Phase 1: Server, UI, and Basic Optimization

Core infrastructure: web server, static frontend, image upload, and running the
`optimize` pipeline with final-result delivery (not yet streaming).

- [ ] Add `shapes_to_svg_string()` to `svg_export.py` (returns SVG markup as a string)
- [ ] Add `fastapi` and `uvicorn` as optional `[web]` dependencies in `pyproject.toml`
- [ ] Create `web/` subpackage with `server.py` (FastAPI app, upload + run endpoints)
- [ ] Create `web/static/index.html` (single-file frontend with HTML + CSS + JS)
- [ ] Implement drag-and-drop image upload and thumbnail preview
- [ ] Implement parameter panel with sliders and inputs
- [ ] Wire up `POST /api/upload` endpoint (save image, return session ID)
- [ ] Wire up run endpoint that executes `optimize()` and returns final SVG
- [ ] Create `commands_serve.py` with `pelican serve` CLI command (host, port, open browser)
- [ ] Register `serve` command in `cli.py` dispatcher
- [ ] Add `create_random_shapes()` utility for random initialization

### Phase 2: Live SSE Streaming and Greedy Refinement

The marquee feature: real-time streaming of intermediate SVG frames as optimization runs,
plus integration with the greedy refinement loop for incremental shape addition.

- [ ] Implement SSE streaming endpoint (`GET /api/stream/{session_id}`)
- [ ] Add thread-based optimization runner with queue-based event dispatch
- [ ] Hook `progress_callback` to generate SVG string + push to SSE queue every N steps
- [ ] Implement stop/cancel endpoint (`POST /api/stop/{session_id}`)
- [ ] Frontend: replace static result display with live-updating SVG via `EventSource`
- [ ] Frontend: progress bar, step counter, and live loss display
- [ ] Integrate `greedy_refinement_loop` with SSE streaming (stream each greedy round)
- [ ] Frontend: mode switcher (optimize vs. greedy) with parameter panel adaptation
- [ ] Frontend: shape count indicator and accepted/rejected shape notifications
- [ ] Add `/api/status` endpoint for polling server state

## Testing Strategy

- **Unit tests**: `shapes_to_svg_string()` returns valid SVG markup
- **Integration tests**: FastAPI test client (`httpx`) to test upload, run, and SSE endpoints
- **Manual testing**: Drop various images, verify streaming works, check parameter changes
  take effect, confirm stop/cancel works mid-optimization
- **Browser testing**: Verify drag-and-drop, SSE connection, SVG rendering across
  Chrome/Firefox/Safari

## Rollout Plan

- Ship as part of the `differentiable-pelican` package with optional `[web]` deps
- `pelican serve` fails gracefully with a clear message if FastAPI/uvicorn not installed
- Document in README under a "Web UI" section

## Open Questions

- Should the web UI support saving/downloading the final SVG and PNG? (Probably yes --
  a simple download button is low effort and high value.)
- Should we show a small loss chart (sparkline or mini line chart) that updates live?
  This would be visually informative but adds JS complexity. Could use a simple
  canvas-based sparkline with no dependencies.
- For greedy mode, should we allow the user to manually trigger "add a shape" rather
  than running the full loop automatically? This would give more interactive control but
  changes the UX model.
- Is there value in showing the SDF or coverage map alongside the SVG for educational
  purposes?

## References

- [Design document](../../design/pelican-plan.md) -- Phase 2 visualization plans
  (line 393) and interactive web viewer future extension (line 1539)
- [Greedy refinement spec](plan-2026-02-13-greedy-refinement-loop.md)
- [Original spec](plan-2026-01-15-differentiable-pelican.md)
- FastAPI SSE streaming: `StreamingResponse` with `text/event-stream` media type
- MDN EventSource API: browser-native SSE client
