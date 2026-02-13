# Differentiable Pelican: Gradient-Based SVG Optimization with LLM Refinement

## Problem Statement

**Goal**: Create a system that generates high-quality SVG drawings of a pelican by
combining:

1. Differentiable rendering and gradient descent to optimize geometric primitives

2. Image-based loss functions to match a target cartoon pelican

3. LLM-guided structural refinement for symbolic/conceptual improvements

**Key Constraints**:

- Output must be clean, minimal SVG (ellipses, circles, polygons—no complex paths
  initially)

- Must work on CPU (basic Linux cloud) and scale to GPU/MPS (M1 Mac, cloud GPU)

- Geometry must remain interpretable (no pixel soup or texture hacks)

- System should be observable: live preview, metrics, intermediate outputs

**Core Challenge**: SVG primitives are discrete and symbolic, but optimization requires
continuous gradients.
We solve this by:

1. Parameterizing shapes with continuous variables (centers, radii, rotations)

2. Rendering via differentiable soft SDFs (signed distance fields)

3. Backpropagating through the rasterized image to update shape parameters

4. Using an LLM to make discrete structural changes (add/remove shapes, change topology)

**Target Image**: Use the image here as the desired pelican drawing:
images/pelican-drawing-1.jpg

* * *

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Differentiable Pelican                    │
└─────────────────────────────────────────────────────────────┘

Phase 1: Gradient-Based Optimization
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Hard-coded  │────▶│ Differentiable│────▶│ Backprop to  │
│  Pelican     │     │ SDF Renderer  │     │ Match Target │
│  Geometry    │     │ (PyTorch)     │     │ Image        │
└──────────────┘     └───────────────┘     └──────────────┘
       │                     │                     │
       │                     ▼                     │
       │              ┌──────────────┐             │
       │              │ Rasterized   │             │
       │              │ 128×128      │◀────────────┘
       │              │ Grayscale    │
       │              └──────────────┘
       │                     │
       ▼                     ▼
┌──────────────────────────────────────┐
│  SVG Export (ellipse, circle, poly)  │
└──────────────────────────────────────┘

Phase 2: LLM Refinement Loop
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Optimized   │────▶│   Judge      │────▶│     LLM      │
│  SVG + PNG   │     │ (Multimodal) │     │  Architect   │
└──────────────┘     └──────────────┘     └──────────────┘
       ▲                     │                ▲    │
       │                     │                │    │
       │                     ▼                │    │
       │              ┌──────────────┐        │    │
       │              │  Structured  │────────┘    │
       │              │  Feedback    │             │
       │              └──────────────┘             │
       │                                           │
       └───────────────────────────────────────────┘
              Apply geometric edits,
              add/remove shapes, re-optimize
```

* * *

## Phase 1: Differentiable Rendering & Gradient Optimization

### Objective

Prove that we can optimize SVG primitive parameters via backpropagation to match a
target pelican drawing.

### Input

- Target image: drawn pelican (simple, clear silhouette, 128×128 or 256×256)

- Initial geometry: hard-coded pelican structure (body, head, beak, eye)

### Process

**1.1 Hard-coded Initial Structure**

- Define pelican as composition of geometric primitives:

  - Body: ellipse (center, rx, ry, rotation)

  - Head: circle (center, radius)

  - Beak: triangle or polygon (3-4 vertices)

  - Eye: small circle (center, radius)

  - Optional: wing ellipse, legs as lines/capsules

**1.2 Differentiable Rendering**

- Implement soft SDF (signed distance field) functions for each primitive

- Render to 128×128 grayscale image using PyTorch operations

- Use sigmoid-based soft coverage: `coverage = sigmoid(-sdf / tau)`

- Composition uses alpha-over with per-shape alpha = coverage:
  ```
  out = 1.0  # white background
  for shape in back_to_front(shapes):
      cov = sigmoid(-sdf(shape, grid) / tau)  # in [0, 1]
      out = (1 - cov) * out + cov * 0.0  # black ink over white
  ```

- τ (softness) is in “pixel units”: start at τ0 ≈ 1.0 px (≈ 1.0 / max(H, W)) and anneal
  exponentially to τ_min ≈ 0.2 px.
  This avoids vanishing gradients early and improves crispness late.

- Use float32 everywhere; precompute and cache the grid on the target device.

- Clamp SDF magnitude when converting to coverage to avoid extreme logits (e.g., clamp
  |sdf/τ| ≤ 10).

**1.2.1 Coordinate System and Parameterization**

- Coordinates are normalized to [0, 1] on both axes.
  The render grid is H×W with the convention that x=0..1 maps to columns and y=0..1 maps
  to rows.

- Pixels sample at cell centers.
  We precompute a device-resident grid tensor of shape [H, W, 2] with (x, y) in [0, 1].

- Parameterization:

  - Circle: center (cx, cy) ∈ [0, 1]^2 via sigmoid; radius r > 0 via softplus.

  - Ellipse: center (cx, cy); radii (rx, ry) via softplus; rotation θ unconstrained
    (wrapped with sin/cos).

  - Triangle: vertices (x_i, y_i) each in [0, 1] via sigmoid; add a small minimum
    edge-length penalty to avoid degeneracy.

- Z-order is explicit and fixed per shape list (painter’s algorithm).

- Constrained parameters are derived from unconstrained tensors.
  We only optimize unconstrained tensors to keep gradients stable and bounds enforced.

**1.3 Loss Functions**

- **Primary: Image MSE**: `L_image = MSE(rendered, target)`

- **Perceptual loss** (optional): use VGG features or CLIP image embeddings

- **Geometric priors**:

  - Perimeter penalty: discourage sprawling shapes

  - Size bounds: keep shapes within reasonable ranges

  - Overlap penalty: discourage excessive shape overlap

- Add silhouette emphasis: optionally binarize both rendered and target via a
  straight-through estimator (STE) for edges in later training (last 20% steps).

- Regularizers:

  - Parameter L2 on unconstrained tensors to discourage extreme values.

  - Triangle degeneracy penalty: sum over edges of relu(ε - edge_length).

  - On-canvas penalty: penalize centers/vertices with margin outside [0, 1].

- Optionally include SSIM (window=7) to encourage structural similarity; keep its weight
  small relative to MSE.

**1.4 Optimization Loop**

- Initialize parameters (positions, sizes, rotations)

- For N steps (500-2000):

  - Render current geometry

  - Compute loss

  - Backpropagate gradients

  - Update parameters with Adam optimizer

  - Anneal softness parameter τ (coarse → fine)

  - Save intermediate frames every K steps

- Optimizer: Adam(lr=2e-2) with cosine or exponential LR decay to 1e-3.

- Gradient clipping to a global norm (e.g., 1.0) to prevent spikes.

- NaN/Inf guard: detect invalid loss/gradients, back off LR and restore last good
  params.

- Determinism: set seeds for Python, NumPy, and torch; enable
  torch.use_deterministic_algorithms(True) when viable (fall back on non-deterministic
  kernels if necessary on MPS).

**1.5 Output**

- Final SVG with optimized parameters

- PNG render at target resolution

- Training metrics (loss curve, per-shape parameters)

- Animated GIF/MP4 showing optimization progress

- SVG export: map normalized coordinates to viewBox [0, 0, W, H]; write ellipses and
  circles directly; apply rotation via a transform about the shape center; emit polygon
  points in pixel space.

### Success Criteria

- Visual similarity: optimized render closely matches target silhouette

- SVG validity: all primitives export cleanly to standard SVG

- Convergence: loss decreases consistently over training

- Speed: optimization completes in <5 minutes on CPU, <1 minute on GPU

* * *

## Phase 2: LLM-Guided Structural Refinement

### Objective

Use an LLM to make discrete, symbolic improvements to the pelican structure that
gradient descent cannot discover (e.g., “add a pouch under the beak,” “split wing into
two segments”).

### Process

**2.1 Judge Component**

- Multimodal LLM (GPT-4V, Claude 3.5 Sonnet, or Gemini) evaluates:

  - Current SVG code

  - Rendered PNG

  - Target image

  - Optimization metrics from Phase 1

- Produces structured critique:

  - Geometric accuracy: “beak too short,” “body too round”

  - Missing features: “no wing visible,” “pouch missing”

  - Topology issues: “head and body should overlap more”

  - Constraint violations: “too many primitives” or “shapes too complex”

**2.2 LLM Architect**

- Takes judge feedback and proposes concrete geometric changes

- Output format: structured edits in JSON or Python dict
  ```json
  {
    "actions": [
      {
        "type": "modify",
        "shape": "beak",
        "changes": {"length": "+20%", "rotation": "+10deg"}
      },
      {
        "type": "add",
        "shape": "pouch",
        "primitive": "ellipse",
        "init_params": {"cx": 0.45, "cy": 0.55, "rx": 0.08, "ry": 0.06}
      },
      {
        "type": "remove",
        "shape": "eye2"
      }
    ],
    "rationale": "Pelicans have distinctive throat pouches..."
  }
  ```

**2.3 Edit Application**

- Parse LLM-generated edits

- Update Python code that defines pelican structure:

  - Modify initial parameters

  - Add new shape classes

  - Remove shapes

  - Change shape types (e.g., triangle → bezier curve)

**2.4 Re-optimization**

- Run Phase 1 again with updated structure

- Use previous best parameters as warm start where applicable

**2.5 Outer Loop**

- Repeat Judge → LLM → Edit → Optimize for M rounds (3-10)

- Track improvement metrics across rounds

- Stop when judge indicates “no major improvements needed”

### Success Criteria

- LLM successfully identifies structural deficiencies

- Proposed edits are geometrically valid and improve similarity

- System converges to high-quality pelican within 5-10 rounds

- Human evaluation: final SVG looks recognizably pelican-like

* * *

## Technical Implementation Plan

### Technology Stack

**Core Libraries**:

- `torch` (2.1+): differentiable rendering, backprop

- `torchvision`: image preprocessing

- `Pillow`: image I/O

- `numpy`: array operations

**CLI & UI**:

- `rich`: beautiful terminal output, progress bars, live displays

- `argparse`: command-line interface

- `typer` (optional): more ergonomic CLI than argparse

**Visualization**:

- `matplotlib`: plot loss curves, show renders

- `imageio`: create animated GIFs

- (Phase 2) Simple web viewer with auto-refresh

**LLM Integration** (Phase 2):

- `anthropic` or `openai`: API clients

- `instructor` (optional): structured output parsing

- `pydantic`: validate LLM-generated edits

**Dependencies Management**:

- `uv`: fast, modern Python package manager

- `pyproject.toml`: project config, dependencies

### Project Structure

```
differentiable-pelican/
├── pyproject.toml          # uv config, dependencies
├── Makefile               # common commands (lint, test, run)
├── README.md              # overview, quickstart
├── docs/design/pelican-plan.md  # this document
│
├── src/
│   └── differentiable_pelican/
│       ├── __init__.py
│       ├── cli.py         # main CLI entry point
│       ├── geometry.py    # shape parameterizations
│       ├── sdf.py         # differentiable SDF functions
│       ├── renderer.py    # soft rasterization
│       ├── loss.py        # loss functions
│       ├── optimizer.py   # training loop
│       ├── svg_export.py  # convert params → SVG
│       ├── validator.py   # Phase 0: image validation via LLM
│       ├── llm/           # Phase 2 (builds on validator)
│       │   ├── __init__.py
│       │   ├── judge.py        # extends validator with SVG awareness
│       │   ├── architect.py
│       │   └── edit_parser.py
│       └── utils.py       # logging, metrics, viz
│
├── tests/
│   ├── test_validator.py
│   ├── test_phase0_integration.py
│   ├── test_sdf.py
│   ├── test_renderer.py
│   └── test_svg_export.py
│
├── images/
│   ├── pelican-drawing-1.jpg  # reference image (vintage engraving)
│   └── LICENSE                # Source: publicdomainpictures.net
│
├── examples/
│   └── prompts/              # LLM prompt templates
│
└── out/                   # generated outputs (gitignored)
    ├── frames/            # intermediate PNGs
    ├── pelican_final.svg
    ├── pelican_final.png
    ├── optimization.gif
    └── metrics.json
```

### Platform Compatibility Strategy

**Problem**: PyTorch installation varies by platform (CPU vs CUDA vs MPS).

**Solution**: PyTorch GPU wheels (CUDA) are distributed on a separate index and are not
reliably installable as an extra in pyproject.

**Strategy**:

- Declare a generic `torch` dependency (CPU) in pyproject.

- Document GPU install path: users who want CUDA install torch/torchvision with the
  appropriate `--index-url https://download.pytorch.org/whl/cu118` (or current) and then
  `uv sync` without torch changes.

- Runtime device selection stays the same.

**Usage**:
```bash
# Basic Linux cloud (CPU) or M1 Mac (auto-detects MPS)
uv sync

# Cloud GPU (install CUDA wheels first)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv sync
```

**Device selection**: Runtime auto-detect in code:
```python
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

* * *

## CLI Design

### Phase 0 Commands

**Validate rendered image** (for agent self-testing):
```bash
uv run pelican validate-image \
  --image out/test_render.png \
  --target images/pelican-drawing-1.jpg  # optional
  --fix-suggestions                       # optional: more detailed feedback
```

Output is JSON to stdout for programmatic parsing.

### Phase 1 Commands

**Main training command**:
```bash
uv run pelican optimize \
  --target images/pelican-drawing-1.jpg \
  --steps 1000 \
  --resolution 128 \
  --lr 0.02 \
  --save-every 25 \
  --output-dir out/run_001 \
  --live              # show live preview in terminal
```

**Quick test**:
```bash
uv run pelican test-render
# Renders initial hard-coded geometry, no optimization
```

**Export only**:
```bash
uv run pelican export \
  --params out/run_001/final_params.json \
  --output pelican.svg
```

### Phase 2 Commands

**Full refinement loop**:
```bash
uv run pelican refine \
  --target images/pelican-drawing-1.jpg \
  --rounds 5 \
  --phase1-steps 500 \
  --llm-provider anthropic \
  --llm-model claude-3-5-sonnet-20241022 \
  --output-dir out/refined_001
```

**Judge only** (evaluate without re-optimization):
```bash
uv run pelican judge \
  --svg out/run_001/pelican_final.svg \
  --target images/pelican-drawing-1.jpg
```

### CLI Packaging

- Provide an entry point in pyproject:
  ```toml
  [project.scripts]
  pelican = "differentiable_pelican.cli:app"
  ```

- Default outputs land under `out/<run_id>/` with a deterministic run_id unless
  `--run-id` is supplied.

- Add `--seed` and `--device` flags; `--device` overrides auto-detection.

### CLI Output Features (using Rich)

- **Live progress**: animated progress bar with ETA

- **Metrics panel**: live-updating loss, similarity score, step count

- **Mini-preview**: ASCII art or small image in terminal (if supported)

- **Color-coded logs**: warnings (yellow), errors (red), success (green)

- **Tables**: final metrics summary, shape parameters

- **Markdown rendering**: judge feedback formatted nicely

Example output:
```
╭─────────────────── Differentiable Pelican ───────────────────╮
│ Target: images/pelican-drawing-1.jpg                         │
│ Device: mps (Apple M1)                                       │
│ Shapes: 5 primitives (body, head, beak, eye, wing)          │
╰──────────────────────────────────────────────────────────────╯

Optimizing... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸━━━━━  85% 0:00:12

┏━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric      ┃ Value    ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Loss        │ 0.0234   │
│ Image MSE   │ 0.0189   │
│ Prior       │ 0.0045   │
│ Similarity  │ 94.2%    │
└─────────────┴──────────┘

✓ Optimization complete!
  → SVG: out/run_001/pelican_final.svg
  → PNG: out/run_001/pelican_final.png
  → Animation: out/run_001/optimization.gif
```

* * *

## Development Phases

**Note**: All phases designed to work initially on basic Linux CPU with minimal compute
requirements. GPU/MPS optimization comes later as an enhancement.
Each phase includes comprehensive tests and can be committed independently.

**Agent Development**: These phases are designed to be implementable by a coding agent.
Phase 0 establishes an automated image evaluation tool that the agent can use for
self-validation without human intervention.

### Phase 0: Image Validation Tool

**Goal**: Create a simple CLI tool that uses multimodal LLM to evaluate rendered images,
enabling automated validation during development.

**Compute Requirements**: API access only (anthropic/openai), no local compute needed.

**Implementation**:

- [ ] Simple CLI: `pelican validate-image --image path/to/render.png
  [--target path/to/target.jpg]`

- [ ] Multimodal LLM prompt (Claude 3.5 Sonnet or GPT-4V):

  - Describe what shapes/objects are visible in the image

  - If target provided: compare similarity to target

  - Identify obvious issues: blank image, all black/white, no recognizable shapes,
    shapes off-canvas

  - Output structured JSON with boolean flags and text description

- [ ] Output schema (Pydantic):
  ```python
  class ImageValidation(BaseModel):
      is_blank: bool  # Image is all white or all black
      has_shapes: bool  # Contains visible geometric shapes
      shapes_recognizable: bool  # Shapes form coherent object
      resembles_pelican: bool  # Looks vaguely pelican-like
      on_canvas: bool  # Shapes are within image bounds
      description: str  # Free-form description
      issues: list[str]  # List of problems found
      similarity_to_target: float | None  # 0-1 if target provided
  ```

- [ ] Add `--fix-suggestions` flag for more detailed debugging feedback

- [ ] Environment variable or config for API key management

**Unit Tests** (`tests/test_validator.py`):
```python
def test_validate_blank_image():
    # All-white image → is_blank=True, has_shapes=False

def test_validate_simple_circle():
    # Single black circle → has_shapes=True, shapes_recognizable=True

def test_validate_off_canvas():
    # Shapes with centers outside [0,1] → on_canvas=False

def test_validate_with_target():
    # Provide target, get similarity score

def test_validation_schema_parsing():
    # Mock LLM response → valid ImageValidation object
```

**Integration Test** (`tests/test_phase0_integration.py`):
```python
@pytest.mark.skipif(not has_api_key(), reason="No API key")
def test_validator_on_test_renders():
    # Render a few test cases, validate each
    # Check that validator can distinguish good from bad

def test_validator_cli():
    # Run CLI command with mock LLM
    # Verify JSON output structure
```

**Usage by Agent**:

The coding agent can call this tool after each render during development:
```bash
uv run pelican validate-image --image out/test_render.png
```

Parse the JSON output to check:

- If `is_blank=True` or `has_shapes=False` → rendering pipeline broken

- If `on_canvas=False` → parameter initialization or constraints broken

- If `has_shapes=True` but `shapes_recognizable=False` → shapes too degenerate

- Use `description` and `issues` for debugging clues

This enables the agent to iterate on Phase 1A-1B implementations without human
validation of each intermediate image.

**Commit checkpoint**: `✓ Phase 0: Image validation tool for agent self-testing`

**Note on Phase 2 Integration**: If the Phase 0 validator works well, it can be extended
or reused as the Judge component in Phase 2. The validator provides basic image quality
checks and descriptions; the judge adds SVG-aware feedback and structural suggestions.
Consider designing the validator with this evolution in mind (shared prompt templates,
schema foundation).

**Agent Workflow Integration**:

After Phase 0 is complete, the agent should use validation at key checkpoints:

1. **After implementing renderer** (Phase 1A):
   ```bash
   uv run pelican test-render  # Generate initial render
   uv run pelican validate-image --image out/test_render.png
   # Check: has_shapes=True, on_canvas=True
   ```

2. **After each optimization milestone** (Phase 1B):
   ```bash
   uv run pelican optimize --steps 50 --output-dir out/test_opt
   uv run pelican validate-image \
     --image out/test_opt/pelican_final.png \
     --target images/pelican-drawing-1.jpg
   # Check: shapes_recognizable=True, similarity_to_target > 0.3
   ```

3. **Before committing each phase**: Run validation on representative outputs to ensure
   no regressions.

The agent can parse the JSON output programmatically and make decisions:

- If validation fails basic checks, debug before proceeding

- Use `description` and `issues` fields as hints for what to fix

- Compare `similarity_to_target` across runs to verify optimization is improving

This creates a tight feedback loop without requiring human review of intermediate
images.

* * *

### Phase 1A: Foundation

**Goal**: Render hard-coded pelican, no optimization yet

**Compute Requirements**: CPU-only, no heavy dependencies, fast (<1 second render),
64×64 resolution

**Implementation**:

- [ ] Project setup: `pyproject.toml`, directory structure, uv config

- [ ] Implement SDF functions: circle, ellipse, triangle

- [ ] Implement soft rasterizer: grid generation, coverage computation

- [ ] Hard-code pelican geometry (5-6 shapes)

- [ ] Render to PNG at low resolution (64×64)

- [ ] SVG export: convert parameters → valid SVG markup

- [ ] CLI: `pelican test-render`

**Unit Tests** (`tests/test_sdf.py`):
```python
def test_sdf_circle_at_origin():
    # Point at center should have negative distance (inside)

def test_sdf_circle_outside():
    # Point far away should have positive distance

def test_sdf_ellipse_on_boundary():
    # Point on ellipse boundary should be ~0

def test_sdf_triangle_vertices():
    # Triangle vertices should be on boundary

def test_coverage_sigmoid_range():
    # Coverage should be in [0, 1]

def test_triangle_sdf_gradients_exist():
    # Finite-difference vs autograd check on a simple triangle near an edge

def test_params_are_clamped_in_range():
    # Derived params (cx, cy ∈ [0,1], r>0) respect transforms
```

**Unit Tests** (`tests/test_renderer.py`):
```python
def test_make_grid_shape():
    # Grid should be HxWx2

def test_make_grid_normalized():
    # Grid values should be in [0, 1]

def test_render_single_circle():
    # White background with black circle

def test_render_composite_shapes():
    # Multiple shapes composite correctly

def test_render_deterministic():
    # Same params → same output
```

**Unit Tests** (`tests/test_svg_export.py`):
```python
def test_export_svg_valid_xml():
    # Output parses as valid XML

def test_export_svg_contains_shapes():
    # Contains expected <ellipse>, <circle>, <polygon>

def test_svg_viewbox_correct():
    # ViewBox dimensions match resolution
```

**Integration Test** (`tests/test_phase1a_integration.py`):
```python
def test_end_to_end_render_and_export():
    # Hard-coded pelican → render PNG → export SVG
    # Check files exist and have reasonable sizes

def test_cli_test_render():
    # Run CLI command, check output files created

def test_rendered_image_not_blank():
    # PNG should have some dark pixels (shapes visible)
```

**Commit checkpoint**: `✓ Phase 1A: Basic rendering working on CPU`

* * *

### Phase 1B: Optimization Loop

**Goal**: Optimize geometry to match target image

**Compute Requirements**: Still CPU-friendly, 64×64 resolution, short runs (100-200
steps), should complete in <30 seconds

**Implementation**:

- [ ] Load target image, preprocess (grayscale, resize)

- [ ] Implement loss functions: MSE, perimeter prior

- [ ] Training loop: forward render, compute loss, backprop, update

- [ ] Softness annealing schedule

- [ ] Save intermediate frames (optional, can disable for speed)

- [ ] CLI: `pelican optimize --steps 100` (default to fast test)

- [ ] Metrics logging: print loss every 10 steps

**Unit Tests** (`tests/test_loss.py`):
```python
def test_mse_loss_zero_for_identical():
    # Same image → loss = 0

def test_mse_loss_positive_for_different():
    # Different images → loss > 0

def test_mse_loss_has_gradient():
    # Loss should backprop to params

def test_perimeter_prior_larger_for_big_shapes():
    # Bigger shapes → higher prior

def test_perimeter_prior_gradient_nonzero():
    # Prior should have gradients for backprop

def test_nan_guard_works():
    # Force extreme params; verify training recovers or aborts cleanly
```

**Unit Tests** (`tests/test_optimizer.py`):
```python
def test_optimizer_step_changes_params():
    # After one step, params should change

def test_optimizer_reduces_loss_simple_case():
    # For trivial case (slightly offset circle), loss should decrease

def test_softness_annealing():
    # Tau should decrease over steps

def test_all_params_receive_gradients():
    # All shape params should have non-None .grad
```

**Integration Test** (`tests/test_phase1b_integration.py`):
```python
def test_optimize_circle_to_circle():
    # Target: single circle, init: slightly offset circle
    # Should converge to near-zero loss in <50 steps

def test_optimize_converges_not_diverges():
    # Run 100 steps, final loss < initial loss

def test_optimization_produces_valid_svg():
    # After optimization, exported SVG is valid

def test_cli_optimize_fast():
    # Run with --steps 20, should complete in <10 seconds on CPU

def test_optimization_improves_similarity():
    # Final render should be closer to target than initial
```

**Commit checkpoint**: `✓ Phase 1B: Gradient optimization working on CPU`

* * *

### Reproducibility

- Record seeds and environment (torch, torchvision, device) in metrics.json.

- Save initial params, best params, and LR/τ schedules for exact reruns.

* * *

### Phase 1C: Full Pelican Optimization

**Goal**: Complete optimization with all features, ready to scale to GPU

**Compute Requirements**: Works on CPU (slow) but optimized for GPU/MPS when available,
128×128 resolution

**Implementation**:

- [ ] Device auto-detection (CPU/CUDA/MPS)

- [ ] Increase default resolution to 128×128

- [ ] Rich CLI with live progress, metrics

- [ ] Generate optimization animation (GIF)

- [ ] Metrics export (JSON with loss history, final params)

- [ ] Handle edge cases: shapes going off-canvas, numerical instability

- [ ] Add `--device` flag to override auto-detection

- [ ] Add `--resolution` flag (default 128)

- [ ] Benchmark different configurations

**Unit Tests** (`tests/test_device.py`):
```python
def test_pick_device_returns_valid_device():
    # Should return cpu/cuda/mps

def test_tensors_on_correct_device():
    # After moving to device, tensors should be there

def test_device_override_flag():
    # --device cpu should force CPU even if GPU available
```

**Unit Tests** (`tests/test_cli.py`):
```python
def test_cli_parse_args():
    # Parse valid args correctly

def test_cli_defaults_sensible():
    # Default args work without errors

def test_cli_invalid_args_error():
    # Bad args → helpful error message

def test_cli_help_message():
    # --help produces useful output
```

**Integration Test** (`tests/test_phase1c_integration.py`):
```python
def test_full_optimization_cpu_128():
    # Full 500 step run on CPU at 128×128, verify convergence

def test_full_optimization_produces_all_outputs():
    # Check SVG, PNG, metrics.json, frames/ all created

def test_optimization_animation_created():
    # GIF file exists and has >10 frames

def test_metrics_json_structure():
    # JSON has expected fields, loss history is list

def test_final_svg_valid():
    # Final SVG parses and renders correctly
```

**Performance Test** (`tests/test_performance.py`):
```python
def test_optimization_speed_cpu_64():
    # 100 steps at 64×64 should complete in <30 seconds on basic CPU

def test_optimization_speed_cpu_128():
    # 100 steps at 128×128 should complete in <2 minutes on basic CPU

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_optimization_speed_gpu():
    # If GPU available, 100 steps at 128×128 should be <10 seconds

def test_memory_usage_reasonable():
    # Peak memory <1GB for 128×128, <2GB for 256×256
```

**Commit checkpoint**: `✓ Phase 1C: Production-ready optimizer with GPU support`

* * *

### Phase 2A: Judge

**Goal**: LLM evaluates optimized pelican

**Compute Requirements**: Minimal compute (inference only), needs API key

**Note**: This can build on or reuse the Phase 0 validator infrastructure.
The judge extends validation with SVG-aware structural feedback.

**Implementation**:

- [ ] Extend Phase 0 validator or implement new judge prompt template (Jinja2 or
  f-strings)

- [ ] Call multimodal LLM with SVG code + rendered image + target

- [ ] Parse structured feedback (JSON schema with Pydantic)

- [ ] CLI: `pelican judge --svg out/pelican.svg --target examples/target.png`

- [ ] Handle API errors gracefully (rate limits, network issues, timeouts)

- [ ] Add `--llm-provider` flag (anthropic/openai/gemini)

- [ ] Add `--dry-run` mode (show prompt without calling API)

- [ ] Enforce strict response schema with Pydantic; retry on parse failure with a
  “respond with only JSON” system reminder and a shorter temperature.

- [ ] Provide a `--dry-run` flag to print the prompt and exit without API calls.

**Unit Tests** (`tests/llm/test_judge.py`):
```python
def test_judge_prompt_renders():
    # Prompt template fills correctly with params

def test_judge_prompt_includes_svg_code():
    # Prompt contains the actual SVG text

def test_judge_response_parsing():
    # Mock LLM response → parsed feedback object

def test_judge_feedback_schema_validation():
    # Valid JSON passes, invalid raises error

def test_judge_handles_malformed_response():
    # LLM returns bad JSON → graceful fallback with retry
```

**Unit Tests** (`tests/llm/test_prompts.py`):
```python
def test_prompt_includes_target_description():
    # Describes what target image shows

def test_prompt_includes_shape_descriptions():
    # Describes what each SVG shape represents

def test_prompt_length_reasonable():
    # Not too long (API token limits)
```

**Integration Test** (`tests/test_phase2a_integration.py`):
```python
def test_judge_end_to_end_mock():
    # Use mock LLM, verify full flow works

def test_judge_cli_with_real_files():
    # Point to actual SVG and target, get feedback

def test_judge_dry_run():
    # --dry-run shows prompt without API call

@pytest.mark.skipif(not has_api_key(), reason="No API key")
@pytest.mark.slow
def test_judge_real_llm():
    # Actually call LLM, check response is reasonable

def test_judge_handles_api_error():
    # Simulate API error → graceful error message
```

**Commit checkpoint**: `✓ Phase 2A: Judge component working`

* * *

### Phase 2B: Architect & Edit Application

**Goal**: LLM proposes changes, system applies them

**Compute Requirements**: Minimal compute, mainly JSON parsing and code generation

**Implementation**:

- [ ] Implement architect prompt (takes judge feedback)

- [ ] Define edit schema: add/remove/modify operations (Pydantic models)

- [ ] Edit parser: JSON → updated geometry Python code or params

- [ ] Validation: ensure edits produce valid geometry

- [ ] Re-optimization with warm start (reuse params where applicable)

- [ ] CLI: `pelican architect --feedback judge_output.json`

- [ ] Add safeguards: bounds checking, shape count limits

- [ ] Bounds checking: all edits validated against [0,1] position ranges and positive
  sizes before application; reject or clip with logged rationale.

- [ ] Edit provenance: store the JSON edit set applied per round in the run directory
  for auditability.

**Unit Tests** (`tests/llm/test_architect.py`):
```python
def test_architect_prompt_includes_feedback():
    # Judge feedback appears in architect prompt

def test_parse_modify_edit():
    # JSON modify action → updated params

def test_parse_add_shape_edit():
    # JSON add action → new shape in geometry

def test_parse_remove_shape_edit():
    # JSON remove action → shape excluded

def test_validate_edit_bounds():
    # Edit with out-of-range values → rejected

def test_validate_edit_complete():
    # Edit missing required fields → rejected

def test_multiple_edits_applied_correctly():
    # List of edits applied in order
```

**Unit Tests** (`tests/llm/test_edit_parser.py`):
```python
def test_apply_edits_to_params():
    # Edits applied correctly to PelicanParams

def test_edits_preserve_valid_geometry():
    # After edits, shapes still render without errors

def test_edits_maintain_shape_order():
    # Rendering order (z-index) preserved

def test_percentage_change_parsing():
    # "increase by 20%" parsed correctly
```

**Integration Test** (`tests/test_phase2b_integration.py`):
```python
def test_one_round_judge_architect_optimize():
    # Complete cycle: optimize → judge → architect → re-optimize
    # Check system doesn't crash

def test_architect_edits_applied_correctly():
    # Mock edit "increase beak length 20%"
    # Verify beak params actually increased

def test_invalid_edit_rejected_gracefully():
    # LLM suggests bad edit → system catches, warns, continues

def test_warm_start_uses_previous_params():
    # Re-optimization starts from edited params, not random init
```

**Commit checkpoint**: `✓ Phase 2B: Architect and edit application working`

* * *

### Phase 2C: Full Refinement Loop

**Goal**: Multi-round optimization with LLM feedback

**Compute Requirements**: Can run on CPU but slow (many minutes); recommend GPU for
production use

**Implementation**:

- [ ] Implement multi-round loop

- [ ] Track improvement metrics across rounds

- [ ] Stopping criteria (convergence or max rounds)

- [ ] CLI: `pelican refine --rounds 3 --target examples/target.png`

- [ ] Generate report: all rounds’ outputs, judge feedback, metrics

- [ ] Add `--max-rounds` flag (default 5)

- [ ] Add `--convergence-threshold` flag (stop if improvement < threshold)

- [ ] Save intermediate state after each round

- [ ] Early stop if improvement < ε for K consecutive rounds; also stop on judge “no
  major improvements” signal.

- [ ] Rollback: if post-edit optimization degrades beyond a threshold vs pre-edit,
  revert the edit set and mark the round as rejected.

**Unit Tests** (`tests/test_refine_loop.py`):
```python
def test_refine_loop_runs_n_rounds():
    # With --rounds 3, should do exactly 3 iterations (or fewer if converged)

def test_refine_loop_stops_on_convergence():
    # If loss delta < threshold, stop early

def test_refine_loop_tracks_metrics():
    # Metrics dict has entry for each round

def test_refine_loop_handles_llm_failure():
    # LLM call fails → retry, skip round, or use fallback

def test_refine_loop_saves_intermediate_state():
    # Can resume from any round
```

**Integration Test** (`tests/test_phase2c_integration.py`):
```python
def test_full_refine_2_rounds_mock_llm():
    # Use mock LLM for 2 rounds
    # Check all intermediate outputs created

def test_refine_report_generation():
    # Report JSON has correct structure
    # Contains all rounds' feedback and metrics

def test_refine_rollback_on_divergence():
    # If loss increases dramatically, rollback edit

@pytest.mark.skipif(not has_api_key(), reason="No API key")
@pytest.mark.slow
def test_full_refine_real_llm():
    # Actually run 2 rounds with real LLM
    # Smoke test for real usage
```

**Performance Test** (`tests/test_refine_performance.py`):
```python
def test_refine_2_rounds_cpu_time():
    # 2 rounds with 200 steps each should complete eventually on CPU
    # (May be slow, but shouldn't hang)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_refine_3_rounds_gpu_reasonable_time():
    # 3 rounds on GPU should be <5 minutes

def test_refine_memory_stable_across_rounds():
    # Memory doesn't grow unbounded (no leaks)
```

**Commit checkpoint**: `✓ Phase 2C: Full refinement pipeline working`

* * *

### Phase 2D: Robustness & Edge Cases

**Goal**: Handle failures, edge cases, improve reliability for production use

**Compute Requirements**: Minimal compute for testing, mainly error handling

**Implementation**:

- [ ] Add edit validation safeguards (strict bounds, type checking)

- [ ] Rollback mechanism if optimization diverges after edit

- [ ] Better error messages for common failures

- [ ] Add `--verbose` flag for detailed logging

- [ ] Documentation: troubleshooting guide, known limitations

- [ ] CI/CD integration: run tests on commit

**Unit Tests** (`tests/test_error_handling.py`):
```python
def test_invalid_target_image_path():
    # Nonexistent file → clear error message

def test_corrupted_target_image():
    # Bad image file → handled gracefully

def test_optimization_divergence_detected():
    # Loss goes to NaN → stop and report

def test_edit_validation_catches_bad_values():
    # Edit with radius=-5 → rejected with reason

def test_llm_timeout_handled():
    # Slow LLM response → timeout, retry with backoff

def test_network_error_handled():
    # API unreachable → clear error, don't crash
```

**Integration Test** (`tests/test_edge_cases.py`):
```python
def test_target_all_black():
    # Edge case: target is solid black

def test_target_all_white():
    # Edge case: target is solid white

def test_target_very_small():
    # Target is 16×16 → should upscale sensibly

def test_target_very_large():
    # Target is 1024×1024 → should downsample

def test_too_many_optimization_steps():
    # --steps 100000 → should handle without crash or excessive memory

def test_llm_suggests_remove_all_shapes():
    # Degenerate case → catch and prevent (need at least 1 shape)

def test_llm_suggests_100_shapes():
    # Too many shapes → warn and cap at reasonable limit
```

**Commit checkpoint**: `✓ Phase 2D: Robust, production-ready system`

* * *

## Rationale & Design Decisions

### Why Soft SDFs Instead of Differentiable SVG Libraries?

**Considered alternatives**:

- `diffvg`: Differentiable vector graphics library

- `LIVE`: Learned vectorization

- Direct SVG manipulation libraries

**Chosen approach: Custom soft SDF rasterizer**

**Reasons**:

1. **Simplicity**: ~100 lines of PyTorch, no external C++ dependencies

2. **Control**: Full transparency over gradient flow

3. **Portability**: Pure Python/PyTorch works everywhere (CPU/GPU/MPS)

4. **Minimalism**: We only need circles, ellipses, triangles initially

5. **Educational**: Easy to understand and extend

**Trade-offs**:

- Limited to simple primitives (no Bezier curves, complex paths)

- Resolution-dependent rendering (need to rasterize at sufficient resolution)

- Soft edges (τ parameter) vs crisp vector graphics

### Why Image-Based Loss Instead of CLIP Semantic Loss?

**Original brainstorm used CLIP** (`1 - cos(CLIP(render), CLIP("pelican"))`).

**Phase 1 uses image MSE** (`MSE(render, target_image)`).

**Reasons**:

1. **Geometric precision**: Image loss directly optimizes shape accuracy

2. **No ambiguity**: Target image is ground truth, CLIP is fuzzy

3. **Faster convergence**: Pixel-level gradients are stronger signal

4. **Debugging**: Easy to visualize error (diff image)

5. **Phase 2 adds semantic reasoning**: LLM judge provides high-level feedback

**Future**: Could add CLIP loss as auxiliary term for style/realism.

### Why LLM in Phase 2 Instead of Pure Gradient Methods?

**Gradient descent limitations**:

- Cannot discover topology changes (add/remove shapes)

- Gets stuck in local minima (wrong structure)

- No semantic understanding (doesn’t know “pelicans have pouches”)

**LLM advantages**:

- Symbolic reasoning: “this shape represents a beak”

- Structural edits: “split wing into two segments”

- Domain knowledge: “pelicans have long beaks and throat pouches”

- Escape local minima: propose radical changes

**Hybrid approach is optimal**: Gradients optimize continuous params, LLM handles
discrete structure.

### Why Rich CLI Instead of Web UI?

**Phase 1 uses terminal UI** (Rich library).

**Reasons**:

1. **Faster development**: No frontend code, backend API, or deployment

2. **Scriptable**: Easy to run in CI, batch jobs, remote servers

3. **Universal**: Works over SSH, cloud instances, headless environments

4. **Focus on core**: Spend time on algorithms, not UI polish

**Future**: Phase 2+ could add lightweight web viewer for browsing results.

* * *

## Success Metrics

### Phase 1 (Quantitative)

- Image MSE between optimized and target: < 0.05

- Perceptual similarity (SSIM): > 0.85

- Optimization time on CPU: < 5 minutes

- Optimization time on GPU: < 1 minute

- Convergence rate: Loss decreases monotonically 90%+ of time

### Phase 2 (Qualitative)

- Human evaluators prefer Phase 2 output over Phase 1: >70% of time

- Judge feedback is actionable: >80% of critiques lead to improvements

- LLM-proposed edits are valid: >90% compile and run

- System converges within 5 rounds: >80% of test cases

### Overall

- Generated SVG is valid (passes SVG validator)

- SVG is minimal (<10 shapes for simple pelican, <20 for detailed)

- Output is recognizable as pelican by humans: >95%

* * *

## Open Questions & Future Directions

### Open Questions

1. **Initial structure sensitivity**: How much does hard-coded starting point matter?

2. **Target image requirements**: Does it work with photos or only cartoons?

3. **Shape budget**: Fixed number of primitives vs dynamic (LLM adds/removes)?

4. **Optimization stability**: How often does it diverge or produce degenerate shapes?

5. **LLM edit safety**: How to prevent LLM from suggesting invalid/broken edits?

### Future Extensions

1. **Bezier curves**: Add support for smooth curves (beak, pouch, wing)

2. **Color**: Extend to RGB rendering, optimize fill colors

3. **Multiple targets**: Optimize for multiple reference images simultaneously

4. **Animation**: Optimize keyframes, generate animated SVG

5. **Other animals**: Generalize to arbitrary subjects (not just pelicans)

6. **Interactive editing**: Human-in-the-loop GUI for adjusting shapes

7. **Stylization**: Add style transfer loss (match artistic style, not just geometry)

* * *

## References & Inspiration

- **Differentiable rendering**: Neural style transfer, DiffVG, LIVE

- **SDF rendering**: Inigo Quilez’s articles on signed distance functions

- **LLM code generation**: AlphaCode, GPT-4 for competitive programming

- **Hybrid optimization**: Combining gradient descent with symbolic search

- **Triangle SDFs**: Inigo Quilez, “Distance functions” and “Triangles” SDF notes for
  robust triangle SDFs

- **Compositing**: SIGGRAPH notes on alpha compositing; Porter-Duff “over” operator

* * *

## Appendix: Example Target Images

Ideal target images for Phase 1:

- Simple cartoon pelican (solid colors, clear silhouette)

- Minimalist icon or logo style

- High contrast (white background, black pelican)

- 128×128 to 512×512 pixels

- Clean edges (not photo-realistic or highly textured)

Examples to find:

- Pelican emoji (🐦)

- Pelican clipart from free icon sites

- Hand-drawn cartoon pelican

- Line art pelican

Avoid:

- Photos (too much detail, texture, lighting)

- Complex backgrounds

- Multiple pelicans

- Occluded or partial views

**Included Reference Image**:

- `images/pelican-drawing-1.jpg`: Vintage engraving from Public Domain Pictures

- Source: https://www.publicdomainpictures.net/en/view-image.php?image=439798

- Note: This image has detailed crosshatching and texture.
  For Phase 1A-1B testing, consider preprocessing:

  - Downsample and crop to square aspect ratio (e.g., 256×256 or 512×512)

  - Convert to grayscale

  - Optionally threshold or posterize to simplify texture

  - The white background makes it suitable for the black-on-white rendering approach
