# Feature: Differentiable Pelican in Rust/Burn with Wasm Browser Deployment

**Date:** 2026-02-15

**Author:** Claude (with Joshua Levy)

**Status:** Draft

## Overview

Port the differentiable-pelican rendering and optimization pipeline from Python/PyTorch to Rust using the [Burn](https://github.com/tracel-ai/burn) deep learning framework, then compile it to WebAssembly for browser-local execution. This creates a self-contained browser demo where users upload a target image and watch gradient-based SVG optimization run live — no server, no Python, no PyTorch.

The Rust core compiles to a ~2 MB Wasm module exposed to JavaScript via `wasm-bindgen`. A minimal TypeScript/HTML frontend handles the UI. The same Rust code also runs natively (CLI) by swapping Burn backends.

## Goals

- Reimplement the full differentiable rendering pipeline (SDFs, coverage, compositing, loss, Adam optimizer) in Rust/Burn
- Compile to Wasm and run gradient-based optimization (forward + backward + parameter updates) entirely in the browser
- Provide a minimal web UI: upload target image, watch optimization progress, download final SVG
- Demonstrate Burn's autograd (`Autodiff<NdArray>`) working in browser Wasm — a first for the Burn ecosystem
- Keep the Rust code backend-agnostic so it can run natively (CUDA/CPU) or in browser (Wasm) with no code changes
- Produce a minimal but complete end-to-end spike first, then refine

## Non-Goals

- Full feature parity with the Python version in the first pass (no LLM integration, no greedy refinement loop initially)
- Mobile-optimized UI
- WebGPU (`Autodiff<Wgpu>`) backend in the initial spike (CPU/NdArray first, GPU later)
- Production-quality web application (this is a proof-of-concept demo)
- Replacing the Python version (it continues to exist for CLI/server use)

## Background

The [research brief](../../research/research-2026-02-15-python-wasm-feasibility.md) established that:

1. **PyTorch cannot run in Wasm** — no wheels, 200 MB+ size, complex native dependencies
2. **Burn is the strongest Rust alternative** — full autograd, PyTorch-like API, proven Wasm deployment for inference, ~357 KB framework overhead
3. **The computation is simple enough to port** — 48 scalar parameters, shallow computation graph (SDFs → sigmoid → compositing → loss), Adam optimizer
4. **Nobody has demonstrated Burn training in Wasm** — all existing demos are inference-only. This would be a first.

The differentiable-pelican algorithm:
1. Define shapes as differentiable parameters (positions, sizes, rotations, intensities)
2. Compute signed distance fields (SDFs) for each shape on a pixel grid
3. Convert SDFs to soft coverage via `sigmoid(-sdf/tau)`
4. Composite shapes back-to-front via Porter-Duff alpha-over
5. Compute multi-component loss (MSE + SSIM + edge + priors)
6. Backpropagate gradients through the entire pipeline
7. Update parameters with Adam optimizer
8. Repeat for ~500 steps

## Design

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Rust Crate: differentiable-pelican-core                        │
│                                                                  │
│  ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │  geometry.rs │  │  sdf.rs  │  │render.rs │  │  loss.rs    │  │
│  │             │  │          │  │          │  │             │  │
│  │  ShapeKind  │  │ sdf_*()  │  │ render() │  │ mse_loss()  │  │
│  │  Circle     │──│ circle   │──│ compose  │──│ edge_loss() │  │
│  │  Ellipse    │  │ ellipse  │  │ alpha    │  │ ssim_loss() │  │
│  │  Triangle   │  │ triangle │  │ over     │  │ priors()    │  │
│  └─────────────┘  └──────────┘  └──────────┘  └─────────────┘  │
│                                                      │          │
│  ┌──────────────┐  ┌──────────────────────────────────┘         │
│  │ optimizer.rs │  │                                            │
│  │              │──┘                                            │
│  │ Adam + loop  │     ┌──────────────┐                          │
│  │ tau anneal   │     │  wasm_api.rs │  ← wasm-bindgen exports  │
│  │ LR schedule  │     │              │                          │
│  │ grad clip    │     │  init()      │                          │
│  └──────────────┘     │  step()      │                          │
│                       │  get_image() │                          │
│  Backend: generic B   │  get_svg()   │                          │
│  (NdArray for Wasm,   │  get_loss()  │                          │
│   CUDA for native)    └──────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
        │ wasm-bindgen (typed arrays, strings)
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Browser (TypeScript / HTML)                                     │
│                                                                  │
│  - Image upload (drag-and-drop or file picker)                   │
│  - Canvas display (rendered image + target side by side)          │
│  - Progress bar + loss chart                                     │
│  - SVG download button                                           │
│  - Start/stop/reset controls                                     │
└──────────────────────────────────────────────────────────────────┘
```

### Rust Module Design

#### `geometry.rs` — Shape Definitions

Port the Python `geometry.py` shape hierarchy to a Rust enum with Burn Module derive:

```rust
use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};

/// Raw (unconstrained) parameters for a circle.
/// Constrained via sigmoid (center, intensity) and softplus (radius).
#[derive(Module, Debug, Clone)]
pub struct CircleShape<B: Backend> {
    pub cx_raw: Param<Tensor<B, 1>>,
    pub cy_raw: Param<Tensor<B, 1>>,
    pub radius_raw: Param<Tensor<B, 1>>,
    pub intensity_raw: Param<Tensor<B, 1>>,
}

// Similarly for EllipseShape (6 params) and TriangleShape (7 params)

/// A shape variant — closed enum, exhaustive match.
#[derive(Module, Debug, Clone)]
pub enum ShapeKind<B: Backend> {
    Circle(CircleShape<B>),
    Ellipse(EllipseShape<B>),
    Triangle(TriangleShape<B>),
}

/// The full model: a list of shapes.
#[derive(Module, Debug, Clone)]
pub struct PelicanModel<B: Backend> {
    pub shapes: Vec<ShapeKind<B>>,
}
```

Key design decisions:
- **Enum over trait objects.** The set of shape types is closed (3 variants). Rust enums give exhaustive matching, no heap allocation for dispatch, and derive `Module` automatically.
- **Raw parameters.** All parameters stored unconstrained (full real line). Constraining happens at use time: `sigmoid(cx_raw)` for positions in `[0,1]`, `softplus(radius_raw)` for positive radii.
- **Single `PelicanModel` struct.** Burn's optimizer API requires a single `Module`. Wrapping all shapes in one struct satisfies this.

#### `sdf.rs` — Signed Distance Fields

Pure tensor math, no learned parameters:

```rust
/// SDF for a circle: distance to boundary (negative inside).
pub fn sdf_circle<B: Backend>(
    points: Tensor<B, 3>,  // [H, W, 2]
    center: Tensor<B, 1>,  // [2]
    radius: Tensor<B, 1>,  // [1]
) -> Tensor<B, 2> {        // [H, W]
    let diff = points - center.unsqueeze(0).unsqueeze(0);  // broadcast
    let dist = diff.powf_scalar(2.0).sum_dim(2).sqrt();
    dist - radius
}

/// Soft coverage from SDF via sigmoid.
pub fn coverage_from_sdf<B: Backend>(
    sdf: Tensor<B, 2>,  // [H, W]
    tau: f64,
) -> Tensor<B, 2> {     // [H, W] in [0, 1]
    let scaled = sdf.neg().div_scalar(tau).clamp(-10.0, 10.0);
    scaled.sigmoid()
}
```

All three SDF functions (circle, ellipse, triangle) are straightforward tensor operations: subtract, norm, dot products, clamp. The ellipse uses the same approximate SDF as the Python version (normalized radii, no Newton iteration). The triangle SDF uses edge-distance with cross-product sign determination.

#### `renderer.rs` — Differentiable Compositing

```rust
pub fn render<B: Backend>(
    model: &PelicanModel<B>,
    height: usize,
    width: usize,
    tau: f64,
    device: &B::Device,
) -> Tensor<B, 2> {  // [H, W]
    let grid = make_grid(height, width, device);  // [H, W, 2]
    let mut canvas = Tensor::ones([height, width], device);  // white

    for shape in &model.shapes {
        let (sdf, intensity) = match shape {
            ShapeKind::Circle(c) => (sdf_circle(&grid, ...), c.intensity()),
            ShapeKind::Ellipse(e) => (sdf_ellipse(&grid, ...), e.intensity()),
            ShapeKind::Triangle(t) => (sdf_triangle(&grid, ...), t.intensity()),
        };
        let coverage = coverage_from_sdf(sdf, tau);
        // Porter-Duff alpha-over: canvas = (1-a)*canvas + a*color
        canvas = canvas.clone() * (coverage.clone().neg() + 1.0)
               + coverage * intensity;
    }
    canvas
}
```

#### `loss.rs` — Multi-Component Loss

- **MSE**: `(rendered - target).powf(2.0).mean()`
- **Edge loss**: Sobel filtering via manual shifted-tensor differences (avoiding conv2d for simplicity in the spike). For a 3x3 Sobel kernel on a `[H,W]` image, compute horizontal and vertical gradients by slicing and subtracting adjacent rows/columns.
- **SSIM**: The full Wang et al. formulation can be deferred to Phase 2. For the spike, MSE + edge loss are sufficient to demonstrate end-to-end optimization.
- **Priors**: Perimeter, degeneracy, on-canvas — all are simple scalar arithmetic on shape parameters, straightforward to port.

#### `optimizer.rs` — Training Loop

```rust
pub fn optimize<B: AutodiffBackend>(
    model: PelicanModel<B>,
    target: Tensor<B, 2>,     // [H, W]
    config: OptimConfig,
) -> (PelicanModel<B>, Vec<f64>) {
    let optim_config = AdamConfig::new().with_learning_rate(config.lr);
    let mut optim = optim_config.init();
    let grid = make_grid(config.resolution, config.resolution, &config.device);

    let mut loss_history = Vec::new();
    let mut model = model;

    for step in 0..config.steps {
        let tau = anneal_tau(step, config.steps, config.tau_start, config.tau_end);

        // Forward pass
        let rendered = render(&model, config.resolution, config.resolution, tau, &config.device);
        let loss = total_loss(&rendered, &target, &model, &config.loss_weights);

        // Backward pass
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        // Gradient clipping (manual)
        // ... compute global norm, scale if > max_norm ...

        // Update
        model = optim.step(config.lr_for_step(step), model, grads);
        loss_history.push(loss_value);
    }
    (model, loss_history)
}
```

Key differences from Python:
- **Burn's optimizer API** takes a `Module` and returns an updated `Module` (functional style, no mutation). This is actually cleaner than PyTorch's in-place mutation.
- **Cosine LR annealing** is computed manually: `lr * 0.5 * (1 + cos(pi * step / total_steps))`.
- **Gradient clipping** is computed manually: global norm of all gradients, scale factor if above threshold.

#### `wasm_api.rs` — Browser Interface

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Optimizer {
    model: PelicanModel<NdArray>,
    target: Tensor<NdArray, 2>,
    optim: Adam<NdArray>,
    step: usize,
    config: OptimConfig,
}

#[wasm_bindgen]
impl Optimizer {
    /// Initialize with target image (RGBA pixels as flat u8 array).
    #[wasm_bindgen(constructor)]
    pub fn new(target_pixels: &[u8], width: u32, height: u32, resolution: u32) -> Self { ... }

    /// Run one optimization step. Returns current loss.
    pub fn step(&mut self) -> f64 { ... }

    /// Run N steps, calling back with progress.
    pub fn run_steps(&mut self, n: u32) -> Vec<f64> { ... }

    /// Get current rendered image as RGBA u8 array.
    pub fn get_rendered_image(&self) -> Vec<u8> { ... }

    /// Get current SVG string.
    pub fn get_svg(&self) -> String { ... }

    /// Get current loss breakdown as JSON.
    pub fn get_loss_json(&self) -> String { ... }

    /// Get current step number.
    pub fn current_step(&self) -> u32 { ... }
}
```

The wasm-bindgen boundary passes:
- **In**: Target image as `&[u8]` (RGBA pixels), configuration as primitive values
- **Out**: Rendered image as `Vec<u8>` (RGBA pixels), SVG as `String`, loss as `f64`
- No complex Rust types cross the boundary.

### Web Frontend

Minimal HTML + TypeScript (or plain JavaScript). No framework required for the spike.

```
web/
  index.html        # Single-page app
  main.ts           # Load Wasm, orchestrate UI
  style.css         # Minimal styling
  pkg/              # wasm-pack output (gitignored, built)
```

The UI flow:
1. User loads page → Wasm module initializes
2. User uploads/drops a target image → JavaScript reads pixels, passes to `Optimizer::new()`
3. User clicks "Start" → JavaScript calls `optimizer.step()` in a `requestAnimationFrame` loop
4. Each frame: render the current image to a `<canvas>`, update loss display
5. User clicks "Stop" → pause the loop
6. User clicks "Download SVG" → `optimizer.get_svg()` → download as file

### Project Structure

```
differentiable-pelican/
  Cargo.toml                    # Workspace root
  crates/
    pelican-core/               # The differentiable rendering library
      Cargo.toml
      src/
        lib.rs
        geometry.rs
        sdf.rs
        renderer.rs
        loss.rs
        optimizer.rs
        svg.rs                  # SVG export
    pelican-wasm/               # Wasm bindings (thin wrapper)
      Cargo.toml
      src/
        lib.rs                  # wasm_api.rs content
    pelican-cli/                # (Optional) Native CLI
      Cargo.toml
      src/
        main.rs
  web/
    index.html
    main.ts
    style.css
    package.json                # For wasm-pack + vite/esbuild
  src/                          # Existing Python source (unchanged)
    differentiable_pelican/
      ...
```

The Rust workspace is additive — it lives alongside the existing Python code. The Python version continues to work as-is.

### Dependencies (Cargo.toml)

```toml
[workspace]
members = ["crates/pelican-core", "crates/pelican-wasm", "crates/pelican-cli"]

# pelican-core dependencies:
burn = { version = "0.20", features = ["ndarray"] }
burn-autodiff = "0.20"
rand = "0.8"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
image = "0.25"    # For image loading (native CLI only)

# pelican-wasm additional dependencies:
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console"] }
```

## Implementation Plan

### Phase 1: Minimal End-to-End Spike (Core Loop in Rust, Running Natively)

Goal: Prove that the differentiable rendering pipeline works in Rust/Burn with autograd. Run natively first (not Wasm yet).

- [ ] Set up Rust workspace with `pelican-core` crate
- [ ] Implement `geometry.rs`: `CircleShape`, `EllipseShape`, `TriangleShape`, `ShapeKind` enum, `PelicanModel` — all with `#[derive(Module)]`
- [ ] Implement `sdf.rs`: `sdf_circle`, `sdf_ellipse`, `sdf_triangle`, `coverage_from_sdf`
- [ ] Implement `renderer.rs`: `make_grid`, `render` (compositing loop)
- [ ] Implement `loss.rs` (spike version): `mse_loss` only (skip SSIM, edge, priors)
- [ ] Implement `optimizer.rs` (spike version): Adam optimizer loop with manual cosine LR annealing, tau annealing. No gradient clipping yet.
- [ ] Create `create_initial_pelican()` — hard-coded 9-shape geometry (port from Python)
- [ ] Write a `main.rs` test that: loads a target image, creates the pelican model, runs 100 optimization steps with `Autodiff<NdArray>`, prints loss at each step, saves the rendered image
- [ ] Verify: loss decreases, rendered image improves, gradients flow correctly
- [ ] Write basic unit tests for SDF functions (compare with known values)

**Success criteria:** `cargo run` produces a rendered pelican image that visibly improves over 100 steps. Loss monotonically decreases (mostly).

### Phase 2: Compile to Wasm and Run in Browser

Goal: Prove that Burn's `Autodiff<NdArray>` works in `wasm32-unknown-unknown`. Get the optimization loop running in a browser.

- [ ] Create `pelican-wasm` crate with `wasm-bindgen` exports
- [ ] Implement `Optimizer` struct wrapping `PelicanModel` + Adam state
- [ ] Implement `new()`: accept target image as `&[u8]`, initialize model
- [ ] Implement `step()`: one forward/backward/update cycle, return loss
- [ ] Implement `get_rendered_image()`: return RGBA pixel array
- [ ] Build with `wasm-pack build --target web`
- [ ] Verify the `.wasm` file compiles and loads in browser (check size)
- [ ] Create minimal `web/index.html` with a hard-coded target image (no upload yet)
- [ ] Wire up: load Wasm → create Optimizer → call `step()` in a loop → draw to canvas
- [ ] Verify: optimization runs in browser, canvas shows improving image, no crashes

**Success criteria:** Open `index.html` in Chrome, see the pelican optimization running live in the browser. Loss displayed and decreasing.

### Phase 3: Complete Web UI and Full Loss Function

Goal: Make it a usable demo. Add the remaining loss components and a proper UI.

- [ ] Add image upload (drag-and-drop + file picker)
- [ ] Add start/stop/reset controls
- [ ] Add loss chart (simple canvas-based or use a lightweight lib)
- [ ] Add side-by-side display: target image | current render
- [ ] Add SVG download button (`get_svg()`)
- [ ] Implement `svg.rs`: convert shape parameters to SVG string
- [ ] Add edge loss (Sobel) to `loss.rs`
- [ ] Add perimeter prior, degeneracy penalty, on-canvas penalty to `loss.rs`
- [ ] Add gradient clipping to optimizer
- [ ] Add SSIM loss (if straightforward in Burn; otherwise defer)
- [ ] Performance: measure step time in browser, optimize if needed
- [ ] Polish: loading states, error handling, responsive layout

**Success criteria:** A self-contained web page where anyone can upload an image and watch the pelican optimizer run. SVG download works. Total Wasm bundle < 3 MB.

### Phase 4: Refinements and Extensions (Future)

- [ ] Greedy refinement loop (port `greedy_refine.py`)
- [ ] WebGPU backend (`Autodiff<Wgpu>`) for GPU-accelerated optimization
- [ ] Native CLI (`pelican-cli` crate) that uses the same core library
- [ ] Higher resolutions (256x256, 512x512)
- [ ] Configurable shape palette (not just the 9-shape pelican)
- [ ] User-adjustable parameters (learning rate, tau, shape count)
- [ ] Animation: export optimization as GIF/video from browser
- [ ] Published npm package (`@differentiable-pelican/core`)

## Key Technical Risks and Mitigations

### Risk 1: Burn's Autodiff May Not Work in Wasm

**Risk:** `Autodiff<NdArray>` may fail to compile to `wasm32-unknown-unknown` or may crash at runtime due to threading, allocation, or float issues.

**Mitigation:** Phase 1 (native) is separated from Phase 2 (Wasm) precisely so we can validate the Rust logic first. If Burn's autodiff doesn't work in Wasm, we can:
- File an issue with Burn (they claim Wasm support via NdArray)
- Fall back to a custom minimal autograd (the computation graph is simple enough)
- Use Candle instead (proven Wasm + autograd, different API)

**Likelihood:** Low. Burn's NdArray backend is pure Rust with no native dependencies. The inference demo already works in Wasm. Autodiff is also pure Rust. There's no technical reason it shouldn't work.

### Risk 2: Performance May Be Too Slow in Browser

**Risk:** 500 optimization steps at 128x128 with 9 shapes may take too long in Wasm.

**Mitigation:**
- Start at 64x64 resolution for the spike (4x fewer pixels = 4x faster)
- Profile: if a single step takes <50ms, 500 steps = 25 seconds — acceptable with progress feedback
- The NdArray backend is single-threaded; if too slow, try `Autodiff<Wgpu>` (WebGPU) for GPU parallelism
- Reduce steps to 200 for the demo; the user can run more if desired

**Estimate:** PyTorch CPU does 128x128 in ~5ms/step. Rust should be comparable or faster natively. Wasm adds 5-45% overhead. So ~5-10ms/step → 500 steps in 2.5-5 seconds. This is very fast.

### Risk 3: Burn API Gaps

**Risk:** Burn may be missing specific operations needed (softplus, specific tensor manipulation patterns, conv2d with custom kernels for SSIM).

**Mitigation:**
- `softplus` can be implemented as `x.exp().log_1p()` or `torch.where(x > 20, x, log(1 + exp(x)))` using basic tensor ops
- conv2d can be avoided for Sobel by using tensor slicing (shift-and-subtract)
- SSIM can be deferred to Phase 3; MSE alone is sufficient for the spike
- Burn's API covers all standard tensor operations; the risk is mainly in edge cases

### Risk 4: wasm-bindgen Data Transfer Overhead

**Risk:** Passing image data (128x128x4 = 65KB per frame) between Wasm and JavaScript on every step may be too slow.

**Mitigation:**
- Only transfer the image every N steps (e.g., every 10 steps for display), not every step
- Transfer loss (a single f64) every step for the chart
- Use `Uint8Array` views into Wasm memory (zero-copy) where possible
- 65KB per transfer is negligible — modern browsers handle this in microseconds

## Testing Strategy

### Unit Tests (Rust)
- SDF functions: known inputs → expected distances (port Python's inline tests)
- Coverage: sigmoid at known values
- Renderer: single shape on white canvas → expected pixel values
- Loss: MSE of known images → expected value
- Geometry: sigmoid/softplus constraints produce valid ranges

### Integration Tests (Rust, native)
- Full optimization loop: 100 steps with `Autodiff<NdArray>`, verify loss decreases
- SVG export: render shapes → export SVG → verify valid SVG XML
- Snapshot test: run deterministic optimization → compare final loss to known value

### Browser Tests (manual for spike)
- Load page → Wasm initializes without errors
- Upload image → optimization starts
- Loss decreases visibly over 100 steps
- Canvas shows improving image
- SVG download produces valid file

## Rollout Plan

1. Phase 1 on feature branch — verify Rust/Burn pipeline works natively
2. Phase 2 on same branch — verify Wasm compilation and browser execution
3. Phase 3 — polish the web demo
4. Merge to main when Phase 2 is proven
5. Deploy demo to GitHub Pages (static hosting, no server needed)

## Open Questions

- Should the Rust crates live in this repository or a separate one? (Leaning: same repo, since it's the same project. Cargo workspace alongside pyproject.toml is fine.)
- Should we use `trunk` (Rust's Wasm build tool) or `wasm-pack` + manual HTML? (Leaning: `wasm-pack` for maximum control over the JS integration.)
- Is 128x128 the right default resolution for the browser demo, or should we start at 64x64? (Leaning: 64x64 for Phase 2 spike, 128x128 for Phase 3.)
- Should the web UI use a framework (React, Svelte) or plain HTML/JS? (Leaning: plain HTML/JS for the spike. Framework optional for Phase 3.)
- What's the right `requestAnimationFrame` strategy — one step per frame (60 steps/sec) or batch multiple steps per frame? (Leaning: batch 10 steps per frame, render every 10th.)

## References

- [Research Brief: Browser-Local Differentiable Rendering via Wasm](../../research/research-2026-02-15-python-wasm-feasibility.md) — detailed feasibility analysis
- [Original Spec: Differentiable Pelican](plan-2026-01-15-differentiable-pelican.md) — Python implementation spec
- [Greedy Refinement Spec](plan-2026-02-13-greedy-refinement-loop.md) — greedy shape-dropping (future Phase 4)
- [Burn Framework](https://github.com/tracel-ai/burn) — Rust deep learning framework (14,358 stars)
- [Burn MNIST Web Example](https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web) — reference for Wasm deployment patterns
- [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen) — Rust/JS FFI for Wasm
- [wasm-pack](https://rustwasm.github.io/wasm-pack/) — Build tool for Rust Wasm packages
