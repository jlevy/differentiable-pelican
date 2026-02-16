# Feature: Python + Rust Dual-Implementation Repo Organization

**Date:** 2026-02-16

**Author:** Claude (with Joshua Levy)

**Status:** Draft

**Depends on:** Merge of PR #3 (greedy-refinement / SVG cleanup) first.

## Overview

Reorganize the differentiable-pelican repository so that the Python and Rust implementations can be maintained side-by-side with confidence. The Rust port mirrors the Python core pipeline, but today there is no shared contract, no cross-language parity testing, and no unified build/test infrastructure. This spec defines incremental steps to fix that — from highest-value/lowest-risk changes to the largest structural moves.

## Goals

- Establish a single source of truth for shared constants (initial pelican geometry, loss weights, default hyperparameters)
- Enable automated cross-language parity testing: same inputs produce same outputs (within tolerance)
- Unify the Makefile so `make test` covers both Python and Rust
- Make the module mapping between Python and Rust explicit and documented
- Close known feature gaps in the Rust port (SSIM loss, per-component loss breakdown, shape names)
- Move inline Python tests to `tests/` for clean separation

## Non-Goals

- Porting Python-only features (LLM integration, greedy refine) to Rust
- Changing the Rust WASM API or web frontend
- Changing the algorithmic approach in either implementation
- Moving to a monorepo tool (Nx, Turborepo, etc.) — the repo is small enough for a Makefile
- Separating into multiple repositories

## Background

### Current state (post-PR #3)

```
differentiable-pelican/
├── pyproject.toml                        # Python build config
├── Cargo.toml                            # Rust workspace (2 members)
├── Makefile                              # Python-only targets
├── src/differentiable_pelican/           # Python: 22 files, ~4,000 LOC
│   ├── sdf.py                            #   Core pipeline (6 modules)
│   ├── renderer.py                       #   ↓
│   ├── optimizer.py                      #   ↓
│   ├── geometry.py                       #   ↓
│   ├── loss.py                           #   ↓
│   ├── svg_export.py                     #   ↓
│   ├── cli.py, commands*.py              #   CLI layer
│   ├── greedy_refine.py, refine.py       #   Greedy refinement
│   ├── utils.py, validator.py            #   Utilities
│   └── llm/                              #   LLM integration (4 files)
├── crates/
│   ├── pelican-core/                     # Rust core: 8 files, ~850 LOC
│   │   └── src/{sdf,renderer,optimizer,geometry,loss,svg_export}.rs
│   └── pelican-wasm/                     # WASM bindings: 1 file, ~210 LOC
├── web/                                  # Browser demo (HTML + JS + compiled WASM)
├── tests/                                # Python integration tests (2 files)
├── docs/results/                         # Python output images
├── docs/results/rust/                    # Rust output images
└── images/                               # Shared target images
```

### Problems

1. **No shared contract.** The 9-shape initial pelican geometry is hardcoded independently in both `geometry.py` (lines 197-340) and `geometry.rs` (lines 95-149). Same magic numbers, two copies. When one changes, the other must be manually updated.

2. **API signature divergence.** Python packs coordinates (`center: [2]`, `vertices: [3, 2]`); Rust decomposes them (`cx, cy` as separate tensors). This isn't wrong — it's idiomatic for each language — but it's undocumented.

3. **Feature gaps in Rust.**
   - SSIM loss: Python has it (weight 0.05), Rust skips it (`// skip SSIM for now`)
   - Per-component loss breakdown: Python returns `(Tensor, dict[str, float])`; Rust returns bare `Tensor<B, 1>`
   - Shape names: Python returns `(list[Shape], list[str])` with human-readable names ("body", "neck", "head"); Rust returns only `PelicanModel<B>`
   - `load_target_image`: Python-only
   - `save_render`: Python-only
   - `composite_stages_svg`: Python-only (used by greedy refine)

4. **No cross-language parity tests.** Python has ~40 tests (32 inline in source files + 8 in `tests/`). Rust has 3 integration tests in `crates/pelican-core/tests/integration_test.rs`. Nothing checks that they produce the same output for the same input.

5. **Inline tests pollute Python source.** `sdf.py`, `renderer.py`, `loss.py`, etc. each have `test_*` functions at the bottom of the module. These are discoverable by pytest but mixed in with production code.

6. **Makefile is Python-only.** `make test` runs `uv run pytest`. `make lint` runs Python linting. Rust is built/tested separately via `cargo test`, `cargo build`, `wasm-pack build`.

7. **Results directory is ad hoc.** Python outputs go to `docs/results/`. Rust outputs go to `docs/results/rust/`. No shared comparison or validation.

8. **CI runs Python only.** `.github/workflows/ci.yml` runs pytest and linting. Rust is not tested in CI.

### Module mapping (current)

| Python module | Rust module | Parity status |
|---|---|---|
| `sdf.py` (231 LOC) | `sdf.rs` (165 LOC) | Algorithmic match, different arg conventions |
| `renderer.py` (183 LOC) | `renderer.rs` (91 LOC) | Match; Rust lacks `save_render` |
| `optimizer.py` (262 LOC) | `optimizer.rs` (156 LOC) | Match; Rust lacks load_target, loss breakdown |
| `geometry.py` (375 LOC) | `geometry.rs` (150 LOC) | Match; Rust lacks shape names |
| `loss.py` (316 LOC) | `loss.rs` (209 LOC) | Rust lacks SSIM; no loss breakdown dict |
| `svg_export.py` (241 LOC) | `svg_export.rs` (75 LOC) | Rust returns String vs file write; no composite_stages_svg |
| `cli.py` + `commands*.py` | `bin/pipeline.rs` | Minimal Rust CLI; not a priority |
| `greedy_refine.py` (483 LOC) | _(none)_ | Python-only, intentional |
| `refine.py` (283 LOC) | _(none)_ | Python-only, intentional |
| `llm/` (745 LOC) | _(none)_ | Python-only, intentional |
| `utils.py` (37 LOC) | _(none)_ | Python-only |
| `validator.py` (179 LOC) | _(none)_ | Python-only |
| _(none)_ | `pelican-wasm/lib.rs` (211 LOC) | Rust/WASM-only |

## Design

### Principle: Python is the reference, Rust mirrors the core

The Python implementation is the full-featured reference. The Rust implementation mirrors only the **core pipeline** (sdf, renderer, optimizer, geometry, loss, svg_export) for performance and WASM deployment. Python-only modules (LLM, greedy refine, CLI, validator) are not ported.

When the two diverge, Python is authoritative. Rust must match Python's numerical output within tolerance.

### What changes and what doesn't

**Changes:**
- Add `shared/` directory for cross-language constants and test vectors
- Move inline Python tests to `tests/`
- Extend Makefile with Rust targets
- Add Rust to CI
- Close Rust feature gaps (SSIM, loss breakdown, shape names)
- Add parity tests

**Doesn't change:**
- Python source stays at `src/differentiable_pelican/` (pyproject.toml expects this)
- Rust source stays at `crates/` (Cargo workspace expects this)
- Web demo stays at `web/`
- No directory restructuring of `python/` vs `rust/` top-level split (too disruptive for a small project)

## Implementation Plan

### Phase 1: Shared Constants (highest value, lowest risk)

#### Step 1.1: Create `shared/initial_pelican.json`

Extract the 9-shape pelican geometry from `geometry.py` lines 197-340 into a JSON file. This becomes the single source of truth.

```json
{
  "shapes": [
    {
      "name": "body",
      "type": "ellipse",
      "cx": 0.42, "cy": 0.55,
      "rx": 0.22, "ry": 0.28,
      "rotation": -0.3,
      "intensity": 0.35
    },
    {
      "name": "neck",
      "type": "ellipse",
      "cx": 0.52, "cy": 0.35,
      "rx": 0.06, "ry": 0.16,
      "rotation": 0.2,
      "intensity": 0.3
    }
  ]
}
```

**Files to create:**
- `shared/initial_pelican.json` — full 9-shape geometry with names, types, and all parameters

**Files to modify:**
- `src/differentiable_pelican/geometry.py` — `create_initial_pelican()` loads from JSON instead of hardcoding. Keep the hardcoded values as fallback/documentation in comments.
- `crates/pelican-core/src/geometry.rs` — `create_initial_pelican()` loads from JSON at compile time via `include_str!` + `serde_json::from_str`. This embeds the JSON in the binary with zero runtime file I/O.
- `crates/pelican-core/Cargo.toml` — add `serde` and `serde_json` dependencies

**Validation:** Both `python -c "from differentiable_pelican.geometry import create_initial_pelican; print(len(create_initial_pelican()[0]))"` and `cargo test` produce 9 shapes with identical parameters.

#### Step 1.2: Create `shared/constants.json`

Extract shared numerical constants into a single file:

```json
{
  "loss_weights": {
    "perimeter": 0.001,
    "degeneracy": 0.1,
    "canvas": 1.0,
    "edge": 0.1,
    "ssim": 0.05
  },
  "optimizer_defaults": {
    "lr": 0.02,
    "lr_end": 0.002,
    "tau_start": 0.05,
    "tau_end": 0.005,
    "grad_clip_norm": 1.0,
    "steps": 500,
    "resolution": 128
  },
  "constraints": {
    "sigmoid_eps": 0.01,
    "softplus_eps": 0.01,
    "coverage_clamp": 10.0
  }
}
```

**Files to create:**
- `shared/constants.json`

**Files to modify:**
- `src/differentiable_pelican/loss.py` — `total_loss()` default weights loaded from JSON
- `src/differentiable_pelican/optimizer.py` — default optimizer params loaded from JSON
- `crates/pelican-core/src/loss.rs` — `LossWeights::default()` loaded from JSON via `include_str!`
- `crates/pelican-core/src/optimizer.rs` — `OptimConfig::new()` loaded from JSON via `include_str!`

#### Step 1.3: Add `shared/README.md`

Brief explanation of what the shared directory is for and how both languages consume it. Include the invariant: "When you change a value here, both implementations pick it up automatically."

---

### Phase 2: Cross-Language Parity Tests

#### Step 2.1: Generate golden test vectors from Python

Create a Python script that generates deterministic test vectors and writes them to `shared/test_vectors/`. These become the ground truth that Rust must match.

**Files to create:**
- `shared/generate_test_vectors.py` — script that produces the files below
- `shared/test_vectors/circle_sdf_64x64.json` — for a circle at (0.5, 0.5) radius 0.2, the SDF values at 5 sampled grid positions
- `shared/test_vectors/ellipse_sdf_64x64.json` — same for an ellipse
- `shared/test_vectors/triangle_sdf_64x64.json` — same for a triangle
- `shared/test_vectors/coverage_from_sdf.json` — SDF → coverage for known tau values
- `shared/test_vectors/render_initial_64x64.json` — initial pelican render: hash of pixel array + loss against known target
- `shared/test_vectors/optimize_10steps.json` — loss values for first 10 optimization steps (fixed seed/init)

**Test vector format (example):**
```json
{
  "description": "SDF of circle at (0.5, 0.5) radius 0.2 on 64x64 grid",
  "inputs": {
    "cx": 0.5, "cy": 0.5, "radius": 0.2,
    "grid_size": 64
  },
  "expected": {
    "sdf_at_center": -0.2,
    "sdf_at_0_0": 0.507,
    "sdf_at_0.5_0.7": 0.0,
    "sdf_min": -0.2,
    "sdf_max": 0.507
  },
  "tolerance": 1e-3
}
```

#### Step 2.2: Python parity tests

**Files to create:**
- `tests/test_parity.py` — loads each test vector from `shared/test_vectors/`, runs the Python function, asserts output matches within tolerance. This validates that the test vectors themselves are correct (since Python is the reference).

#### Step 2.3: Rust parity tests

**Files to create:**
- `crates/pelican-core/tests/parity_test.rs` — loads each test vector from `shared/test_vectors/` (via `include_str!`), runs the Rust function, asserts output matches within tolerance.

**Key tolerances:**
- SDF values: `|python - rust| < 1e-3` (float32 precision)
- Rendered pixel values (0-255 uint8): `|python - rust| <= 1` per pixel
- Loss values: `|python - rust| < 1e-2` (accumulated floating point differences)
- Optimization trajectory (10 steps): loss at each step within 5% relative error

---

### Phase 3: Move Inline Python Tests to `tests/`

#### Step 3.1: Extract inline tests from source files

Currently these Python source files have `test_*` functions at the bottom:
- `sdf.py` — `test_sdf_circle`, `test_sdf_ellipse`, `test_sdf_triangle`, `test_coverage`
- `renderer.py` — `test_make_grid`, `test_render_single_circle`, `test_render_compositing`
- `loss.py` — `test_mse_loss`, `test_edge_loss`, `test_ssim_loss`, `test_total_loss`, `test_perimeter_prior`
- `geometry.py` — `test_circle_constraints`, `test_ellipse_constraints`, `test_create_initial_pelican`
- `svg_export.py` — `test_shapes_to_svg`
- `optimizer.py` — `test_anneal_tau`, `test_cosine_lr`
- `greedy_refine.py` — `test_greedy_refine_*` (multiple)
- `refine.py` — various test functions

**Files to create:**
- `tests/test_sdf.py` — move SDF tests here
- `tests/test_renderer.py` — move renderer tests here
- `tests/test_loss.py` — move loss tests here
- `tests/test_geometry.py` — move geometry tests here
- `tests/test_svg_export.py` — move SVG export tests here
- `tests/test_optimizer.py` — move optimizer tests here (anneal_tau, cosine_lr)
- `tests/test_greedy_refine.py` — move greedy refine tests here

**Files to modify:**
- `src/differentiable_pelican/sdf.py` — remove inline test functions
- `src/differentiable_pelican/renderer.py` — remove inline test functions
- `src/differentiable_pelican/loss.py` — remove inline test functions
- `src/differentiable_pelican/geometry.py` — remove inline test functions
- `src/differentiable_pelican/svg_export.py` — remove inline test functions
- `src/differentiable_pelican/optimizer.py` — remove inline test functions
- `src/differentiable_pelican/greedy_refine.py` — remove inline test functions
- `src/differentiable_pelican/refine.py` — remove inline test functions

**Validation:** `uv run pytest tests/` passes with the same number of tests as before.

---

### Phase 4: Unified Build and CI

#### Step 4.1: Extend the Makefile

**Files to modify:**
- `Makefile` — add these targets:

```makefile
# Rust targets
rust-build:
	cargo build --manifest-path Cargo.toml

rust-test:
	cargo test --manifest-path Cargo.toml

rust-clippy:
	cargo clippy --manifest-path Cargo.toml -- -D warnings

wasm-build:
	cd crates/pelican-wasm && wasm-pack build --target web

# Unified targets
test: python-test rust-test       # replaces current 'test' target
lint: python-lint rust-clippy     # replaces current 'lint' target

# Parity check
parity: python-test rust-test
	uv run pytest tests/test_parity.py -v

# Rename existing targets
python-test:
	uv run pytest

python-lint:
	uv run python devtools/lint.py

# Results for both
rust-results:
	cargo run --manifest-path Cargo.toml --bin pelican-pipeline

results: python-results rust-results
```

#### Step 4.2: Add Rust to CI

**Files to modify:**
- `.github/workflows/ci.yml` — add a `rust` job:

```yaml
rust:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo test
    - run: cargo clippy -- -D warnings
```

Also add a `parity` job that depends on both `python` and `rust` jobs, running `tests/test_parity.py` to validate cross-language consistency.

---

### Phase 5: Close Rust Feature Gaps

These are the known divergences in the core pipeline that should be closed for full parity.

#### Step 5.1: Add shape names to Rust

**Files to modify:**
- `crates/pelican-core/src/geometry.rs`:
  - Add a `name: String` field to `CircleShape`, `EllipseShape`, `TriangleShape` (or add it to `ShapeKind` as a wrapper)
  - Alternative: store names in `PelicanModel` as a parallel `Vec<String>` (simpler, avoids `Module` derive issues with `String`)
  - `create_initial_pelican()` populates names from `shared/initial_pelican.json`
- `crates/pelican-core/src/svg_export.rs` — include shape name as XML comment in SVG output

#### Step 5.2: Add per-component loss breakdown to Rust

**Files to modify:**
- `crates/pelican-core/src/loss.rs`:
  - Change `total_loss` return type from `Tensor<B, 1>` to `(Tensor<B, 1>, LossBreakdown)`
  - Add `pub struct LossBreakdown { pub mse: f32, pub edge: f32, pub perimeter: f32, pub degeneracy: f32, pub canvas: f32 }`
- `crates/pelican-core/src/optimizer.rs` — store `LossBreakdown` in `StepResult` and `loss_history`
- `crates/pelican-wasm/src/lib.rs` — expose loss breakdown via `get_loss_json()` (already stubbed in the original spec)

#### Step 5.3: Add SSIM loss to Rust

**Files to modify:**
- `crates/pelican-core/src/loss.rs`:
  - Implement `ssim_loss()` — Gaussian-window SSIM using the Wang et al. formulation
  - The main challenge is the Gaussian window convolution. Options:
    a. Manual implementation via shifted-tensor weighted sums (like the existing Sobel)
    b. Use Burn's `conv2d` if available in 0.16
    c. Approximate with a box filter (simpler but less accurate)
  - Add `ssim: f32` field to `LossWeights` (default 0.05)
  - Include in `total_loss`

#### Step 5.4: Align SVG precision and comments

**Files to modify:**
- `crates/pelican-core/src/svg_export.rs`:
  - Change coordinate precision from `{:.1}` to `{:.2}` (match Python's `{:.2f}`)
  - Add `<!-- Shape N: name -->` comments (once shape names are available from Step 5.1)

---

### Phase 6: Documentation (optional, do last)

#### Step 6.1: Add cross-language architecture note to `docs/development.md`

**Files to modify:**
- `docs/development.md` — add a section explaining:
  - The module mapping table (from this spec)
  - How to add a new shape type in both languages
  - How to add a new loss component in both languages
  - How parity tests work
  - The `shared/` directory convention

#### Step 6.2: Update `docs/design/implementation-progress.md`

**Files to modify:**
- `docs/design/implementation-progress.md` — update Rust parity status with a checklist of what's done vs remaining

---

## Phase Summary and Dependencies

```
Phase 1: Shared Constants
  ├── 1.1: shared/initial_pelican.json     (no dependencies)
  ├── 1.2: shared/constants.json           (no dependencies)
  └── 1.3: shared/README.md               (after 1.1 + 1.2)

Phase 2: Parity Tests                      (depends on Phase 1)
  ├── 2.1: Generate test vectors           (after 1.1)
  ├── 2.2: Python parity tests             (after 2.1)
  └── 2.3: Rust parity tests               (after 2.1)

Phase 3: Move Inline Tests                 (independent of Phase 1-2)
  └── 3.1: Extract to tests/

Phase 4: Unified Build & CI                (depends on Phase 2-3)
  ├── 4.1: Extend Makefile                 (after 2.3 + 3.1)
  └── 4.2: Add Rust to CI                  (after 4.1)

Phase 5: Close Rust Feature Gaps           (independent, can start anytime)
  ├── 5.1: Shape names                     (after 1.1)
  ├── 5.2: Loss breakdown                  (independent)
  ├── 5.3: SSIM loss                       (independent)
  └── 5.4: SVG precision + comments        (after 5.1)

Phase 6: Documentation                    (after everything else)
  ├── 6.1: Development docs
  └── 6.2: Progress tracking
```

Phases 1-3 can be parallelized. Phase 5 can start at any time. Phase 4 should come after 2-3 so CI has something to run. Phase 6 is last.

## Testing Strategy

- **Phase 1:** `cargo test` and `uv run pytest` both pass; both load from `shared/` JSON files
- **Phase 2:** `uv run pytest tests/test_parity.py` passes; `cargo test parity` passes; both validate against same golden vectors
- **Phase 3:** `uv run pytest` runs same number of tests as before, all passing
- **Phase 4:** `make test` runs both Python and Rust; CI checks both; `make parity` validates cross-language consistency
- **Phase 5:** Parity tests updated to include SSIM, loss breakdown, shape names; all pass

## Rollout Plan

1. Implement Phase 1-3 in a single PR (shared constants + parity tests + test reorganization)
2. Phase 4 (Makefile + CI) as a follow-up PR
3. Phase 5 steps as individual PRs (each is self-contained)
4. Phase 6 as a final cleanup PR

## Open Questions

- **JSON loading in Rust at compile time vs runtime?** `include_str!` embeds at compile time (zero runtime cost, but requires rebuild on JSON change). `std::fs::read` loads at runtime (flexible, but needs file path). Recommendation: `include_str!` for constants and initial geometry (they change rarely); runtime loading only if we add user-configurable shape files later.
- **Should parity tolerance be configurable?** For now, hardcode tolerances in test files. If we find they need tuning across platforms (e.g., ARM vs x86 float differences), extract to `shared/test_tolerances.json`.
- **Should `shared/` live at repo root or inside `docs/`?** Recommendation: repo root — it's consumed by source code, not documentation.
- **What about the `PelicanModel.shapes` ordering?** Both implementations iterate shapes front-to-back for compositing. The JSON file defines the canonical ordering. Both must load in the same order.

## References

- [Existing Rust/WASM port spec](plan-2026-02-15-burn-rust-wasm-port.md) — original porting plan
- [Original Python spec](plan-2026-01-15-differentiable-pelican.md)
- [Greedy refinement spec](plan-2026-02-13-greedy-refinement-loop.md)
- [Research brief: Python WASM feasibility](../../research/research-2026-02-15-python-wasm-feasibility.md)
