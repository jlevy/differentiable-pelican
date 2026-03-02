# Feature: Synchronized Python + Rust Architecture

**Date:** 2026-03-02

**Author:** Claude (with Joshua Levy)

**Status:** Draft

**Supersedes:** [plan-2026-02-16-python-rust-repo-organization.md](plan-2026-02-16-python-rust-repo-organization.md) (incorporates and extends)

**References:**
- [Rust Porting Playbook](https://github.com/jlevy/rust-porting-playbook) — full methodology
- [Burn/WASM port spec](plan-2026-02-15-burn-rust-wasm-port.md) — original Rust port plan
- [Original Python spec](plan-2026-01-15-differentiable-pelican.md)
- [Prior repo organization spec](plan-2026-02-16-python-rust-repo-organization.md)

## Overview

This spec establishes a coherent, synchronized dual-implementation architecture for
differentiable-pelican: Python (reference) + Rust (performance/WASM). It consolidates
the previous organization spec with concrete alignment to the
[rust-porting-playbook](https://github.com/jlevy/rust-porting-playbook) best practices,
and includes all the structural, testing, CI, and documentation changes needed to make
this a clean example of maintaining synchronized Python and Rust implementations in a
single repo.

This project also serves as a **case study for the playbook itself** — learnings from
this numerical/differentiable-programming port feed back into the playbook as a second
case study (complementing the flowmark CLI text-processing port).

## Goals

1. **Align Rust project setup with playbook best practices** — edition, MSRV, linting,
   release profile, Cargo.toml metadata, `deny.toml`, `rustfmt.toml`
2. **Establish shared contract** — single source of truth for initial geometry, loss
   weights, and default hyperparameters via `shared/` JSON files
3. **Enable cross-language parity testing** — golden test vectors generated from Python,
   validated in both languages
4. **Unify build/test/CI** — Makefile covers both languages; CI runs both Python and Rust
5. **Close known Rust feature gaps** — SSIM loss, loss breakdown, shape names
6. **Update README** as a dual-implementation project with version correspondence
7. **Extract learnings for the playbook** — document what's different about a
   numerical/tensor port vs a CLI text-processing port

## Non-Goals

- Porting Python-only features (LLM, greedy refine, CLI commands) to Rust
- Changing algorithmic approach in either implementation
- Moving to a monorepo tool
- Separating into multiple repositories
- WebGPU backend (future work)
- Publishing to crates.io (premature for a prototype)

## Current State Assessment

### Repository structure (post-PR #5)

```
differentiable-pelican/
├── pyproject.toml                        # Python build config (uv/hatchling)
├── Cargo.toml                            # Rust workspace (2 members)
├── Makefile                              # Python-only targets
├── src/differentiable_pelican/           # Python: ~3,250 LOC across 19 files
│   ├── sdf.py, renderer.py, loss.py      #   Core pipeline (6 modules)
│   ├── geometry.py, optimizer.py         #   ↓
│   ├── svg_export.py                     #   ↓
│   ├── cli.py, commands*.py              #   CLI layer (5 files)
│   ├── greedy_refine.py, refine.py       #   Greedy/LLM refinement
│   ├── utils.py, validator.py            #   Utilities
│   └── llm/                              #   LLM integration (4 files)
├── crates/
│   ├── pelican-core/                     # Rust core: 7 files, ~1,050 LOC
│   │   ├── src/{lib,sdf,renderer,optimizer,geometry,loss,svg_export}.rs
│   │   ├── src/bin/pipeline.rs           #   CLI binary
│   │   └── tests/integration_test.rs     #   3 integration tests
│   └── pelican-wasm/                     # WASM bindings: 1 file, ~210 LOC
├── tests/                                # Python integration tests (2 files)
├── docs/                                 # Design docs, research log, results
├── images/                               # Shared target images
└── .github/workflows/                    # Python-only CI
```

### Module mapping (current parity status)

| Python module | Rust module | Algorithmic parity | API parity | Notes |
|---|---|---|---|---|
| `sdf.py` (231 LOC) | `sdf.rs` (165 LOC) | Yes | Different arg style | Py packs `center: [2]`, Rust uses `cx, cy` separately |
| `renderer.py` (183 LOC) | `renderer.rs` (91 LOC) | Yes | Close | Rust lacks `save_render` (not needed) |
| `optimizer.py` (262 LOC) | `optimizer.rs` (156 LOC) | Yes | Partial | Rust lacks load_target, has different config API |
| `geometry.py` (375 LOC) | `geometry.rs` (150 LOC) | Yes | Partial | Rust lacks shape names |
| `loss.py` (316 LOC) | `loss.rs` (209 LOC) | Partial | Partial | Rust lacks SSIM; no loss breakdown dict |
| `svg_export.py` (241 LOC) | `svg_export.rs` (75 LOC) | Partial | Different | Rust returns String; `{:.1}` vs `{:.2f}` precision |
| `cli.py` + `commands*.py` | `bin/pipeline.rs` | N/A | Minimal | Minimal Rust CLI; not a priority |
| `greedy_refine.py` (483 LOC) | _(none)_ | — | — | Python-only, intentional |
| `refine.py` (283 LOC) | _(none)_ | — | — | Python-only, intentional |
| `llm/` (745 LOC) | _(none)_ | — | — | Python-only, intentional |
| _(none)_ | `pelican-wasm/lib.rs` (211 LOC) | — | — | Rust/WASM-only |

### Gaps vs playbook best practices

| Playbook recommendation | Current state | Gap |
|---|---|---|
| Edition 2024, `rust-version` declared | Edition 2021, no MSRV | Needs update |
| Clippy pedantic linting in Cargo.toml | No lint config | Needs `[lints]` section |
| Release profile (LTO, strip, etc.) | No release profile | Needs `[profile.release]` |
| `rustfmt.toml` | None | Needs creation |
| `deny.toml` for dependency policy | None | Needs creation |
| Cargo.toml metadata (license, desc, etc.) | Minimal | Needs workspace-level metadata |
| Tests run in CI | Python only | Rust not in CI |
| Cross-validation testing | None | No cross-language parity tests |
| Module-level port traceability comments | None | Rust files lack `//! Port of ...` |
| Shared test fixtures | None | Geometry/constants duplicated |
| `HACK:`/`FIXME:` workaround tracking | Ad hoc `// skip SSIM` | Needs systematic approach |

## Design

### Principles (from playbook, adapted for this project)

1. **Python is the reference implementation.** When they diverge, Python is authoritative.
   Rust must match Python's numerical output within tolerance.

2. **Port behavior, not implementation.** Rust uses Burn idioms (functional optimizer,
   `Module` derive, etc.) rather than mechanically translating PyTorch patterns.

3. **Tests as specification.** Parity is defined by golden test vectors generated from
   Python. A passing Rust test means the Rust output matches Python's.

4. **Numerical tolerance, not byte-for-byte.** Unlike a CLI text-processing port, we
   expect floating-point differences. Tolerances are explicit:
   - SDF values: `|python - rust| < 1e-3`
   - Pixel values (0-255): `|python - rust| <= 1`
   - Loss values: `|python - rust| < 1e-2`
   - Optimization trajectory: within 5% relative error per step

5. **Parity scope.** The Rust port covers the **core pipeline only**: sdf, renderer,
   loss, optimizer, geometry, svg_export. LLM integration, greedy refinement, CLI
   commands, and validation are Python-only and not part of the parity contract.

### Target structure

```
differentiable-pelican/
├── pyproject.toml
├── Cargo.toml                            # Workspace + workspace metadata
├── Cargo.lock                            # Committed (binary/WASM project)
├── Makefile                              # Unified Python + Rust targets
├── rustfmt.toml                          # Rust formatting config
├── deny.toml                             # Dependency policy
├── src/differentiable_pelican/           # Python source (unchanged layout)
├── crates/
│   ├── pelican-core/
│   │   ├── Cargo.toml                    # Edition 2024, MSRV, lints, release profile
│   │   ├── src/
│   │   │   ├── lib.rs                    # //! Port of Python differentiable_pelican core
│   │   │   ├── geometry.rs               # //! Port of Python geometry.py
│   │   │   ├── sdf.rs                    # //! Port of Python sdf.py
│   │   │   ├── renderer.rs               # //! Port of Python renderer.py
│   │   │   ├── loss.rs                   # //! Port of Python loss.py
│   │   │   ├── optimizer.rs              # //! Port of Python optimizer.py
│   │   │   ├── svg_export.rs             # //! Port of Python svg_export.py
│   │   │   └── bin/pipeline.rs
│   │   └── tests/
│   │       ├── integration_test.rs       # Existing integration tests
│   │       └── parity_test.rs            # Cross-language parity tests
│   └── pelican-wasm/
├── shared/                               # Cross-language contract
│   ├── README.md
│   ├── initial_pelican.json              # 9-shape geometry (single source of truth)
│   ├── constants.json                    # Loss weights, optimizer defaults
│   └── test_vectors/                     # Golden outputs from Python
│       ├── generate_test_vectors.py      # Script to regenerate
│       ├── circle_sdf_64x64.json
│       ├── ellipse_sdf_64x64.json
│       ├── triangle_sdf_64x64.json
│       ├── coverage_from_sdf.json
│       ├── render_initial_64x64.json
│       └── optimize_10steps.json
├── tests/                                # Python tests (expanded)
│   ├── test_end_to_end.py
│   ├── test_validator.py
│   └── test_parity.py                    # Python-side parity validation
├── .github/workflows/
│   ├── ci.yml                            # Python + Rust + parity
│   └── publish.yml
└── docs/
    ├── development.md                    # Updated with dual-language workflow
    └── ...
```

## Implementation Plan

### Phase 1: Rust Project Setup Alignment (playbook best practices)

Bring `Cargo.toml`, tooling, and configuration in line with the playbook.

#### Step 1.1: Update workspace `Cargo.toml`

Add resolver v3, workspace-level lints, and metadata:

```toml
[workspace]
members = ["crates/pelican-core", "crates/pelican-wasm"]
resolver = "3"

[workspace.package]
edition = "2024"
rust-version = "1.85"
license = "MIT"
repository = "https://github.com/jlevy/differentiable-pelican"

[workspace.lints.clippy]
pedantic = { level = "warn", priority = -1 }
missing_errors_doc = "allow"
missing_panics_doc = "allow"
module_name_repetitions = "allow"
must_use_candidate = "allow"

[workspace.lints.rust]
unsafe_code = "forbid"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true
panic = "abort"
```

**Files to modify:**
- `Cargo.toml` — add workspace metadata, lints, release profile

#### Step 1.2: Update crate `Cargo.toml` files

Both `pelican-core` and `pelican-wasm` inherit from workspace:

```toml
[package]
name = "pelican-core"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
description = "Differentiable SVG renderer — core pipeline (Rust/Burn port)"

[lints]
workspace = true
```

**Files to modify:**
- `crates/pelican-core/Cargo.toml` — inherit workspace settings, add serde/serde_json deps
- `crates/pelican-wasm/Cargo.toml` — inherit workspace settings

#### Step 1.3: Add `rustfmt.toml`

```toml
edition = "2024"
max_width = 100
use_small_heuristics = "Max"
```

**Files to create:**
- `rustfmt.toml`

#### Step 1.4: Add `deny.toml`

Standard dependency policy from the playbook.

**Files to create:**
- `deny.toml`

#### Step 1.5: Add port traceability comments

Every Rust source file gets a module-level comment tracing back to its Python origin:

```rust
//! Port of Python `differentiable_pelican/sdf.py`
//!
//! Signed distance field implementations for circle, ellipse, and triangle
//! primitives, plus soft coverage conversion via sigmoid.
```

**Files to modify:**
- `crates/pelican-core/src/lib.rs`
- `crates/pelican-core/src/geometry.rs`
- `crates/pelican-core/src/sdf.rs`
- `crates/pelican-core/src/renderer.rs`
- `crates/pelican-core/src/loss.rs`
- `crates/pelican-core/src/optimizer.rs`
- `crates/pelican-core/src/svg_export.rs`

**Validation:** `cargo fmt --check` and `cargo clippy -- -D warnings` both pass.

---

### Phase 2: Shared Constants

Single source of truth for values that both implementations must agree on.

#### Step 2.1: Create `shared/initial_pelican.json`

Extract the 9-shape pelican geometry from `geometry.py` into JSON. This becomes the
canonical definition.

```json
{
  "shapes": [
    {
      "name": "body",
      "type": "ellipse",
      "cx": 0.42, "cy": 0.55, "rx": 0.22, "ry": 0.28,
      "rotation": -0.3, "intensity": 0.35
    }
  ]
}
```

Both `geometry.py` and `geometry.rs` load from this file:
- Python: `json.load()` at module import
- Rust: `include_str!` at compile time + `serde_json::from_str`

**Files to create:**
- `shared/initial_pelican.json`
- `shared/constants.json` — loss weights, optimizer defaults, constraint epsilons
- `shared/README.md` — explains the shared contract

**Files to modify:**
- `src/differentiable_pelican/geometry.py` — `create_initial_pelican()` loads from JSON
- `crates/pelican-core/Cargo.toml` — add `serde`, `serde_json` deps
- `crates/pelican-core/src/geometry.rs` — load from JSON via `include_str!`
- `crates/pelican-core/src/loss.rs` — `LossWeights::default()` from JSON

**Validation:** Both implementations create 9 shapes with identical parameters.

---

### Phase 3: Cross-Language Parity Tests

#### Step 3.1: Generate golden test vectors from Python

Create a script that runs the Python pipeline with deterministic inputs and writes
structured JSON test vectors to `shared/test_vectors/`.

Test vectors to generate:
- `circle_sdf_64x64.json` — SDF at 5 sampled points for a standard circle
- `ellipse_sdf_64x64.json` — same for a rotated ellipse
- `triangle_sdf_64x64.json` — same for a triangle
- `coverage_from_sdf.json` — SDF → coverage at known tau values
- `render_initial_64x64.json` — full render of initial pelican, pixel hash + stats
- `optimize_10steps.json` — loss values for first 10 optimization steps (fixed seed)

**Files to create:**
- `shared/test_vectors/generate_test_vectors.py`
- `shared/test_vectors/*.json` — one per test case

#### Step 3.2: Python parity tests

Load test vectors and validate they match Python's live output. This confirms the
vectors are correct (since Python is the reference).

**Files to create:**
- `tests/test_parity.py`

#### Step 3.3: Rust parity tests

Load test vectors via `include_str!` and validate Rust output matches within tolerance.

**Files to create:**
- `crates/pelican-core/tests/parity_test.rs`

**Tolerances (codified in test files):**
- SDF values: `|delta| < 1e-3` (float32 arithmetic differences)
- Pixels (0-255 uint8): `|delta| <= 1` per pixel
- Loss values: `|delta| < 1e-2` (accumulated float differences)
- Optimization trajectory: loss at each step within 5% relative error

---

### Phase 4: Unified Build and CI

#### Step 4.1: Extend Makefile

```makefile
# Unified
default: install lint test

# Python
install:
	uv sync --all-extras
python-test:
	uv run pytest
python-lint:
	uv run python devtools/lint.py

# Rust
rust-build:
	cargo build
rust-test:
	cargo test --workspace
rust-lint:
	cargo clippy --workspace --all-targets -- -D warnings
rust-fmt:
	cargo fmt --all -- --check

# WASM
wasm-build:
	cd crates/pelican-wasm && wasm-pack build --target web

# Combined
lint: python-lint rust-lint rust-fmt
test: python-test rust-test
parity: test
	uv run pytest tests/test_parity.py -v
```

**Files to modify:**
- `Makefile`

#### Step 4.2: Add Rust to CI

Add a parallel `rust` job and a `parity` job:

```yaml
rust:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v6
    - uses: dtolnay/rust-toolchain@stable
      with:
        components: rustfmt, clippy
    - uses: Swatinem/rust-cache@v2
    - run: cargo fmt --all -- --check
    - run: cargo clippy --workspace --all-targets -- -D warnings
    - run: cargo test --workspace --locked

parity:
  needs: [build, rust]
  runs-on: ubuntu-latest
  steps:
    # Install both Python and Rust, run parity tests
    - run: uv run pytest tests/test_parity.py -v
    - run: cargo test --workspace parity --locked
```

**Files to modify:**
- `.github/workflows/ci.yml`

---

### Phase 5: Close Rust Feature Gaps

#### Step 5.1: Add shape names to Rust

Store names as a parallel `Vec<String>` in `PelicanModel` (avoids `Module` derive
issues with non-tensor fields). Populate from `shared/initial_pelican.json`.

**Files to modify:**
- `crates/pelican-core/src/geometry.rs` — add `names: Vec<String>` to `PelicanModel`
- `crates/pelican-core/src/svg_export.rs` — include shape name as `<!-- name -->` comment

#### Step 5.2: Add per-component loss breakdown to Rust

Change `total_loss` to return `(Tensor<B, 1>, LossBreakdown)` where:

```rust
pub struct LossBreakdown {
    pub mse: f32,
    pub edge: f32,
    pub perimeter: f32,
    pub degeneracy: f32,
    pub canvas: f32,
    pub ssim: f32,  // 0.0 until Step 5.3
}
```

**Files to modify:**
- `crates/pelican-core/src/loss.rs` — return type, `LossBreakdown` struct
- `crates/pelican-core/src/optimizer.rs` — store breakdown in `StepResult`
- `crates/pelican-wasm/src/lib.rs` — expose via `get_loss_json()`

#### Step 5.3: Add SSIM loss to Rust

Implement the Wang et al. SSIM using Gaussian-windowed statistics. For the Burn
backend without conv2d, use shifted-tensor weighted sums (same approach as existing
Sobel implementation in `loss.rs`).

**Files to modify:**
- `crates/pelican-core/src/loss.rs` — `ssim_loss()` function + include in `total_loss`

#### Step 5.4: Align SVG precision

Change Rust SVG coordinate precision from `{:.1}` to `{:.2}` to match Python's `{:.2f}`.

**Files to modify:**
- `crates/pelican-core/src/svg_export.rs` — precision format strings

---

### Phase 6: README and Documentation

#### Step 6.1: Update README as dual-implementation project

The README should reflect this is both a Python project and a Rust/WASM port. Key
additions:

```markdown
## Implementations

This repository maintains two synchronized implementations:

| | Python (reference) | Rust (performance/WASM) |
|---|---|---|
| **Location** | `src/differentiable_pelican/` | `crates/pelican-core/` |
| **Framework** | PyTorch | Burn |
| **Scope** | Full pipeline + LLM + CLI | Core pipeline + WASM |
| **Size** | ~3,250 LOC | ~1,050 LOC |
| **Use case** | CLI, experimentation | Browser demo, performance |

### Version correspondence

| Milestone | Python | Rust | Notes |
|---|---|---|---|
| Initial port | v0.1.x | 0.1.0 | Core pipeline only |
| This spec | — | 0.2.0 | Shared constants, parity tests, SSIM |

### Building

**Python:**
```bash
uv sync && pelican test-render
```

**Rust:**
```bash
cargo build --release
cargo run --bin pelican-pipeline
```

**WASM:**
```bash
cd crates/pelican-wasm && wasm-pack build --target web
```

### Testing

```bash
make test        # Both Python and Rust
make parity      # Cross-language parity validation
```

### Parity contract

The Rust implementation matches the Python core pipeline (SDF, renderer, loss,
optimizer, SVG export) within numerical tolerance. Parity is validated by golden
test vectors generated from Python and checked in CI.

Shared constants live in `shared/` — when a value changes there, both
implementations pick it up automatically.
```

**Files to modify:**
- `README.md` — add dual-implementation section, version table, build/test instructions

#### Step 6.2: Update development docs

Add a section to `docs/development.md` explaining:
- The module mapping table
- How to add a new shape type in both languages
- How to add a new loss component in both languages
- How parity tests work
- The `shared/` directory convention

**Files to modify:**
- `docs/development.md`

---

### Phase 7: Playbook Feedback

Document learnings from this numerical/tensor port that differ from the flowmark
CLI text-processing port. This feeds back into the playbook as a second case study.

#### Key differences for numerical ports

| Aspect | CLI text port (flowmark) | Numerical/tensor port (pelican) |
|---|---|---|
| **Output matching** | Byte-for-byte text diff | Numerical tolerance per element |
| **Cross-validation** | Shell script running both CLIs | Python script comparing tensor outputs |
| **Test fixtures** | Text files | JSON with sampled values + statistics |
| **Performance target** | <10MB binary, <50ms startup | <3MB WASM, <50ms/step |
| **Entry point** | CLI with clap | `#[wasm_bindgen]` API |
| **Framework mapping** | stdlib → stdlib | PyTorch → Burn (deep API differences) |
| **Gradient correctness** | N/A | Critical — autograd must produce correct grads |
| **Determinism** | Byte-for-byte | Floating-point platform variance expected |
| **Submodule structure** | Python as submodule | Same repo (both live together) |

#### Proposed additions to playbook

1. **New case study document:** `case-studies/pelican/` — numerical port example
2. **Guideline additions:** Notes on tensor library mapping (PyTorch → Burn),
   floating-point tolerance strategies, WASM-specific autograd considerations
3. **Cross-validation for numerical ports:** Script template for comparing tensor
   outputs with tolerance

**Files to create (in playbook repo, separate PR):**
- `case-studies/pelican/overview.md`
- Additions to existing guidelines as appropriate

---

## Phase Summary and Dependencies

```
Phase 1: Rust Project Setup (no deps)
  ├── 1.1: Workspace Cargo.toml
  ├── 1.2: Crate Cargo.toml files
  ├── 1.3: rustfmt.toml
  ├── 1.4: deny.toml
  └── 1.5: Port traceability comments

Phase 2: Shared Constants (no deps, can parallel with Phase 1)
  ├── 2.1: initial_pelican.json + constants.json + README
  └── 2.2: Update both implementations to load from JSON

Phase 3: Parity Tests (depends on Phase 2)
  ├── 3.1: Generate test vectors from Python
  ├── 3.2: Python parity tests
  └── 3.3: Rust parity tests

Phase 4: Unified Build & CI (depends on Phase 1, 3)
  ├── 4.1: Extend Makefile
  └── 4.2: Add Rust to CI

Phase 5: Close Rust Feature Gaps (can start after Phase 1)
  ├── 5.1: Shape names (after Phase 2)
  ├── 5.2: Loss breakdown (independent)
  ├── 5.3: SSIM loss (independent)
  └── 5.4: SVG precision (independent)

Phase 6: Documentation (after Phase 1-5)
  ├── 6.1: README update
  └── 6.2: Development docs

Phase 7: Playbook Feedback (after Phase 1-5, separate repo)
```

## Rollout Plan

1. **PR 1:** Phase 1 (Rust setup alignment) — low risk, high confidence
2. **PR 2:** Phase 2 (shared constants) — moderate, core structural change
3. **PR 3:** Phase 3 + 4 (parity tests + CI) — depends on PR 2
4. **PR 4+:** Phase 5 steps (feature gaps) — individual PRs, can interleave
5. **PR N:** Phase 6 (docs) — after everything else settles
6. **Separate repo PR:** Phase 7 (playbook feedback)

## Parity Definition (playbook principle #1)

Per the playbook's requirement for a crisp parity definition:

**Scope:** The Rust implementation is a numerical equivalent of the Python core pipeline
(sdf, renderer, loss, optimizer, geometry, svg_export). Given the same inputs (initial
geometry, target image, optimizer config), the Rust implementation must produce:

- Rendered pixel arrays within ±1 per uint8 pixel
- Loss values within 1e-2 absolute error
- SVG output with equivalent geometry (coordinate precision may differ by ≤0.01)
- Optimization trajectories that converge to similar final loss (within 5% relative)

**Tolerated variations:**
- Floating-point ordering differences (e.g., fused multiply-add availability)
- SVG formatting differences (attribute order, whitespace)
- Burn-specific API patterns that differ from PyTorch idiom
- Python-only features (LLM, greedy refine, CLI commands) — not in scope

**Not tolerated (must fail CI):**
- Algorithmic differences in SDF formulations
- Different compositing order or alpha blending
- Missing loss components
- Different constraint functions (logit, softplus)
- Different initial pelican geometry

## Open Questions

1. **Should Rust edition 2024 be used?** Burn 0.16 may not support edition 2024.
   Need to verify compatibility. Fallback: stay on 2021 with `rust-version = "1.85"`.

2. **JSON loading overhead in Python.** The `create_initial_pelican()` function is called
   frequently. Should JSON be loaded once at module level and cached? (Likely yes.)

3. **Parity of random shape generation.** `greedy_refine.py` uses random candidates.
   If we ever port greedy refine to Rust, we'd need deterministic cross-language RNG.
   Not needed now but worth noting.

4. **Burn version.** The crates currently use Burn 0.16. The playbook references 0.20.
   Should we upgrade as part of this work? (Separate concern — defer to avoid scope creep.)

5. **`pelican-cli` crate.** The WASM port spec mentioned a possible `pelican-cli` crate.
   This is out of scope for now but could be added later as a third workspace member.
