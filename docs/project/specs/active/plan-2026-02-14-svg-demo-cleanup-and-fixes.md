# Feature: SVG Demo Cleanup and Fixes

**Date:** 2026-02-14

**Author:** Claude (from senior review feedback)

**Status:** Implemented

## Overview

Address priority fixes identified in a senior-engineer design review of the differentiable
pelican demo. The goal is to make the existing demo cleaner, more correct, and more
impressive without fundamentally changing the approach. This covers code correctness bugs,
misleading comments, and the LLM judge/architect schema mismatch that causes silent failures.

## Goals

- Fix pixel grid semantics so `tau` has a consistent physical meaning
- Fix misleading comments that don't match actual behavior
- Make the LLM architect loop mechanically correct (no more silently ignored edits)
- Make the LLM judge token-efficient (don't send full loss history)
- Improve ellipse SDF accuracy for better gradient quality
- All changes should be backward-compatible with existing CLI commands

## Non-Goals

- No new rendering backends (DiffVG, Bezier splatting)
- No new loss functions (CLIP, perceptual, etc.)
- No new optimization strategies (L-BFGS, multi-resolution pyramid, shape gating)
- No strokes or new primitive types
- No fundamental architecture changes

## Background

The differentiable pelican project is a demonstration of differentiable programming: an
SVG-like scene of primitives is optimized via gradient descent to match a target image,
with an optional LLM judge/architect loop for structural refinement. A senior design
review identified several correctness and quality issues that should be fixed before
pursuing more ambitious improvements.

## Design

### Approach

Five focused fixes, each isolated to 1-2 files, with corresponding test updates:

1. **Grid pixel-center fix** — `renderer.py` `make_grid()`
2. **Comment/docstring accuracy** — `optimizer.py`
3. **Architect prompt grounding** — `llm/architect.py`
4. **Judge metrics summarization** — `llm/judge.py`
5. **Ellipse SDF improvement** — `sdf.py`

### Components

| File | Change |
|------|--------|
| `src/differentiable_pelican/renderer.py` | Fix `make_grid()` to use pixel centers |
| `src/differentiable_pelican/optimizer.py` | Fix misleading comments on lines 93 and 96 |
| `src/differentiable_pelican/llm/architect.py` | Ground prompt with actual shape state |
| `src/differentiable_pelican/llm/judge.py` | Summarize metrics instead of dumping full history |
| `src/differentiable_pelican/sdf.py` | Improve ellipse SDF approximation |
| `src/differentiable_pelican/refine.py` | Pass shape info to architect call |

### API Changes

- `architect_edits()` signature changes to accept current shapes and names
  (in addition to judge feedback)
- `judge_svg()` metrics handling changes internally (external signature unchanged)
- No CLI-level changes

## Implementation Plan

### Phase 1: Correctness fixes (no behavior change risk)

- [ ] **Fix misleading optimizer comments** (`optimizer.py:93,96`):
  Line 93 says "Create optimizer with weight decay" but no weight decay is used.
  Line 96 says "warm up then cosine decay" but there is no warmup phase.
  Fix both comments to match actual behavior.

- [ ] **Fix `make_grid()` pixel-center semantics** (`renderer.py:11-28`):
  Current code uses `torch.linspace(0, 1, H)` which places coordinates at endpoints
  (0.0 and 1.0 inclusive). For a grid representing pixel centers, coordinates should be
  `(torch.arange(N) + 0.5) / N`, giving values like `0.5/N, 1.5/N, ..., (N-0.5)/N`.
  This makes `tau ≈ 1/N` correspond to exactly one pixel width.
  Update `test_make_grid_corners` to assert pixel-center values instead of endpoint values.

- [ ] **Improve ellipse SDF approximation** (`sdf.py:24-60`):
  The current approximation `(norm(p/r) - 1) * avg_radius` distorts gradients for
  eccentric ellipses. Replace with the better-known iterative Newton's method SDF
  from the shader/SDF literature (Inigo Quilez style). This uses a few Newton iterations
  to solve the closest-point-on-ellipse problem, giving much more faithful distance
  values and gradients. Keep it to 3-4 iterations for differentiability and speed.
  Update `test_sdf_ellipse_on_boundary` tolerance from `0.05` to `0.01`.

### Phase 2: LLM loop improvements

- [ ] **Summarize metrics for judge** (`llm/judge.py`):
  When `metrics` is passed to `judge_svg()`, it currently dumps the entire dict
  including `loss_history` (one entry per optimization step — potentially 500+ entries).
  This wastes context tokens and can degrade LLM performance.
  Instead, build a summary: final loss breakdown, initial vs final total loss,
  and step count. Do not send the per-step history.

- [ ] **Ground architect prompt with actual shape state** (`llm/architect.py`, `refine.py`):
  The architect prompt currently uses a hardcoded "typical pelican structure" description
  and an example that references non-existent parameters (e.g., `"length": "+20%"` for
  a beak shape, and shape name `"beak"` when actual names are `"beak_upper"` /
  `"beak_lower"`). The LLM can easily propose edits that are silently ignored by
  `parse_edits()`.
  Fix by:
  1. Change `architect_edits()` to accept `shapes: list[Shape]` and `shape_names: list[str]`
  2. Build a "current scene state" block listing each shape's name, primitive type,
     and current parameter values (from `get_params()`)
  3. List only the actually-editable fields per primitive type
  4. Remove the misleading hardcoded example
  5. Update `refine.py` to pass shapes and names to `architect_edits()`

## Testing Strategy

- Run existing test suite (`pytest`) — all existing tests should pass (with updated
  assertions for the grid fix)
- The grid fix changes numerical values, so `test_make_grid_corners` needs updated
  expected values
- The ellipse SDF fix should make `test_sdf_ellipse_on_boundary` pass with tighter
  tolerance
- LLM-dependent tests (judge/architect) are already behind `@pytest.mark.slow` markers
  and won't run in CI by default

## Rollout Plan

Single branch, single PR. All fixes are backward-compatible at the CLI level.

## Open Questions

- Should the ellipse SDF use 3 or 4 Newton iterations? (3 is likely sufficient for
  the aspect ratios in our pelican, but 4 would handle more extreme cases)

## Future Work

The following improvements were identified in the design review but are out of scope
for this cleanup. They are documented here for prioritized follow-up:

### High Priority (next specs)

- **Multi-resolution pyramid optimization** (B1): Optimize at increasing resolutions
  (e.g., 32 → 64 → 128 → 256) with tau annealing at each level. This is the single
  biggest practical improvement for convergence reliability. Already have tau annealing;
  add resolution annealing.

- **Shape gating for continuous topology** (C3): Start with N candidate primitives
  (e.g., 200), each with a learnable opacity gate g_i ∈ [0,1]. Add L0/L1 sparsity
  penalty on gates. This replaces greedy shape addition with a fully differentiable
  topology optimization — much more compelling as a "differentiable programming" demo.

- **Two-optimizer strategy** (B2): Adam to find a good region, then L-BFGS for sharp
  convergence at the end. Standard pattern for smooth raster objectives.

### Medium Priority

- **Parameter groups with different learning rates** (B3): Group parameters by type
  (center, size, rotation, intensity) and give each group its own LR schedule.
  Centers often need higher LR early; rotation can be tricky.

- **Semantic/perceptual loss** (C1): Add CLIP embedding similarity or VGG/LPIPS
  perceptual loss as an optimization objective. This unlocks the "optimize under
  unusual criteria" thesis (e.g., "match target but look more minimalist").

- **Structured constraint losses** (C2): Symmetry penalty, limited palette via soft
  clustering, stroke consistency, minimum feature size for printability, shape count
  budget. These "design constraint" objectives are where differentiable SVG can beat
  pure generative models.

### Lower Priority / Research Direction

- **Layer ordering optimization** (C4): Assign each shape a continuous "depth" scalar,
  use soft sorting / Sinkhorn relaxation for differentiable approximate permutation.

- **Better anti-aliasing** (A2): Multi-sample AA (even 2×2 MSAA) or analytic coverage
  estimation per pixel to improve convergence at higher resolutions.

- **Strokes and paths** (A4): Add stroke rendering (variable width, round joins) to
  enable line drawings, icons, and typography with far fewer primitives.

- **Real SVG path support** (A5): Parse actual SVG path commands, allow gradients to
  update control points / stroke widths / colors, re-export. Consider DiffVG-like
  Bezier primitives.

- **TextGrad-style formalization** (D3): Formalize the judge/architect loop as a
  "textual gradient descent" system following the TextGrad (2024) abstraction.

- **Swappable objectives demo** (Demo 1 from review): Same SVG program optimized under
  different losses (pixel, CLIP, classifier, symmetry, "minimize shapes") shown as a
  comparison grid.

## References

- Senior design review (2026-02-14) — source of all items in this spec
- DiffVG: Differentiable Vector Graphics Rasterization (Li et al., 2020)
- Bézier Splatting (2025) — fast differentiable VG rasterization
- TextGrad (2024) — LLM feedback as backprop-like signal
- Inigo Quilez SDF functions — ellipse SDF reference
- Wang et al., 2004 — SSIM
