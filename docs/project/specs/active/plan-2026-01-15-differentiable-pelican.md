# Feature: Differentiable Pelican SVG Optimization

**Date:** 2026-01-15 (last updated 2026-02-13)

**Author:** Joshua Levy

**Status:** Implemented

## Overview

A hybrid system that combines gradient-based optimization with LLM-guided structural
refinement to generate SVG drawings of pelicans from a target image. Uses differentiable
rendering via soft signed distance fields (SDFs) with sigmoid-based coverage, composed
using the Porter-Duff "over" operator.

## Goals

- Optimize SVG primitive parameters (position, size, rotation, intensity) via
  backpropagation to match a target pelican image
- Use LLM (Claude) to propose discrete structural changes (add/remove/modify shapes)
  that escape local minima
- Produce clean, minimal, interpretable SVG output
- Run on CPU with optional GPU/MPS acceleration

## Non-Goals

- Photo-realistic rendering (grayscale silhouette only)
- Complex SVG paths or Bezier curves (simple primitives only in v1)
- Multi-provider LLM support (Anthropic only)
- Web UI (CLI only)

## Background

SVG primitives are discrete and symbolic, but optimization requires continuous gradients.
We solve this by parameterizing shapes with continuous variables, rendering via
differentiable soft SDFs, and backpropagating through the rasterized image. An LLM
handles the discrete structural decisions that gradient descent cannot discover.

See [pelican-plan.md](../../../design/pelican-plan.md) for the full design rationale
and detailed phase descriptions.

## Design

### Approach

Two-phase architecture:

1. **Phase 1 (Gradient)**: Differentiable SDF rendering → multi-component loss
   (MSE + SSIM + edge + priors) → Adam optimizer with tau annealing
2. **Phase 2 (LLM)**: Judge evaluates → Architect proposes edits → Edit parser applies
   → Re-optimize → Rollback on degradation

### Components

| Module | Purpose |
|--------|---------|
| `geometry.py` | Shape primitives (Circle, Ellipse, Triangle) with per-shape intensity |
| `sdf.py` | Signed distance fields (Quilez formulations) |
| `renderer.py` | Soft rasterization with Porter-Duff compositing |
| `loss.py` | MSE + SSIM (Wang et al.) + Sobel edge + geometric priors |
| `optimizer.py` | Adam with cosine LR, tau annealing, gradient clipping |
| `refine.py` | Multi-round refinement loop with automatic rollback |
| `llm/client.py` | Shared Anthropic API client with retry logic |
| `llm/judge.py` | Multimodal SVG evaluation |
| `llm/architect.py` | Structural edit generation |
| `llm/edit_parser.py` | Edit application (modify/add/remove with intensity) |

### API Changes

CLI entry point: `pelican` with five commands:
- `test-render` - Render initial geometry
- `optimize` - Gradient-based optimization
- `judge` - LLM evaluation
- `refine` - Full refinement loop
- `validate-image` - Image validation

## Implementation Plan

### Phase 0: Image Validation [COMPLETE]

- [x] Multimodal LLM image validation
- [x] Pydantic schema for structured output
- [x] CLI: `pelican validate-image`

### Phase 1A: Foundation [COMPLETE]

- [x] SDF functions for circle, ellipse, triangle
- [x] Soft rasterizer with sigmoid coverage
- [x] 9-shape initial pelican geometry with named roles
- [x] Per-shape grayscale intensity
- [x] SVG export with grayscale fills

### Phase 1B: Optimization [COMPLETE]

- [x] Multi-component loss (MSE + SSIM + edge + priors)
- [x] Adam optimizer with cosine LR annealing
- [x] Tau annealing (coarse → fine)
- [x] Gradient clipping, NaN guards
- [x] Progress callbacks

### Phase 1C: Full Features [COMPLETE]

- [x] Device auto-detection (CPU/CUDA/MPS)
- [x] 128x128 resolution
- [x] GIF generation from saved frames
- [x] Metrics export (JSON)
- [x] Rich progress display

### Phase 2A: Judge [COMPLETE]

- [x] Claude multimodal evaluation
- [x] JudgeFeedback Pydantic schema
- [x] CLI: `pelican judge`

### Phase 2B: Architect & Edit Parser [COMPLETE]

- [x] Structural edit generation from judge feedback
- [x] Edit types: modify, add, remove (with intensity)
- [x] Bounds validation

### Phase 2C: Refinement Loop [COMPLETE]

- [x] Multi-round judge → architect → edit → optimize
- [x] Convergence detection
- [x] Per-round output directories

### Phase 2D: Robustness [COMPLETE]

- [x] Automatic rollback on quality degradation
- [x] Consecutive failure limit (default: 2)
- [x] Best-state tracking and restoration
- [x] Shared LLM client with retry logic

## Testing Strategy

- **37 inline unit tests**: SDF gradients, loss functions, geometry constraints,
  renderer output, SVG validity (run in CI)
- **6 integration/e2e tests**: Full pipeline with API calls (marked `slow`, skip in CI)
- **Linting**: ruff + basedpyright + codespell (0 warnings, 0 errors)
- **CI**: GitHub Actions on Python 3.11-3.14, ubuntu-latest

## Rollout Plan

1. Development on feature branch with PR review
2. CI validation (lint + unit tests)
3. Manual e2e testing with target image
4. Merge to main, publish to PyPI via GitHub release

## Open Questions

- Bezier curve support for smoother contours?
- Full RGB color rendering?
- Multi-resolution optimization strategy?

## References

- DiffVG: Li et al. (2020), ACM Trans. Graphics
- SSIM: Wang et al. (2004), IEEE Trans. Image Processing
- SDF formulas: Quilez, iquilezles.org
- Porter-Duff compositing: Porter & Duff (1984), SIGGRAPH

See [pelican-plan.md](../../../design/pelican-plan.md) for full citation details.
