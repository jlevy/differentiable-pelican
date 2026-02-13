# Feature: Greedy Shape-Dropping Refinement Loop

**Date:** 2026-02-13

**Author:** Joshua Levy

**Status:** Draft

## Overview

Replace the current LLM-guided batch-edit refinement loop with a greedy
shape-dropping approach. Instead of asking an LLM architect to propose 8-11
edits per round, we add one shape at a time, let gradient descent find its
optimal placement, and keep it only if it improves the loss. This builds up
complexity incrementally and rejects shapes that don't help.

## Goals

- Incrementally add shapes one at a time, testing each before committing
- Let gradient descent handle placement (no LLM needed for positioning)
- Two-phase per candidate: settle new shape (frozen existing), then re-optimize all
- Reject candidates that don't improve overall loss
- Configurable shape budget, scale, freeze behavior
- Produce cleaner results than the current batch-edit approach

## Non-Goals

- Replacing the LLM judge/architect entirely (they may be useful for suggesting
  WHAT shapes to try, even if not WHERE to place them)
- Full evolutionary/population-based search
- Multi-resolution optimization (separate improvement)
- RGB color support

## Background

The current refinement loop (`refine.py`) asks an LLM architect to propose
structural edits after each optimization round. In practice this degrades
quality: the architect adds 8-11 shapes per round, many overlapping or
redundant, and the loss metric rewards more shapes (more degrees of freedom)
even when visual quality doesn't improve. The final result with 19 shapes
(loss 0.032) looks visually worse than the clean 9-shape optimization
(loss 0.035) despite the lower loss.

The greedy approach treats shape addition as a search problem: try one shape,
optimize, keep if better, discard if not. This is analogous to greedy forward
selection in feature selection or boosting in ML.

## Design

### Approach

**Two-phase trial per candidate shape:**

1. **Phase A (Settle):** Drop a random shape into the scene. Freeze all
   existing shapes. Optimize only the new shape for N steps. Gradient descent
   pulls it to where it reduces loss most.

2. **Phase B (Re-optimize):** Unfreeze all shapes. Optimize everything together
   for M steps. This lets existing shapes adjust to accommodate the newcomer.

3. **Decision:** Compare loss after both phases to loss before the shape was
   added. If improved: keep shape, save state. If not: discard shape, restore
   previous state, try a different shape type.

**Shape cycling:** Cycle through shape types (circle, ellipse, triangle) to
ensure variety. Each candidate gets random initial position, size, and intensity.

**Termination:** Stop when shape budget is reached or after K consecutive
rejections (no shape type helps anymore).

### Components

| Module | Purpose |
|--------|---------|
| `greedy_refine.py` | Core greedy refinement loop with two-phase trial |
| `commands_greedy_refine.py` | CLI command: `pelican greedy-refine` |
| `geometry.py` | Existing shape creation (Circle, Ellipse, Triangle) |
| `optimizer.py` | Existing gradient descent (reused per phase) |

### API Changes

New CLI command:

```bash
pelican greedy-refine \
  --target images/pelican-drawing-1.jpg \
  --resolution 128 \
  --max-shapes 20 \
  --initial-steps 500 \
  --settle-steps 100 \
  --reoptimize-steps 200 \
  --freeze-existing \
  --scale 1.0 \
  --max-failures 5 \
  --output-dir out/greedy
```

Key parameters:
- `--max-shapes`: Shape budget (default 20)
- `--settle-steps`: Phase A optimization steps (default 100)
- `--reoptimize-steps`: Phase B optimization steps (default 200)
- `--freeze-existing`: Only optimize new shape during Phase A (default on)
- `--scale`: Size multiplier for new shapes (default 1.0)
- `--max-failures`: Consecutive rejections before stopping (default 5)

## Implementation Plan

### Phase 1: Core Greedy Loop

- [x] `greedy_refine.py`: Core module with `greedy_refinement_loop()`
- [x] `create_random_shape()`: Random shape factory with configurable scale
- [x] `_freeze_shapes()` / `_unfreeze_shapes()`: Parameter freezing helpers
- [x] Two-phase trial logic (settle + re-optimize + keep/discard)
- [x] Deep copy for rollback on rejection
- [x] Per-round output saving (PNG/SVG for accepted rounds)
- [ ] `commands_greedy_refine.py`: CLI command with all parameters
- [ ] Register in `cli.py` dispatcher
- [ ] End-to-end test run and comparison with current refine

### Phase 2: Shape Replacement (Future)

- [ ] Identify least-helpful shape (remove one, measure loss change)
- [ ] Replace worst shape with a different type/size
- [ ] Split: replace one shape with two smaller shapes
- [ ] Shape contribution scoring for diagnostics

## Testing Strategy

- **Unit tests:** Shape creation, freeze/unfreeze, loss evaluation
- **Integration:** Full greedy loop with 3-5 rounds at low resolution (64x64)
- **Comparison:** Run greedy vs current refine on same target, compare loss
  and visual quality at equivalent shape counts
- **Regression:** Ensure existing optimize and refine commands still work

## Rollout Plan

1. Implement Phase 1 on feature branch
2. Run side-by-side comparison (greedy vs LLM refine)
3. Document results in docs/results/
4. Merge; keep both commands available (`refine` for LLM, `greedy-refine` for greedy)

## Open Questions

- Should we try multiple random placements per shape type and pick the best?
  (Currently: one random trial per type, cycling through types.)
- Should the LLM suggest what shape to try next instead of random cycling?
  (Hybrid: LLM for "what", gradient descent for "where".)
- Optimal settle vs re-optimize step ratio? (Currently 100:200.)
- Should we add a "warm-up" where the new shape starts near the highest-error
  region instead of random placement?

## References

- [Pelican Plan](../../../design/pelican-plan.md) - Full design document
- [Original spec](plan-2026-01-15-differentiable-pelican.md) - Differentiable pelican spec
- [Pipeline results](../../../results/README.md) - Current E2E results
- Greedy forward selection: analogous to stepwise regression / boosting
