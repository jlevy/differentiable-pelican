# Research Experiment Log

Chronological log of experiments, algorithm improvements, and their results
for the differentiable pelican SVG optimization project.

All experiments run at 128x128 resolution on CPU unless noted otherwise.

---

## Experiment 1: Baseline Gradient Optimization

**Date:** 2026-02-13
**Commit:** `4df9354`
**Command:** `pelican optimize --steps 500 --resolution 128`

**Setup:** 9 hard-coded shapes (body, neck, head, beak_upper, beak_lower,
wing, tail, eye, feet) with per-shape grayscale intensity. Adam optimizer
with cosine LR annealing. Multi-component loss: MSE + SSIM + Sobel edge +
geometric priors (perimeter, degeneracy, on-canvas).

**Results:**

| Metric | Value |
|--------|-------|
| Final loss | 0.0351 |
| Shapes | 9 |
| Steps | 500 |
| LR | 0.02 |

**Qualitative:** Produces a recognizable pelican silhouette. Body, beak,
and head are clearly visible. Feet and fine details are approximate. The
9-shape constraint limits expressiveness but keeps the output clean.

**Key insight:** Gradient descent is very effective at placing and sizing
shapes. The main limitation is the fixed topology -- we can't add or
remove shapes.

---

## Experiment 2: LLM-Guided Refinement (Batch Edits)

**Date:** 2026-02-13
**Commit:** `e92e9df`
**Command:** `pelican refine --rounds 5 --steps-per-round 200`

**Setup:** Multi-round loop: optimize all shapes, then LLM judge evaluates
the result, LLM architect proposes structural edits (add/remove/modify),
edits are applied, re-optimize. Rollback on degradation.

**Algorithm:**
```
for round in 1..N:
  optimize(all_shapes, 200 steps)
  judge(result) → feedback
  architect(feedback) → 8-11 edits
  apply_all_edits(shapes)
  if loss > previous * 1.1: rollback
```

**Results:**

| Metric | Value |
|--------|-------|
| Final loss | 0.0320 |
| Shapes | 19 (started at 9) |
| Rounds | 5 |
| Improvement over baseline | 9% |

**Per-round breakdown:**

| Round | Loss | Shapes | Notes |
|-------|------|--------|-------|
| 1 | 0.0382 | 9 | Initial optimization |
| 2 | 0.0342 | 14 | +5 shapes |
| 3 | 0.0326 | 18 | +4 shapes |
| 4 | 0.0320 | 19 | +1 shape (best) |
| 5 | 0.0337 | 19 | Slight regression |

**Qualitative:** The LLM architect adds too many shapes per round (8-11).
Many are overlapping or redundant. The loss improves because more shapes =
more degrees of freedom, but the visual quality doesn't necessarily improve.
Final result looks "cluttered" compared to the clean 9-shape optimization.

**Bugs found and fixed:**
- Architect returned triangle vertices as `[x, y]` lists but Pydantic schema
  expected `dict[str, float]` -- fixed with `dict[str, Any]`
- Best-state restore crashed when shape count changed (9 vs 19 shapes) --
  fixed with `copy.deepcopy(shapes)` instead of state dict restoration
- Triangle modify was missing from `edit_parser.py` -- added

**Key insight:** LLM is good at naming shapes and understanding anatomy but
bad at spatial reasoning. Adding 8 shapes at once makes it impossible to
tell which ones help. Need per-shape evaluation.

---

## Experiment 3: Greedy Shape-Dropping Refinement

**Date:** 2026-02-13
**Commit:** `2a88795`
**Command:** `pelican greedy-refine --max-shapes 20 --settle-steps 100 --reoptimize-steps 200`

**Setup:** Replace LLM batch edits with greedy forward selection. Add one
shape at a time, two-phase trial:
1. **Phase A (Settle):** Freeze existing shapes, optimize only the new shape
   for 100 steps. Gradient descent pulls it to the best position.
2. **Phase B (Re-optimize):** Unfreeze all shapes, optimize together for 200
   steps. Everything adjusts to accommodate the newcomer.
3. **Decision:** If loss after both phases < loss before: keep. Otherwise: discard.

Shape types cycle: circle → ellipse → triangle → circle → ...
Random initial position, size, and intensity for each candidate.

**Algorithm:**
```
optimize(initial_shapes, 500 steps)
for round in 1..max_shapes:
  candidate = random_shape(type=cycling, scale=1.0)
  save_state()
  shapes.append(candidate)
  freeze(existing); optimize(100 steps); unfreeze()   # Phase A
  optimize(all, 200 steps)                             # Phase B
  if loss_after < loss_before: keep
  else: restore_state()
```

**Results:**

| Metric | Value |
|--------|-------|
| Final loss | 0.0259 |
| Shapes | 20 (9 initial + 11 added) |
| Accepted | 11/11 (100% acceptance) |
| Rejected | 0 |
| Improvement over baseline | 26% |
| Improvement over LLM refine | 19% |

**Per-round breakdown:**

| Round | Type | Loss Before | Loss After | Improvement | Decision |
|-------|------|-------------|------------|-------------|----------|
| 1 | circle | 0.0352 | 0.0350 | 0.0002 | Accepted |
| 2 | ellipse | 0.0350 | 0.0349 | 0.0001 | Accepted |
| 3 | triangle | 0.0349 | 0.0340 | 0.0009 | Accepted |
| 4 | circle | 0.0341 | 0.0331 | 0.0010 | Accepted |
| 5 | ellipse | 0.0333 | 0.0305 | 0.0028 | Accepted |
| 6 | triangle | 0.0309 | 0.0290 | 0.0018 | Accepted |
| 7 | circle | 0.0295 | 0.0290 | 0.0005 | Accepted |
| 8 | ellipse | 0.0293 | 0.0286 | 0.0007 | Accepted |
| 9 | triangle | 0.0289 | 0.0284 | 0.0005 | Accepted |
| 10 | circle | 0.0286 | 0.0284 | 0.0002 | Accepted |
| 11 | ellipse | 0.0286 | 0.0259 | 0.0027 | Accepted |

**Qualitative:** Much cleaner than LLM refine. Each shape earns its place.
The pelican has better-defined body contours, visible feet, and better tonal
variation. Two notable "big jumps" in rounds 5 and 11 (both ellipses) suggest
ellipses are particularly effective for covering large tonal areas.

**Key insights:**
- Gradient descent is excellent at placement -- no LLM needed for WHERE
- Every single shape improved loss -- the greedy approach with two-phase
  trial is highly effective
- Freezing existing shapes during settle prevents disruption to established
  layout
- Ellipses provide the biggest improvements (flexible, smooth coverage)
- No API key needed -- runs fully offline

---

## Summary: Algorithm Progression

| Experiment | Strategy | Loss | Shapes | vs Baseline |
|-----------|----------|------|--------|-------------|
| 1. Baseline | Gradient only | 0.0351 | 9 | -- |
| 2. LLM Refine | Batch edits + rollback | 0.0320 | 19 | -9% |
| 3. Greedy | Drop-and-optimize | **0.0259** | **20** | **-26%** |

## Next Experiments to Try

- **Error-guided placement:** Initialize new shapes at the location of
  highest per-pixel error instead of random position
- **Shape replacement:** Identify the least-helpful shape, remove it, try
  a different type/size in its place
- **Variable scale:** Try small shapes (scale=0.5) for fine details after
  large shapes (scale=2.0) are placed
- **Progressive resolution:** 32 → 64 → 128 to get coarse structure right
  before fine-tuning
- **LLM-guided shape selection:** Let an LLM suggest what TYPE of shape to
  add next (hybrid: LLM for strategy, gradient for placement)
- **Longer optimization:** More settle steps (200+) and re-optimize steps
  (500+) to see if results improve further
