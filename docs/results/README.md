# End-to-End Pipeline Results

Full pipeline run on 2026-02-14 at 128x128 resolution on CPU.

## Pipeline Stages

### 0. Target Image

The vintage engraving pelican from Public Domain Pictures:

![Target](00_target.jpg)

### 1. Test Render (Initial Geometry)

9-shape hard-coded pelican with named anatomical parts (body, neck, head,
beak_upper, beak_lower, wing, tail, eye, feet), each with grayscale intensity:

![Test Render](01_test_render.png)

### 2. Optimization (500 steps, Adam, lr=0.02)

Gradient descent minimizing MSE + SSIM + edge + geometric priors.
Final loss: **0.0341** after 500 steps.

![Optimized](02_optimized.png)

Optimization animation:

![Optimization GIF](02_optimization.gif)

### 3. Greedy Refinement

Greedy forward selection: add one shape at a time, let gradient descent
find optimal placement, keep only if loss improves.

Two-phase trial per candidate:
1. **Settle** (100 steps): Freeze existing shapes, optimize only new shape
2. **Re-optimize** (200 steps): Unfreeze all, optimize together

**After initial optimization** (9 shapes, same as Step 2):

![Greedy Initial](03_greedy_initial.png)

**Greedy refinement animation** (initial → each accepted shape → final):

![Greedy Refinement GIF](04_greedy_refinement.gif)

**Final** (24 shapes, 15 added, 6 rejected):

![Greedy Final](04_greedy_final.png)

## Metrics Summary

| Stage | Loss | Shapes | Notes |
|-------|------|--------|-------|
| Test render | N/A | 9 | Initial hard-coded geometry |
| Optimize (500 steps) | 0.0341 | 9 | Baseline single-run result |
| Greedy round 1 | 0.0334 | 10 | +circle, accepted |
| Greedy round 2 | 0.0326 | 11 | +ellipse, accepted |
| Greedy round 3 | 0.0326 | 12 | +triangle, accepted |
| Greedy round 4 | 0.0314 | 13 | +circle, accepted |
| Greedy round 5 | 0.0295 | 14 | +ellipse, accepted (big improvement) |
| Greedy round 6 | 0.0283 | 15 | +triangle, accepted |
| Greedy round 7 | 0.0280 | 16 | +circle, accepted |
| Greedy round 8 | 0.0278 | 17 | +ellipse, accepted |
| Greedy round 9 | 0.0275 | 18 | +triangle, accepted |
| Greedy round 10 | 0.0274 | 19 | +circle, accepted |
| Greedy round 11 | 0.0275 | 20 | +ellipse, accepted |
| Greedy round 12 | 0.0276 | 21 | +triangle, accepted |
| Greedy round 13 | 0.0275 | 22 | +circle, accepted |
| Greedy round 14 | 0.0275 | 23 | +ellipse, accepted |
| Greedy round 16 | 0.0264 | 24 | +circle, accepted (plateau-breaker) |
| **Greedy final** | **0.0264** | **24** | **23% better than optimize-only** |

## Observations

1. **Optimization works well**: 500 steps of gradient descent produces a
   recognizable pelican silhouette from the initial 9-shape geometry.

2. **Greedy refinement is effective**: 15 of 21 candidate shapes were
   accepted. Loss dropped 23% from 0.0341 to 0.0264. The two-phase
   approach (settle then re-optimize) lets each shape find its best placement
   before the whole scene adjusts.

3. **NaN instability at high shape counts**: After ~24 shapes, NaN losses
   begin appearing during optimization, causing most candidates to be rejected.
   This is due to the analytical ellipse SDF (Quilez method) which has
   numerical edge cases in its cubic solver. A lower learning rate or gradient
   clipping may help extend the shape budget.

4. **No LLM needed for placement**: Gradient descent handles WHERE to put
   each shape. The greedy loop just decides WHAT to add (cycling through
   circle, ellipse, triangle). No API key required.

5. **Key areas for further improvement**:
   - Gradient clipping or adaptive LR to reduce NaN instabilities at high shape counts
   - Shape replacement: swap out the least-helpful shape for a different type
   - Error-guided placement: initialize new shapes near highest-error regions
   - LLM-guided shape selection: let an LLM suggest what to try next
