# End-to-End Pipeline Results

Full pipeline run on 2026-02-13 at 128x128 resolution on CPU.

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
Final loss: **0.0351** after 500 steps.

![Optimized](02_optimized.png)

Optimization animation:

![Optimization GIF](02_optimization.gif)

### 3. Greedy Refinement (shape-dropping)

Greedy forward selection: add one shape at a time, let gradient descent
find optimal placement, keep only if loss improves.

Two-phase trial per candidate:
1. **Settle** (100 steps): Freeze existing shapes, optimize only new shape
2. **Re-optimize** (200 steps): Unfreeze all, optimize together

**After initial optimization** (9 shapes, same as Step 2):

![Greedy Initial](03_greedy_initial.png)

**Final** (20 shapes, 11 added, 0 rejected):

![Greedy Final](04_greedy_final.png)

## Metrics Summary

| Stage | Loss | Shapes | Notes |
|-------|------|--------|-------|
| Test render | N/A | 9 | Initial hard-coded geometry |
| Optimize (500 steps) | 0.0351 | 9 | Baseline single-run result |
| Greedy round 1 | 0.0350 | 10 | +circle, accepted |
| Greedy round 2 | 0.0349 | 11 | +ellipse, accepted |
| Greedy round 3 | 0.0340 | 12 | +triangle, accepted |
| Greedy round 4 | 0.0331 | 13 | +circle, accepted |
| Greedy round 5 | 0.0305 | 14 | +ellipse, accepted (big improvement) |
| Greedy round 6 | 0.0290 | 15 | +triangle, accepted |
| Greedy round 7 | 0.0290 | 16 | +circle, accepted |
| Greedy round 8 | 0.0286 | 17 | +ellipse, accepted |
| Greedy round 9 | 0.0284 | 18 | +triangle, accepted |
| Greedy round 10 | 0.0284 | 19 | +circle, accepted |
| Greedy round 11 | 0.0259 | 20 | +ellipse, accepted (big improvement) |
| **Greedy final** | **0.0259** | **20** | **26% better than optimize-only** |

## Observations

1. **Optimization works well**: 500 steps of gradient descent produces a
   recognizable pelican silhouette from the initial 9-shape geometry.

2. **Greedy refinement is highly effective**: Every candidate shape was
   accepted (11/11). Loss dropped 26% from 0.0351 to 0.0259. The two-phase
   approach (settle then re-optimize) lets each shape find its best placement
   before the whole scene adjusts.

3. **No LLM needed for placement**: Gradient descent handles WHERE to put
   each shape. The greedy loop just decides WHAT to add (cycling through
   circle, ellipse, triangle). No API key required.

4. **Key areas for further improvement**:
   - Shape replacement: swap out the least-helpful shape for a different type
   - Error-guided placement: initialize new shapes near highest-error regions
   - LLM-guided shape selection: let an LLM suggest what to try next
