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
Final loss: **0.0345** after 500 steps.

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

**Greedy refinement animation** (initial -> each accepted shape -> final):

![Greedy Refinement GIF](04_greedy_refinement.gif)

**Final** (35 shapes, 26 added, 3 rejected):

![Greedy Final](04_greedy_final.png)

## Metrics Summary

| Stage | Loss | Shapes | Notes |
|-------|------|--------|-------|
| Test render | N/A | 9 | Initial hard-coded geometry |
| Optimize (500 steps) | 0.0345 | 9 | Baseline single-run result |
| Greedy round 1 | 0.0343 | 10 | +circle, accepted |
| Greedy round 2 | 0.0342 | 11 | +ellipse, accepted |
| Greedy round 3 | 0.0343 | 12 | +triangle, accepted |
| Greedy round 4 | 0.0342 | 13 | +circle, accepted |
| Greedy round 5 | 0.0335 | 14 | +ellipse, accepted |
| Greedy round 6 | 0.0323 | 15 | +triangle, accepted (big improvement) |
| Greedy round 7 | 0.0320 | 16 | +circle, accepted |
| Greedy round 8 | 0.0314 | 17 | +ellipse, accepted |
| Greedy round 9 | 0.0311 | 18 | +triangle, accepted |
| Greedy round 10 | 0.0305 | 19 | +circle, accepted |
| Greedy round 13 | 0.0305 | 20 | +circle, accepted |
| Greedy round 14 | 0.0300 | 21 | +ellipse, accepted |
| Greedy round 16 | 0.0280 | 22 | +circle, accepted (big improvement) |
| Greedy round 17 | 0.0280 | 23 | +ellipse, accepted |
| Greedy round 18 | 0.0278 | 24 | +triangle, accepted |
| Greedy round 19 | 0.0278 | 25 | +circle, accepted |
| Greedy round 20 | 0.0271 | 26 | +ellipse, accepted |
| Greedy round 21 | 0.0272 | 27 | +triangle, accepted |
| Greedy round 22 | 0.0271 | 28 | +circle, accepted |
| Greedy round 23 | 0.0257 | 29 | +ellipse, accepted (big improvement) |
| Greedy round 24 | 0.0257 | 30 | +triangle, accepted |
| Greedy round 25 | 0.0255 | 31 | +circle, accepted |
| Greedy round 26 | 0.0253 | 32 | +ellipse, accepted |
| Greedy round 27 | 0.0253 | 33 | +triangle, accepted |
| Greedy round 28 | 0.0253 | 34 | +circle, accepted |
| Greedy round 29 | 0.0251 | 35 | +ellipse, accepted |
| **Greedy final** | **0.0251** | **35** | **27% better than optimize-only** |

## Observations

1. **Optimization works well**: 500 steps of gradient descent produces a
   recognizable pelican silhouette from the initial 9-shape geometry.

2. **Greedy refinement is highly effective**: 26 of 29 candidate shapes were
   accepted. Loss dropped 27% from 0.0345 to 0.0251. The two-phase
   approach (settle then re-optimize) lets each shape find its best placement
   before the whole scene adjusts.

3. **Gradient-stable ellipse SDF**: Using the normalized-distance approximation
   (scaled by geometric mean radius) instead of the Quilez analytical method
   eliminates all NaN gradient instabilities. The visual difference is
   imperceptible after sigmoid smoothing at 128x128 resolution. This allows
   the pipeline to reach the full 35-shape budget without any NaN issues.

4. **No LLM needed for placement**: Gradient descent handles WHERE to put
   each shape. The greedy loop just decides WHAT to add (cycling through
   circle, ellipse, triangle). No API key required.

5. **Key areas for further improvement**:
   - Shape replacement: swap out the least-helpful shape for a different type
   - Error-guided placement: initialize new shapes near highest-error regions
   - LLM-guided shape selection: let an LLM suggest what to try next
