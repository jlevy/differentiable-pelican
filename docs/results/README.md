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

### 3. Refinement Loop (LLM Judge + Architect)

Multi-round refinement with Claude Sonnet 4 judge and architect:

**Round 1** (post-optimization, before LLM edits):

![Round 0](03_refine_round0.png)

**Final** (after 4 rounds, 2 rollbacks):

![Final](04_refine_final.png)

## Metrics Summary

| Stage | Loss | Shapes | Notes |
|-------|------|--------|-------|
| Test render | N/A | 9 | Initial hard-coded geometry |
| Optimize (500 steps) | 0.0351 | 9 | Best single-run result |
| Refine round 1 | 0.0382 | 9 | First optimization in refinement |
| Refine round 2 | 0.0401 | 10 | LLM added throat_pouch |
| Refine round 3 | 0.0722 | 12 | Degraded, rolled back |
| Refine final | 0.0382 | 12 | Best from round 1 (shape count mismatch) |

## Observations

1. **Optimization works well**: 500 steps of gradient descent produces a
   recognizable pelican silhouette from the initial geometry.

2. **LLM refinement needs improvement**: The architect's edits (adding duplicate
   shapes, overlapping primitives) currently degrade quality. The rollback
   mechanism correctly catches this.

3. **Key areas for improvement**:
   - Architect prompt needs more context about current shape layout
   - Edit validation should reject duplicate shape names
   - Consider warm-starting optimization with existing params after edits
