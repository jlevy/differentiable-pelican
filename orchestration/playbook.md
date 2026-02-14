# Differentiable Pelican Pipeline Playbook

An LLM-walkable playbook for running the full differentiable pelican pipeline
end-to-end. Each step includes the command, expected output, and what to check
before proceeding.

## Prerequisites

- Python 3.11+ with uv installed
- Dependencies synced: `uv sync`
- For LLM refinement (Step 3b only): `ANTHROPIC_API_KEY` set in `.env.local`

## Step 1: Test Render

Render the initial hard-coded pelican geometry (no optimization).

```bash
uv run pelican test-render --resolution 128 --output-dir out/test_render
```

**Expected output:**
- `out/test_render/pelican_test.png` -- 9-shape pelican silhouette
- `out/test_render/pelican_test.svg` -- SVG export

**Check:** Open the PNG. You should see a recognizable pelican shape with 9
anatomical parts (body, neck, head, beak_upper, beak_lower, wing, tail, eye,
feet) in grayscale.

**If it fails:** Verify `uv sync` completed successfully and the package is
installed.

## Step 2: Optimize

Run gradient descent to match the target image.

```bash
uv run pelican optimize \
  --target images/pelican-drawing-1.jpg \
  --steps 500 \
  --resolution 128 \
  --save-every 50 \
  --output-dir out/optimize
```

**Expected output:**
- `out/optimize/pelican_optimized.png` -- Optimized render
- `out/optimize/pelican_optimized.svg` -- SVG export
- `out/optimize/optimization.gif` -- Animation (400ms/frame, includes final best-param frame)
- `out/optimize/metrics.json` -- Loss history and final metrics
- `out/optimize/frames/` -- Individual frame PNGs

**Check:**
- Final loss should be approximately 0.035 or lower
- The optimized PNG should resemble the target pelican more closely than the
  test render
- The GIF should show gradual convergence with the final frame matching the PNG

**If it fails:** Check that `images/pelican-drawing-1.jpg` exists. If loss
diverges (NaN), try reducing `--lr 0.01`.

## Step 3: Greedy Refinement (recommended)

Add shapes one at a time. Each candidate goes through a two-phase trial:
1. **Settle**: Freeze existing shapes, optimize only the new shape (finds best placement)
2. **Re-optimize**: Unfreeze all, optimize together (adjusts for the newcomer)

Keep the shape if loss improves after both phases; discard otherwise.

```bash
uv run pelican greedy-refine \
  --target images/pelican-drawing-1.jpg \
  --resolution 128 \
  --max-shapes 20 \
  --initial-steps 500 \
  --settle-steps 100 \
  --reoptimize-steps 200 \
  --max-failures 5 \
  --output-dir out/greedy_refine
```

**Expected output:**
- `out/greedy_refine/round_00_initial/` -- Initial 9-shape optimization
- `out/greedy_refine/round_NN_accept_TYPE/` -- Per-accepted-shape outputs
  - `optimized.png`, `optimized.svg`
- `out/greedy_refine/final/` -- Best shapes from all rounds
  - `pelican_final.png`, `pelican_final.svg`
- `out/greedy_refine/greedy_history.json` -- Full round-by-round decisions

**Check:**
- Loss should decrease incrementally with each accepted shape
- Rejected shapes should show loss_after >= loss_before in history
- Final PNG should look like a clean, detailed pelican
- Final shape count should be 9 + (number of accepted shapes)

**Key parameters to tune:**
- `--settle-steps 100`: More steps = new shape finds better placement (slower)
- `--reoptimize-steps 200`: More steps = better global optimization (slower)
- `--max-shapes 20`: Shape budget; fewer = cleaner, more = more detail
- `--scale 1.0`: Size of new shapes (0.5 = small details, 2.0 = large shapes)
- `--no-freeze`: Let all shapes move during settle (more freedom, less stable)

**If it fails:** No API key needed (no LLM). If loss doesn't improve at all,
try `--scale 0.5` for smaller shapes or `--settle-steps 200` for more
optimization time per candidate.

## Step 3b: LLM Refinement (alternative, requires API key)

Uses an LLM judge and architect to propose structural edits. More creative
but less predictable than greedy refinement.

```bash
uv run pelican refine \
  --target images/pelican-drawing-1.jpg \
  --rounds 5 \
  --steps-per-round 200 \
  --resolution 128 \
  --output-dir out/refine
```

**Expected output:**
- `out/refine/round_00/` through `out/refine/round_04/` -- Per-round outputs
- `out/refine/final/pelican_final.png`, `pelican_final.svg`
- `out/refine/refinement_history.json`

**If it fails:**
- Missing API key: Set `ANTHROPIC_API_KEY` in `.env.local`
- Rate limits: The client retries automatically with exponential backoff

## Step 4: Collect Results

Copy the key outputs to `docs/results/` for documentation.

```bash
# Target image (only needed once)
cp images/pelican-drawing-1.jpg docs/results/00_target.jpg

# Test render
cp out/test_render/pelican_test.png docs/results/01_test_render.png
cp out/test_render/pelican_test.svg docs/results/01_test_render.svg

# Optimization
cp out/optimize/pelican_optimized.png docs/results/02_optimized.png
cp out/optimize/pelican_optimized.svg docs/results/02_optimized.svg
cp out/optimize/optimization.gif docs/results/02_optimization.gif

# Greedy refinement (recommended)
cp out/greedy_refine/round_00_initial/optimized.png docs/results/03_greedy_initial.png
cp out/greedy_refine/final/pelican_final.png docs/results/04_greedy_final.png
cp out/greedy_refine/final/pelican_final.svg docs/results/04_greedy_final.svg
cp out/greedy_refine/greedy_refinement.gif docs/results/04_greedy_refinement.gif
cp out/greedy_refine/pipeline_stages.svg docs/results/pipeline_stages.svg
```

**Check:** All files exist in `docs/results/` and match the latest run.

## Step 5: Update Results Documentation

Update `docs/results/README.md` with the latest metrics from
`out/greedy_refine/greedy_history.json` and `out/optimize/metrics.json`.

Key values to update:
- Optimize final loss (from `metrics.json`)
- Per-round accept/reject decisions (from `greedy_history.json`)
- Shapes added vs rejected
- Any changes to observations

## Step 6: Run Tests

Verify nothing is broken.

```bash
uv run pytest src/ -x -q       # Unit tests (37 expected)
uv run ruff check src/          # Linting
uv run basedpyright src/        # Type checking
```

**Expected:** All tests pass, 0 lint errors, 0 type errors.

## Step 7: Commit and Push

```bash
git add docs/results/ orchestration/ src/ docs/
git commit -m "docs: update pipeline results and playbook"
git push -u origin <branch-name>
```

## Quick Run (All Steps)

For a fast end-to-end run (lower resolution, fewer steps):

```bash
uv run pelican test-render --resolution 64 --output-dir out/quick_test
uv run pelican optimize --target images/pelican-drawing-1.jpg \
  --steps 100 --resolution 64 --output-dir out/quick_opt
uv run pelican greedy-refine --target images/pelican-drawing-1.jpg \
  --resolution 64 --max-shapes 15 --initial-steps 200 \
  --settle-steps 50 --reoptimize-steps 100 --output-dir out/quick_greedy
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `uv sync` |
| `ANTHROPIC_API_KEY not found` | Only needed for `refine`, not `greedy-refine` |
| Loss goes to NaN | Reduce learning rate: `--lr 0.01` |
| GIF not created | Install imageio: `uv add imageio` |
| No shapes accepted | Try `--scale 0.5` or `--settle-steps 200` |
| Too many shapes | Reduce `--max-shapes` or increase `--max-failures` |
| Rate limit errors | Client retries automatically (2s, 4s, 8s backoff) |
