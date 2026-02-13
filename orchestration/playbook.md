# Differentiable Pelican Pipeline Playbook

An LLM-walkable playbook for running the full differentiable pelican pipeline
end-to-end. Each step includes the command, expected output, and what to check
before proceeding.

## Prerequisites

- Python 3.11+ with uv installed
- Dependencies synced: `uv sync`
- For refinement: `ANTHROPIC_API_KEY` set in `.env.local`

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
- `out/optimize/optimization.gif` -- Animation of the optimization process
- `out/optimize/metrics.json` -- Loss history and final metrics
- `out/optimize/frames/` -- Individual frame PNGs

**Check:**
- Final loss should be approximately 0.035 or lower
- The optimized PNG should resemble the target pelican more closely than the
  test render
- The GIF should show gradual convergence

**If it fails:** Check that `images/pelican-drawing-1.jpg` exists. If loss
diverges (NaN), try reducing `--lr 0.01`.

## Step 3: Refine (requires API key)

Run the full LLM refinement loop: optimize, judge, architect edits, re-optimize.

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
  - `optimized.png`, `optimized.svg` -- Round render
  - `feedback.json` -- Judge evaluation
  - `architect.json` -- Proposed edits
- `out/refine/final/` -- Best shapes from all rounds
  - `pelican_final.png`, `pelican_final.svg`
- `out/refine/refinement_history.json` -- Full round-by-round metrics

**Check:**
- Final loss should be approximately 0.032 or lower (better than optimize-only)
- The final PNG should look like a well-formed pelican
- The refinement history should show loss generally decreasing across rounds
- Shape count may increase as the architect adds anatomical details

**If it fails:**
- Missing API key: Set `ANTHROPIC_API_KEY` in `.env.local`
- Rate limits: The client retries automatically with exponential backoff
- If all rounds roll back, the architect prompt may need tuning

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

# Refinement
cp out/refine/round_00/optimized.png docs/results/03_refine_round0.png
cp out/refine/final/pelican_final.png docs/results/04_refine_final.png
cp out/refine/final/pelican_final.svg docs/results/04_refine_final.svg
```

**Check:** All files exist in `docs/results/` and match the latest run.

## Step 5: Update Results Documentation

Update `docs/results/README.md` with the latest metrics from
`out/refine/refinement_history.json` and `out/optimize/metrics.json`.

Key values to update:
- Optimize final loss (from `metrics.json`)
- Per-round loss and shape count (from `refinement_history.json`)
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
uv run pelican refine --target images/pelican-drawing-1.jpg \
  --rounds 3 --steps-per-round 100 --resolution 64 --output-dir out/quick_refine
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `uv sync` |
| `ANTHROPIC_API_KEY not found` | Create `.env.local` with your key |
| Loss goes to NaN | Reduce learning rate: `--lr 0.01` |
| GIF not created | Install imageio: `uv add imageio` |
| Refine final looks bad | Bug was fixed: best shapes now deep-copied |
| Rate limit errors | Client retries automatically (2s, 4s, 8s backoff) |
