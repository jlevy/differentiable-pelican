# Differentiable Pelican

**What if you could teach a computer to draw a pelican by telling it
"you're wrong" 10,000 times, and letting calculus do the rest?**

![Greedy refinement, shape by shape](docs/results/05_greedy_extended.gif)

This project approximates a target pelican image using circles,
ellipses, and triangles whose parameters are optimized end-to-end
via gradient descent through a differentiable renderer. No neural
network. No pixel buffer. Just shapes, a loss function, and
backpropagation.

## Why This Is Interesting

Most rendering pipelines are black boxes to an optimizer: you put
parameters in, you get pixels out, but you can't ask "how should I
nudge this ellipse to make the output look more like the target?"

Differentiable rendering changes that. By implementing the renderer
in a framework that tracks gradients (here, PyTorch), every pixel
in the output carries information about how it depends on each
shape's position, size, rotation, and intensity. We can compute
the gradient of a loss function (measuring how far the rendered
image is from the target) with respect to every shape parameter at
once, then follow those gradients to improve the image iteratively.

This is the same principle behind training neural networks --
define a loss, backpropagate, update parameters -- but applied to
an interpretable, symbolic representation rather than millions of
opaque weights. The shapes remain editable SVG primitives
throughout.

Several aspects make this a useful illustration of differentiable
programming techniques that appear in much larger systems:

- **Continuous relaxation of discrete structure.** An SVG circle is
  either there or it isn't, but we use
  [soft signed distance fields](https://iquilezles.org/articles/distfunctions2d/)
  with a sigmoid to make each shape's coverage a smooth function
  of its parameters. This trick -- replacing hard decisions with
  soft approximations -- appears throughout differentiable
  programming, from
  [attention mechanisms](https://arxiv.org/abs/1706.03762) to
  [differentiable sorting](https://arxiv.org/abs/2002.08871).

- **Compositing as a differentiable program.** Shapes are layered
  with [Porter-Duff](https://dl.acm.org/doi/10.1145/800031.808606)
  alpha compositing, where each shape's soft coverage acts as an
  alpha mask. The full pipeline -- raw parameters through SDF
  evaluation, sigmoid activation, and layer composition -- forms a
  single differentiable computation graph. Gradients flow from the
  pixel-level loss back to every shape parameter in one backward
  pass.

- **Composite loss design.** The loss combines pixel MSE,
  [structural similarity (SSIM)](https://doi.org/10.1109/TIP.2003.819861),
  Sobel edge matching, and geometric priors (overlap penalties,
  boundary constraints, degeneracy guards). Balancing pixel accuracy
  against structural and regularization objectives is a recurring
  design challenge in differentiable systems, from image
  reconstruction to physics simulation.

- **Greedy search over discrete topology.** Gradient descent
  optimizes continuous parameters but can't decide _whether_ to add
  a shape. We use greedy forward selection: propose a candidate, let
  gradient descent find its optimal placement, keep it only if loss
  improves. This interplay between discrete search (what to add) and
  continuous optimization (where to put it) parallels patterns in
  neural architecture search, program synthesis, and
  mixture-of-experts routing.

## Results

Starting from 9 hand-coded shapes and a
[vintage pelican engraving](images/pelican-drawing-1.jpg) as the
target, the pipeline first optimizes via gradient descent, then
greedily adds shapes one at a time. Each candidate is placed by
gradient descent alone -- no heuristics, no LLM, no human in the
loop.

### Target and baseline optimization (9 shapes, 500 steps)

<p>
<img src="docs/results/00_target.jpg" width="200" alt="Target pelican engraving"/>
&nbsp;&nbsp;
<img src="docs/results/02_optimized.png" width="200" alt="Optimized with 9 shapes"/>
</p>

![Optimization animation](docs/results/02_optimization.gif)

### Greedy refinement (up to 35 shapes)

Each round: freeze existing shapes, optimize only the newcomer for
100 steps (settle phase), then unfreeze and re-optimize all shapes
together for 200 steps (joint phase). Keep only if loss drops.
All 26 candidates were accepted.

![Extended greedy refinement animation](docs/results/05_greedy_extended.gif)

| Stage | Loss | Shapes | vs Baseline |
|-------|------|--------|-------------|
| Optimize (500 steps) | 0.0351 | 9 | -- |
| Greedy (20 shapes) | 0.0259 | 20 | -26% |
| Greedy (35 shapes) | 0.0238 | 35 | -32% |

<p>
<img src="docs/results/04_greedy_final.png" width="200" alt="20 shapes final"/>
&nbsp;&nbsp;
<img src="docs/results/05_greedy_extended_final.png" width="200" alt="35 shapes final"/>
</p>

Per-round metrics and observations are in the
[research log](docs/research-log.md). The full image progression
is in [detailed results](docs/results/README.md).

## Quick Start

```bash
uv sync

# Render the initial hard-coded pelican (no optimization)
pelican test-render --resolution 128

# Optimize against the target image
pelican optimize --target images/pelican-drawing-1.jpg --steps 500

# Greedy refinement (no API key needed)
pelican greedy-refine --max-shapes 35

# LLM refinement loop (requires ANTHROPIC_API_KEY)
pelican refine --target images/pelican-drawing-1.jpg --rounds 5
```

## How It Works

```
Target Image  -->  Differentiable Renderer  -->  Loss Function
     ^                    |                          |
     |              Soft SDF + alpha-over      MSE + SSIM + Edge
     |                    |                          |
     |              Shape Parameters  <---  Gradient Descent (Adam)
     |                    |
     |              Greedy Refinement: add one shape, optimize, keep if better
     |                    |
     |              (Optional) LLM Judge + Architect for structural edits
     |                    |
     └────────────  Refinement Loop
```

1. **Differentiable rendering.** Each shape is evaluated as a soft
   SDF on a pixel grid, converted to coverage via sigmoid, and
   composited back-to-front. All ops are PyTorch tensors, so
   gradients flow from pixels to shape parameters.

2. **Gradient descent.** Adam with cosine LR annealing and tau
   (softness) scheduling minimizes a composite loss against the
   target image. Tau starts large (blurry shapes, strong gradients)
   and anneals to small (crisp edges).

3. **Greedy refinement.** Candidate shapes are proposed one at a
   time, cycling through circle/ellipse/triangle. Gradient descent
   handles all placement decisions; the loop only decides whether
   each shape earned its keep.

4. **LLM refinement (optional).** A multimodal LLM judge evaluates
   the current image + SVG, an architect proposes structural edits,
   and the system rolls back automatically on quality degradation.

## Design

The full design document -- covering the differentiable rendering
approach, loss function design, LLM integration, and rationale for
key decisions -- is the
[Pelican Plan](docs/design/pelican-plan.md).

## CLI Commands

| Command | Description |
|---------|-------------|
| `pelican test-render` | Render initial geometry without optimization |
| `pelican optimize` | Optimize shapes to match target image |
| `pelican greedy-refine` | Greedy forward-selection refinement loop |
| `pelican judge` | Evaluate optimized SVG with LLM |
| `pelican refine` | Full refinement loop with LLM feedback |
| `pelican validate-image` | Validate a rendered image with LLM |

## Docs

- [Installation](docs/installation.md) -- uv and Python setup
- [Development](docs/development.md) -- dev workflows
- [Publishing](docs/publishing.md) -- PyPI publishing
- [Research log](docs/research-log.md) -- experiment history

---

*Built from [simple-modern-uv](https://github.com/jlevy/simple-modern-uv).*
