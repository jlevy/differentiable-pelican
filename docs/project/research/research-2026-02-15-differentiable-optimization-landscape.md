# Research: Differentiable Optimization Landscape for SVG Generation

**Date:** 2026-02-15

**Status:** Complete

## Overview

This document surveys the landscape of differentiable programming techniques
relevant to the Differentiable Pelican project -- a system that optimizes SVG
primitives via differentiable rendering (soft SDFs in PyTorch) with optional
LLM-guided structural refinement.

The project sits at the intersection of several active research areas:
differentiable solvers, textual gradients (LLM-as-optimizer), compound AI
systems, discrete structure relaxation, and test-time optimization. This
brief maps the relevant prior work, identifies where the project's approach
aligns with or diverges from the state of the art, and highlights future
directions.

## Questions Answered

1. What differentiable optimization frameworks exist, and how does Pelican's
   soft-SDF renderer relate to them?
2. How do "textual gradient" systems (ProTeGi, TextGrad, DSPy) compare to
   Pelican's LLM judge/architect loop?
3. What is the "compound AI systems" framing, and how does it describe
   Pelican's hybrid gradient+LLM architecture?
4. What techniques exist for relaxing discrete structure decisions (topology,
   shape type) into differentiable operations?
5. How does Pelican's iterative refinement relate to test-time compute scaling?

## Scope

**Included:** Differentiable solvers, differentiable vector graphics, textual
gradient methods, compound AI system design, discrete relaxation techniques
(Gumbel-Softmax, STE), test-time optimization. Focus on systems published
2017-2025 with direct relevance to SVG optimization.

**Excluded:** Neural radiance fields (NeRFs), 3D differentiable rendering,
diffusion-based image generation, GANs for vector graphics. These are related
but address different problem formulations.

---

## Findings

### 1. Differentiable Solvers and Optimization Frameworks

The core idea behind Differentiable Pelican -- treating an entire program as a
differentiable function and optimizing its inputs via gradients -- is shared by
a growing family of differentiable solver frameworks. These systems embed
optimization procedures as differentiable layers, enabling end-to-end
backpropagation through structured computations.

**JAXopt** (Blondel et al., NeurIPS 2022) provides implicit differentiation
through optimization solvers in JAX. Users define optimality conditions in
Python, and the framework handles differentiation through the solver's fixed
point without manual derivation. Supports both implicit differentiation and
autodiff of unrolled iterates. Note: JAXopt is no longer actively maintained;
some features have been ported into Optax.

**cvxpylayers** (Agrawal et al., NeurIPS 2019) embeds parametrized convex
optimization problems as differentiable layers in neural networks. Any
disciplined convex program written in CVXPY decomposes into an affine map, a
solver call, and another affine map -- each differentiable. Supports PyTorch,
JAX, and MLX backends. Relevant if SVG constraints (non-overlap, bounding-box
containment) were formulated as convex programs.

**Differentiable MPC** (Amos et al., NeurIPS 2018) uses Model Predictive
Control as a differentiable policy class. Differentiates through the
controller via KKT conditions of the convex approximation at a fixed point.
Demonstrates that structured optimization procedures (planning, control) can
be made differentiable and embedded in learning loops.

**DiLQR** (Wang et al., ICML 2025) provides analytical gradients for iLQR
controllers via implicit differentiation, achieving O(1) backward-pass
complexity regardless of forward iteration count -- up to 128x speedup over
autodiff through unrolled iterates. Shows that even iterative solvers can be
differentiated efficiently.

**MPAX** (Lu et al., MIT, 2024) is a hardware-accelerated, differentiable
solver for large-scale LP and QP natively in JAX. Implements PDHG variants
with adaptive restarts and supports multi-GPU sharding.

**Theseus** (Pineda et al., Meta AI, NeurIPS 2022) is the closest
architectural analog in PyTorch: an application-agnostic library for
differentiable nonlinear least squares optimization. Provides a common
framework for end-to-end structured learning with efficient sparse solvers
and custom C++/CUDA backends. If Pelican moved to a least-squares formulation
of shape fitting, Theseus could be used directly.

**Connection to Pelican:** All of these frameworks share the insight that
computational procedures (solvers, controllers, renderers) can be treated as
differentiable functions. Pelican instantiates this pattern in the domain of
vector graphics: the "solver" is a soft-SDF renderer with Porter-Duff
compositing, and the "inputs" are shape parameters. Pelican's approach is
simpler than general-purpose frameworks (pure PyTorch autograd, no custom
backward passes) but limited to SDF-representable primitives.

### 2. Textual Gradients and LLM-as-Optimizer

A recent line of work replaces numerical gradients with natural language
critiques, enabling LLMs to optimize non-differentiable systems. Pelican's
LLM judge/architect loop is a practical instance of this paradigm.

**ProTeGi** (Pryzant et al., EMNLP 2023) pioneered the "textual gradient"
metaphor. It uses LLMs to generate natural language critiques of prompts --
describing how a prompt should change to reduce errors -- then edits the
prompt in the opposite semantic direction. A beam search with bandit selection
guides exploration. Achieves up to +31 percentage points in F1 over baselines
on tasks like jailbreak detection.

**TextGrad** (Yuksekgonul et al., published in *Nature* vol. 639, 2025;
first arXiv June 2024) generalizes backpropagation to compound AI systems.
Each component is a node in a computation graph; textual feedback propagates
as "gradients" through arbitrary, non-differentiable functions (LLM calls,
simulators, external solvers). Provides a PyTorch-like API. This is the most
directly relevant framework for Pelican's LLM-guided refinement -- Pelican's
judge/architect architecture is a two-node computation graph expressible in
TextGrad's formalism. The LLM's critique of the SVG render is literally a
textual gradient driving structural changes.

**DSPy** (Khattab et al., Stanford, NeurIPS 2023) abstracts LM pipelines as
text transformation graphs with declarative modules (analogous to PyTorch
layers). A compiler automatically optimizes modules by bootstrapping few-shot
demonstrations and tuning prompts to maximize a user-defined metric. DSPy
could formalize Pelican's LLM pipeline: instead of hand-crafting judge and
architect prompts, DSPy's compiler could optimize them against loss-reduction
metrics.

**OPRO** (Yang et al., Google DeepMind, 2023) uses LLMs as general-purpose
optimizers via a meta-prompt containing previously generated solutions paired
with their scores. At each step, the LLM generates new candidates, which are
evaluated and fed back. Applied to prompt optimization (up to 8% improvement
on GSM8K, up to 50% on Big-Bench Hard) and even demonstrated on linear
regression and TSP.

**Connection to Pelican:** These methods collectively validate the approach
of using LLMs as optimizers for non-differentiable decisions. Pelican's
hybrid architecture -- numerical gradients for continuous parameters
(position, size, rotation, intensity), textual gradients for discrete
structure (add/remove/modify shapes) -- is a natural instantiation of this
paradigm. OPRO's meta-prompt strategy of including previous rounds' loss
values to help the LLM learn from the optimization trajectory is directly
applicable to Pelican's refinement loop.

### 3. Compound AI Systems

**"The Shift from Models to Compound AI Systems"** (Zaharia, Khattab, Chen,
Davis, Miller, Potts, Zou, Carbin, Frankle, Rao, Ghodsi; BAIR Blog,
February 2024) argues that state-of-the-art AI results increasingly come from
compound systems with multiple interacting components, not monolithic models.
At Databricks, 60% of LLM applications use RAG, 30% use multi-step chains.
The post identifies design, end-to-end optimization, and operation as key
open challenges.

Differentiable Pelican is a textbook compound AI system:
- **Gradient optimizer** -- continuous parameter refinement via backpropagation
  through the soft-SDF renderer
- **Greedy forward selection** -- discrete topology decisions (add a shape,
  keep if loss drops)
- **LLM judge** -- evaluates current render quality using multimodal
  understanding
- **LLM architect** -- proposes structural edits based on the judge's critique
- **Rollback mechanism** -- rejects changes that degrade quality

Each component has a different strength: the gradient optimizer handles
continuous tuning efficiently; the greedy search handles discrete topology
decisions; the LLM provides high-level structural reasoning that neither
gradient descent nor greedy search can achieve.

**Trace** (Microsoft Research + Stanford, NeurIPS 2024) is the most general
framework for optimizing such systems. It treats computational workflows as
graphs and optimizes heterogeneous parameters (prompts, hyperparameters, code)
using execution traces as the analog of backpropagated gradients. The OPTO
(Optimization with Trace Oracle) formalism uses LLMs to propose parameter
updates based on execution traces and rich feedback. Trace achieves 10%
higher accuracy than hand-designed DSPy optimizers on BigBenchHard.

**Implication for Pelican:** The entire pipeline -- renderer, loss function,
greedy search, LLM judge, LLM architect -- could be expressed as a Trace
computation graph, with OPTO jointly optimizing all components (tau schedule,
loss weights, judge prompts, architect prompts). This would automate the
meta-optimization that currently requires manual tuning.

### 4. Differentiable Logic and Discrete Structure Relaxation

SVG optimization involves both continuous parameters (position, size,
rotation, intensity) and discrete structural decisions (number of shapes,
shape types, layer ordering, add/remove). Several techniques exist for
relaxing discrete decisions into differentiable operations.

**Gumbel-Softmax** (Jang, Gu, Poole, ICLR 2017; independently Maddison,
Mnih, Teh, ICLR 2017) provides a differentiable approximation to sampling
from categorical distributions. As temperature approaches 0, samples become
one-hot (discrete); at higher temperatures, they are smooth and
differentiable. In Pelican, discrete decisions -- which shape type to add,
whether to keep or remove a shape, layer ordering -- could be parameterized
as categorical distributions and relaxed via Gumbel-Softmax.

**Straight-Through Estimator (STE)** (Bengio, Leonard, Courville, 2013)
passes gradients through non-differentiable operations unchanged in the
backward pass, treating them as the identity function. Despite lacking
theoretical justification, it works well empirically and is standard for
training quantized and binary neural networks. Relevant to Pelican's
potential extension to hard SDF boundaries: crisp rendering in the forward
pass, smooth gradients in the backward pass.

**DiffVG** (Li, Lukac, Gharbi, Ragan-Kelley; SIGGRAPH Asia 2020) is the
most direct prior art for Differentiable Pelican. It is a differentiable
vector graphics rasterizer supporting full SVG paths (linear, quadratic,
cubic Bezier), ellipses, circles, and rectangles. Computes gradients of pixel
values with respect to curve parameters using analytical prefiltering or
multisampling anti-aliasing. Key differences from Pelican: DiffVG supports
arbitrary SVG paths via analytical anti-aliasing differentiation; Pelican uses
soft SDFs with sigmoid coverage. DiffVG is a C++/CUDA rasterizer with Python
bindings; Pelican is pure PyTorch. Pelican's approach is simpler and more
portable but limited to SDF-representable primitives.

**LIVE** (Ma et al., CVPR 2022, Oral) is the closest algorithmic precedent
to Pelican's greedy refinement loop. It recursively adds new optimizable
closed Bezier paths one at a time, optimizing all paths jointly after each
addition. Uses a component-wise initialization strategy. With only 5 paths,
LIVE reconstructs images that DiffVG needs 256 paths for. Both LIVE and
Pelican use the same core strategy: add one shape/path at a time, optimize
jointly, keep if loss improves. LIVE uses DiffVG as its renderer and Bezier
paths as primitives; Pelican uses soft SDFs with geometric primitives.

**Differentiable Sorting** (Blondel et al., ICML 2020) enables gradient-based
optimization of ranking and permutation operations. Relevant to layer ordering
in SVG compositing: instead of fixed z-order, differentiable sorting could
allow gradient descent to discover optimal compositing order.

**Current state in Pelican:** The project sidesteps most discrete relaxation
via greedy search + LLM architect, which is pragmatic but leaves room for
fully differentiable topology optimization. Pelican's existing
`sigmoid(-sdf / tau)` softness parameter is itself a continuous relaxation of
the hard inside/outside decision, so the principle is already embedded in the
design.

### 5. Test-Time Optimization and Planning

Pelican's entire pipeline is test-time optimization: given a target image, it
runs iterative gradient descent (500+ steps), greedy refinement (dozens of
rounds), and optional LLM refinement (multiple rounds).

**Scaling LLM Test-Time Compute** (Snell, Lee, Xu, Kumar, August 2024) studies
how to optimally allocate additional computation at inference time. Two
mechanisms are analyzed: searching against process-based verifier reward
models, and adaptively updating the model's response distribution. The key
finding is that optimal allocation depends on prompt difficulty, and that
smaller models with optimized test-time compute can outperform 14x larger
pretrained models.

**Relevance to Pelican:** Pelican's pipeline maps onto test-time optimization:
1. **Gradient descent** (500 steps of Adam) -- classical numerical test-time
   optimization. The differentiable loss function serves as a continuous
   process reward signal.
2. **Greedy refinement** (add shapes one at a time) -- discrete search at
   test time. Analogous to beam search over structural modifications.
3. **LLM refinement** (multi-round judge + architect) -- LLM-guided test-time
   search. Analogous to self-refinement with a language model critic.

Each phase uses more expensive computation to make diminishing-but-meaningful
improvements. Snell et al.'s insight -- that adaptive compute allocation
based on difficulty is optimal -- suggests Pelican should vary the number
of optimization steps and LLM rounds based on image complexity rather than
using fixed budgets.

The broader trend of inference-time compute encompasses strategies directly
used by Pelican:
- **Repeated sampling and filtering** (as in AlphaCode 2) -- Pelican's greedy
  loop proposes candidate shapes, evaluates via loss, keeps the best.
- **Iterative self-refinement** -- Pelican's LLM judge/architect loop
  generates, critiques, and revises.
- **Process reward models** -- Pelican's loss function scores every
  intermediate optimization step, not just the final output.

---

## Comparison: Pelican's Approach vs. Alternatives

### Rendering Strategy

| Approach | System | Primitives | Portability | Complexity |
|----------|--------|------------|-------------|------------|
| Soft SDF + sigmoid | **Pelican** | Circle, ellipse, triangle | Pure PyTorch (CPU/GPU/MPS) | ~100 lines |
| Analytical anti-aliasing | DiffVG | Full SVG paths, Bezier curves | C++/CUDA + Python bindings | ~10k lines |
| Neural SDF | DeepSDF, etc. | Learned shapes | GPU-only (network inference) | ~1k+ lines |

Pelican trades expressiveness (no Bezier curves) for simplicity and
portability. This is the right tradeoff for the project's scope (geometric
primitives for a pelican silhouette) but limits generalization to complex
vector art.

### Topology Search Strategy

| Approach | System | Mechanism | Fully Differentiable? |
|----------|--------|-----------|----------------------|
| Greedy forward selection | **Pelican**, LIVE | Add one shape, keep if loss drops | No (discrete add/drop) |
| LLM-guided structural edits | **Pelican** (Phase 2) | LLM proposes add/remove/modify | No (LLM is black box) |
| Gumbel-Softmax relaxation | (Not yet applied to SVG) | Relax discrete choices to soft | Yes |
| Neural architecture search | DARTS, etc. | Continuous relaxation of architecture | Yes |

The greedy + LLM approach is pragmatic and effective (100% acceptance rate in
experiments, 32% loss improvement), but not differentiable end-to-end. A
Gumbel-Softmax relaxation of shape type and presence decisions could enable
fully differentiable topology optimization.

### Optimization Paradigm

| Approach | Continuous Params | Discrete Structure | End-to-End? |
|----------|------------------|--------------------|-------------|
| **Pelican** | Gradient descent (Adam) | Greedy search + LLM | No |
| LIVE | Gradient descent (DiffVG) | Greedy layerwise addition | No |
| TextGrad formulation | Numerical gradients | Textual gradients | Partially (text backprop) |
| Trace/OPTO formulation | Execution trace optimization | LLM-proposed updates | Partially (trace-based) |
| Full Gumbel-Softmax | Gradient descent | Relaxed categorical | Yes |

---

## Recommendations

### What to Keep

1. **Soft SDF rendering.** Pure PyTorch, portable, simple, and sufficient for
   geometric primitives. No reason to adopt DiffVG's complexity unless Bezier
   curves become essential.

2. **Greedy forward selection.** 100% acceptance rate across experiments
   validates this approach. LIVE uses the same strategy successfully.

3. **Hybrid gradient + LLM architecture.** The compound AI system design is
   well-supported by the TextGrad / BAIR framing. Numerical gradients for
   continuous parameters, LLM feedback for discrete structure.

### What to Consider Adding

1. **OPRO-style trajectory prompting.** Include previous rounds' loss values
   in the LLM architect's context so it can learn from the optimization
   trajectory, not just the current state.

2. **DSPy-style prompt optimization.** Instead of hand-crafting judge and
   architect prompts, use DSPy's compiler to optimize them against
   loss-reduction metrics. This would automate a currently manual design
   process.

3. **Adaptive compute allocation.** Per Snell et al., vary optimization
   budget (steps, rounds) based on image complexity rather than fixed budgets.
   Simple heuristic: if early loss is high, allocate more steps; if greedy
   rounds stop improving, stop early.

4. **Error-guided shape placement.** Initialize new shapes at the location of
   highest per-pixel error instead of random position. This is a lightweight
   improvement that doesn't require new frameworks.

### Future Directions

1. **Gumbel-Softmax shape type selection.** Replace the cycling
   circle-ellipse-triangle pattern with a learned categorical distribution
   over shape types, relaxed via Gumbel-Softmax. This would enable gradient
   descent to discover which shape type is optimal at each round.

2. **Differentiable layer ordering.** Use differentiable sorting (Blondel et
   al., 2020) to optimize z-order via gradients rather than fixing it.

3. **Trace-based meta-optimization.** Express the full pipeline as a Trace
   computation graph and use OPTO to jointly optimize tau schedule, loss
   weights, judge prompts, architect prompts, and greedy search parameters.

4. **Bezier curve support.** Add differentiable Bezier paths (either via
   DiffVG integration or soft Bezier SDFs) for smoother contours. This would
   substantially increase the expressiveness of the primitive set.

---

## References

### Differentiable Solvers

- Blondel, M. et al. (2022). "Efficient and Modular Implicit
  Differentiation." NeurIPS 2022.
  [arXiv:2105.15183](https://arxiv.org/abs/2105.15183)

- Agrawal, A. et al. (2019). "Differentiable Convex Optimization Layers."
  NeurIPS 2019.
  [arXiv:1910.12430](https://arxiv.org/abs/1910.12430)

- Amos, B. et al. (2018). "Differentiable MPC for End-to-end Planning and
  Control." NeurIPS 2018.
  [arXiv:1810.13400](https://arxiv.org/abs/1810.13400)

- Wang, S. et al. (2025). "DiLQR: Differentiable Iterative Linear Quadratic
  Regulator via Implicit Differentiation." ICML 2025.
  [arXiv:2506.17473](https://arxiv.org/abs/2506.17473)

- Lu, H., Peng, Z., Yang, J. (2024). "MPAX: Mathematical Programming in
  JAX." [arXiv:2412.09734](https://arxiv.org/abs/2412.09734)

- Pineda, L. et al. (2022). "Theseus: A Library for Differentiable Nonlinear
  Optimization." NeurIPS 2022.
  [arXiv:2207.09442](https://arxiv.org/abs/2207.09442)

### Textual Gradients and LLM-as-Optimizer

- Pryzant, R. et al. (2023). "Automatic Prompt Optimization with 'Gradient
  Descent' and Beam Search." EMNLP 2023.
  [arXiv:2305.03495](https://arxiv.org/abs/2305.03495)

- Yuksekgonul, M. et al. (2025). "TextGrad: Automatic 'Differentiation' via
  Text." Nature, 639, 609-616.
  [arXiv:2406.07496](https://arxiv.org/abs/2406.07496)

- Khattab, O. et al. (2023). "DSPy: Compiling Declarative Language Model
  Calls into Self-Improving Pipelines." NeurIPS 2023.
  [arXiv:2310.03714](https://arxiv.org/abs/2310.03714)

- Yang, C. et al. (2023). "Large Language Models as Optimizers." Google
  DeepMind. [arXiv:2309.03409](https://arxiv.org/abs/2309.03409)

### Compound AI Systems

- Zaharia, M. et al. (2024). "The Shift from Models to Compound AI Systems."
  BAIR Blog.
  [bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)

- "Trace is the Next AutoDiff." NeurIPS 2024.
  [arXiv:2406.16218](https://arxiv.org/abs/2406.16218)

### Discrete Relaxation

- Jang, E., Gu, S., Poole, B. (2017). "Categorical Reparameterization with
  Gumbel-Softmax." ICLR 2017.
  [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)

- Maddison, C.J., Mnih, A., Teh, Y.W. (2017). "The Concrete Distribution."
  ICLR 2017.
  [arXiv:1611.00712](https://arxiv.org/abs/1611.00712)

- Bengio, Y., Leonard, N., Courville, A. (2013). "Estimating or Propagating
  Gradients Through Stochastic Neurons."
  [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)

- Blondel, M. et al. (2020). "Fast Differentiable Sorting and Ranking."
  ICML 2020.
  [arXiv:2002.08871](https://arxiv.org/abs/2002.08871)

### Differentiable Vector Graphics

- Li, T.-M. et al. (2020). "Differentiable Vector Graphics Rasterization
  for Editing and Learning." SIGGRAPH Asia 2020.
  [Project page](https://people.csail.mit.edu/tzumao/diffvg/)

- Ma, X. et al. (2022). "Towards Layer-wise Image Vectorization." CVPR 2022
  (Oral). [arXiv:2206.04655](https://arxiv.org/abs/2206.04655)

### Test-Time Optimization

- Snell, C. et al. (2024). "Scaling LLM Test-Time Compute Optimally Can Be
  More Effective than Scaling Model Parameters."
  [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)

### Image Quality

- Wang, Z. et al. (2004). "Image Quality Assessment: From Error Visibility
  to Structural Similarity." IEEE TIP, 13(4), 600-612.
  [DOI: 10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861)

### Compositing

- Porter, T. and Duff, T. (1984). "Compositing Digital Images." SIGGRAPH 84.
  [DOI: 10.1145/800031.808606](https://doi.org/10.1145/800031.808606)
