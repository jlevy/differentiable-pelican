# differentiable-pelican

A hybrid system that combines **gradient-based optimization** with
**LLM-guided structural refinement** to generate SVG drawings of pelicans
from a target image.

The core idea: use differentiable rendering (soft signed distance fields)
to optimize shape parameters via gradient descent, then use an LLM to
propose structural edits (add/remove/modify shapes) that escape local
minima. Repeat in a refinement loop.

## How It Works

1. **Differentiable Rendering**: Shapes (circles, ellipses, triangles) are
   rendered using soft SDFs with sigmoid-based coverage. Each shape has
   optimizable position, size, rotation, and grayscale intensity parameters.

2. **Gradient Descent**: Adam optimizer minimizes a multi-component loss
   (MSE + SSIM + edge + priors) to match a target image.

3. **LLM Refinement Loop**: A judge (Claude) evaluates the result, an
   architect proposes structural edits, and the cycle repeats with
   automatic rollback on quality degradation.

## Quick Start

```bash
# Install
uv sync

# Render initial pelican (no optimization)
pelican test-render --resolution 128

# Optimize against target image
pelican optimize --target images/pelican-drawing-1.jpg --steps 500

# Full refinement loop with LLM feedback (requires ANTHROPIC_API_KEY)
pelican refine --target images/pelican-drawing-1.jpg --rounds 5
```

## Architecture

```
Target Image  -->  Differentiable Renderer  -->  Loss Function
     ^                    |                          |
     |              Soft SDF + α-over         MSE + SSIM + Edge
     |                    |                          |
     |              Shape Parameters  <---  Gradient Descent (Adam)
     |                    |
     |              Judge (Claude)  -->  Architect (Claude)
     |                    |                    |
     └────────────  Refinement Loop  <-- Edit Parser
```

See [pelican-plan.md](docs/design/pelican-plan.md) for the full design document.

## CLI Commands

| Command | Description |
|---------|-------------|
| `pelican test-render` | Render initial geometry without optimization |
| `pelican optimize` | Optimize shapes to match target image |
| `pelican judge` | Evaluate optimized SVG with LLM |
| `pelican refine` | Full refinement loop with LLM feedback |
| `pelican validate-image` | Validate a rendered image with LLM |

## Project Docs

For how to install uv and Python, see [installation.md](docs/installation.md).

For development workflows, see [development.md](docs/development.md).

For instructions on publishing to PyPI, see [publishing.md](docs/publishing.md).

---

*This project was built from
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv).*
