# Research: Python-in-WebAssembly for Browser-Local Differentiable Rendering

**Date:** 2026-02-15

**Author:** Claude (with Joshua Levy)

**Status:** In Progress

## Overview

This research investigates the feasibility of running Python-based differentiable rendering algorithms in the browser via WebAssembly (Wasm). The motivating use case is the [differentiable-pelican](https://github.com/jlevy/differentiable-pelican) project — a gradient-based SVG optimization system that uses PyTorch for differentiable rendering, automatic differentiation, and optimization. We explore whether this kind of workload could be made entirely browser-local.

## Questions to Answer

1. What is the current state of Python execution in WebAssembly (runtimes, maturity, performance)?
2. Can PyTorch or its core functionality run in the browser via Wasm?
3. What scientific Python packages work in Wasm today (NumPy, SciPy, etc.)?
4. How mature is package management for Python-in-Wasm, and could a `uv`-managed project be automatically mapped to a Wasm build?
5. What are the realistic options for getting differentiable rendering working in the browser?
6. What are the main technical barriers for a project like differentiable-pelican?

## Scope

**Included:**
- All major Python-in-Wasm runtimes (Pyodide, CPython-Wasm, MicroPython, RustPython, py2wasm)
- PyTorch and ML framework compatibility with Wasm
- Scientific Python stack in Wasm (NumPy, SciPy, Pillow, matplotlib)
- Python package management in Wasm (micropip, PEP 783, uv integration)
- Build processes (Emscripten, WASI)
- GitHub ecosystem analysis (trending and popular repositories)
- Alternative approaches (ONNX Runtime Web, Transformers.js, pure JS/TS reimplementation)

**Excluded:**
- Non-Python approaches to differentiable rendering (e.g., pure Rust-to-Wasm)
- Mobile-native deployment
- Server-side Wasm (except as context)

---

## Findings

### 1. The Differentiable-Pelican Project: What Needs to Run

The project uses **PyTorch exclusively** as its differentiable computing framework. Key dependencies and capabilities required:

| Dependency | Version | Role | Wasm Status |
|---|---|---|---|
| `torch` | >=2.1.0 | Autograd, nn.Module, Adam optimizer, tensor ops | **Not available** |
| `torchvision` | >=0.16.0 | Image utilities | **Not available** |
| `numpy` | >=1.24.0 | Array conversions | **Available** (Pyodide) |
| `pillow` | >=10.0.0 | Image I/O | **Available** (Pyodide) |
| `matplotlib` | >=3.7.0 | Plotting | **Available** (Pyodide) |
| `rich` | >=13.0.0 | Terminal UI | N/A (browser) |
| `anthropic` | >=0.18.0 | LLM API client | Network-dependent |
| `pydantic` | >=2.0.0 | Data validation | **Available** (Pyodide) |
| `imageio` | >=2.31.0 | GIF generation | Likely available (pure Python) |

**Critical PyTorch features used:**
- `torch.nn.Module` with `nn.Parameter` for all shape primitives
- `torch.autograd` — automatic differentiation through the entire render pipeline
- `torch.optim.Adam` with `CosineAnnealingLR` scheduling
- `torch.nn.functional.conv2d` for Sobel edge detection and SSIM
- `torch.sigmoid`, `torch.nn.functional.softplus` for reparameterization
- `torch.nn.utils.clip_grad_norm_` for gradient clipping
- Device management (CPU/CUDA/MPS)

The core algorithm is:
1. **SDF computation** — signed distance fields for circles, ellipses, triangles
2. **Soft coverage** — `sigmoid(-sdf / tau)` for differentiable boundaries
3. **Alpha compositing** — Porter-Duff back-to-front composition
4. **Composite loss** — MSE + SSIM + edge + perimeter + degeneracy + canvas penalties
5. **Adam optimization** — ~500 gradient descent steps
6. **Greedy topology search** — random shape proposals with accept/reject

### 2. Python-in-Wasm Runtimes

#### Pyodide (the dominant runtime)

**Pyodide 0.29.3** (released 2026-01-28) is a full port of CPython 3.13 to WebAssembly via Emscripten. It is the clear ecosystem leader at **14,217 GitHub stars** with very active development (3 releases in January 2026 alone).

Key capabilities:
- Full CPython 3.13 interpreter
- 200+ pre-built packages including NumPy, SciPy, pandas, matplotlib, scikit-learn, Pillow
- Runs in all modern browsers and Node.js
- Over 1 billion requests on jsDelivr in 2025 (doubling year-over-year)

Recent milestones:
- **Pyodide 0.27** (Jan 2025): Decoupled `pyodide-build`, FFI performance via wasm-gc
- **Pyodide 0.28** (Jul 2025): Stabilized ABI (`pyodide_2025_0`), Python 3.13, WebAssembly exception handling, **JSPI enabled by default** (Stage 4), new Matplotlib backend
- **Pyodide 0.29** (Oct 2025): cibuildwheel integration as official platform maintainer

Performance: ~3-5x slower than native CPython for compute-heavy tasks. NumPy operations are closer to native due to compiled C extensions.

**Limitation: No PyTorch.** [Pyodide issue #1625](https://github.com/pyodide/pyodide/issues/1625) remains open with no near-term solution.

#### CPython Wasm (upstream)

- **Emscripten target** (`wasm32-emscripten`): Restored to Tier 3 for Python 3.14 via [PEP 776](https://peps.python.org/pep-0776/)
- **WASI target** (`wasm32-wasi`): Promoted to Tier 2 for Python 3.13 via [PEP 816](https://peps.python.org/pep-0816/)
- All extensions must be statically linked; no dynamic linking
- WASI missing: threads, sockets, dynamic linking, subprocesses

#### Other runtimes

| Runtime | Approach | Size | Maturity | Notes |
|---|---|---|---|---|
| **MicroPython Wasm** | Lightweight interpreter | ~303 KB | Production-usable | No scientific stack |
| **RustPython** | Rust-based interpreter | Medium | Experimental | Limited stdlib, no packages |
| **py2wasm** (Wasmer) | AOT via Nuitka→C→Wasm | Varies | Available | ~3x faster than interpreter, Python 3.11 only |
| **PyScript** | User-facing framework on Pyodide/MicroPython | N/A | Production | `<script type="py">` tags |

### 3. PyTorch in Wasm: The Core Blocker

**PyTorch cannot run in WebAssembly as of early 2026.** The reasons:

1. **No Wasm wheels exist.** PyTorch wheels contain native x86_64/aarch64 compiled extensions. Nobody has compiled them for Emscripten.
2. **Enormous size.** PyTorch CPU-only is ~200 MB. Even with tree-shaking, the Wasm binary would be impractical for browser download.
3. **Complex native dependencies.** PyTorch depends on BLAS/LAPACK, OpenMP, and other system libraries that don't have Wasm ports or are extremely difficult to compile.
4. **`cffi` requirement.** PyTorch requires `cffi` at runtime, which was not supported in Pyodide.
5. **Threading model.** PyTorch's internal parallelism relies on pthreads/OpenMP, which map poorly to Wasm's threading model (SharedArrayBuffer + Web Workers).

**GitHub ecosystem confirms this:** Searching "pytorch wasm" and "pytorch webassembly" across all GitHub repositories returns **zero results** with meaningful star counts. Nobody has successfully built or maintained a PyTorch-on-Wasm project.

### 4. Package Management for Python in Wasm

#### micropip (current state)

[micropip 0.11.0](https://micropip.pyodide.org/) is Pyodide's package manager. It installs:
- Pure Python wheels from PyPI (fetched automatically)
- Pre-built Pyodide packages (from Pyodide CDN)
- Wheels from arbitrary URLs
- Regular `pip` does **not** work (requires subprocess support)

#### PEP 783: Wasm Wheels on PyPI (pending)

[PEP 783](https://peps.python.org/pep-0783/) proposes standardized Emscripten/Pyodide wheels on PyPI with platform tags like `pyodide_2025_0_wasm32`. This would let package maintainers publish Wasm wheels to PyPI directly. Building uses `pyodide build` or cibuildwheel. Still pending approval as of early 2026.

#### PEP 818: Upstream JS FFI in CPython (proposed Jan 2026)

[PEP 818](https://peps.python.org/pep-0818/) proposes upstreaming Pyodide's Python/JavaScript FFI into CPython itself, making JS interop a first-class CPython feature on Emscripten.

#### uv Integration

There is **no direct `uv` → Pyodide/micropip bridge** for in-browser use. The closest integration:

- **Cloudflare Python Workers** (late 2025): Uses `uv` via `pywrangler` CLI to resolve dependencies from `pyproject.toml`, then deploys to Pyodide/Wasm on Cloudflare's edge. Heavy packages like FastAPI load in ~1 second via memory snapshots.
- For a project like differentiable-pelican that uses `uv` with `pyproject.toml`: the pure-Python dependencies could be installed via micropip, but native-extension packages need pre-built Pyodide versions.

**Mapping uv → Wasm is not automatic.** You would need to:
1. Parse `pyproject.toml` dependencies
2. Check each against Pyodide's package list
3. Fall back to micropip for pure-Python packages
4. Flag unsupported native-extension packages (like PyTorch)

Nobody has built a general-purpose `uv-to-wasm` tool. This would be a novel contribution.

### 5. GitHub Ecosystem: Popular Python+Wasm Projects

#### Major ecosystem projects (1,000+ stars)

| Repository | Stars | Description |
|---|---|---|
| [pyodide/pyodide](https://github.com/pyodide/pyodide) | 14,217 | CPython for browser/Node.js via Wasm |
| [jupyterlite/jupyterlite](https://github.com/jupyterlite/jupyterlite) | 4,753 | Jupyter notebooks entirely in the browser |
| [StructuredLabs/preswald](https://github.com/StructuredLabs/preswald) | 4,296 | Wasm packager for Python data apps (Pyodide + DuckDB + Pandas + Plotly) |
| [whitphx/stlite](https://github.com/whitphx/stlite) | 1,597 | Streamlit entirely in the browser via Wasm |

#### ML-in-browser projects (ONNX Runtime Web ecosystem)

| Repository | Stars | Description |
|---|---|---|
| [microsoft/onnxruntime-web-demo](https://github.com/microsoft/onnxruntime-web-demo) | 218 | Official ONNX Runtime Web demos |
| [Hyuto/yolov8-onnxruntime-web](https://github.com/Hyuto/yolov8-onnxruntime-web) | 192 | YOLOv8 object detection in browser |
| [neuroneural/brainchop](https://github.com/neuroneural/brainchop) | 520 | In-browser 3D MRI segmentation |

#### Notable Python-in-browser projects

| Repository | Stars | Description |
|---|---|---|
| [simonw/datasette-lite](https://github.com/simonw/datasette-lite) | 396 | Datasette via Pyodide |
| [kkinder/puepy](https://github.com/kkinder/puepy) | 321 | Python frontend framework via PyScript |
| [elilambnz/react-py](https://github.com/elilambnz/react-py) | 295 | Run Python in React apps (via Pyodide) |
| [langchain-ai/langchain-sandbox](https://github.com/langchain-ai/langchain-sandbox) | 234 | Sandboxed Python execution via Pyodide + Deno |

**Key takeaway:** No project in the entire GitHub ecosystem has successfully run PyTorch in Wasm. The ML-in-browser space is dominated by ONNX Runtime Web with WebGPU acceleration.

### 6. Alternative Approaches for ML in Browser

#### ONNX Runtime Web

The most mature option for ML inference in browser. Supports:
- **WebAssembly backend**: CPU-based, works everywhere
- **WebGPU backend**: GPU-accelerated, up to 100x faster than Wasm CPU, Chrome 113+
- **WebNN**: Emerging standard for native NN acceleration

Workflow: Train in PyTorch → export to ONNX → run in browser with ONNX Runtime Web.

**Limitation for differentiable-pelican:** ONNX Runtime Web is an **inference** engine. It does not support automatic differentiation or gradient-based optimization. You can run a forward pass but not backpropagate.

#### Transformers.js v3

Hugging Face's library for running ONNX models in browser with WebGPU. Supports NLP, vision, audio, multimodal tasks. Models up to ~3B parameters (quantized 4-bit) run on consumer hardware.

**Limitation:** Same as ONNX — inference only, no autograd.

#### Custom Autograd in JavaScript/TypeScript

Libraries that provide differentiable tensor operations in JS:
- **[TensorFlow.js](https://www.tensorflow.org/js)**: Full autograd, runs on WebGL/WebGPU/Wasm. Could theoretically implement the differentiable rendering pipeline.
- **[�OML/autograd.js](https://github.com/nicktomlin/autograd)**: Lightweight autograd libraries exist but are not production-grade.
- **Custom implementation**: The SDF/rendering math is not complex — it could be reimplemented in TypeScript with a minimal autograd system.

---

## Options Considered

### Option A: Port Differentiable-Pelican to Pyodide (Replace PyTorch with NumPy + Custom Autograd)

**Description:** Rewrite the differentiable rendering pipeline to use NumPy (available in Pyodide) with a custom autograd implementation. NumPy provides the tensor operations; a lightweight reverse-mode AD library handles gradient computation.

**What changes:**
- Replace `torch.nn.Module` → plain Python classes with NumPy arrays
- Replace `torch.autograd` → custom reverse-mode AD (e.g., [autograd](https://github.com/HIPS/autograd) or [JAX-like tracing](https://jax.readthedocs.io/en/latest/autodidax.html))
- Replace `torch.optim.Adam` → custom Adam implementation (~30 lines)
- Replace `torch.sigmoid`, `softplus` → NumPy equivalents
- Replace `torch.nn.functional.conv2d` → SciPy `signal.convolve2d` or manual implementation
- Keep `Pillow`, `matplotlib` as-is (both available in Pyodide)

**Autograd options for NumPy:**
- [HIPS/autograd](https://github.com/HIPS/autograd): Pure Python, auto-diffs NumPy code. ~50 KB. Likely works in Pyodide.
- [Google/JAX](https://github.com/google/jax): Full autograd + JIT. **Not available in Pyodide** (requires XLA).
- Custom minimal AD: The computation graph for this project is relatively simple (SDFs → coverage → compositing → loss). A purpose-built tape-based AD would be ~200-300 lines of Python.

**Pros:**
- Stays in Python — minimal algorithmic rewrite
- Pyodide + NumPy + SciPy is mature and well-tested
- HIPS/autograd is a pure-Python library, likely installable via micropip
- User sees familiar Python code
- Could share code between server and browser versions

**Cons:**
- Performance: NumPy in Pyodide is 3-5x slower than native. Without PyTorch's optimized kernels, optimization loops would be significantly slower.
- Custom autograd adds maintenance burden
- No GPU acceleration (Pyodide runs on CPU only)
- 500-step optimization at 256×256 resolution could take 30-60+ seconds in browser
- HIPS/autograd may have edge cases in Pyodide; needs testing

**Feasibility: Medium.** The algorithmic rewrite is tractable — the core math (SDFs, sigmoid, compositing) maps cleanly to NumPy. Performance is the main concern.

### Option B: Reimplement in TypeScript with TensorFlow.js

**Description:** Rewrite the differentiable rendering pipeline in TypeScript using TensorFlow.js for autograd and tensor operations. TensorFlow.js provides GPU-accelerated computation via WebGL/WebGPU in the browser.

**What changes:**
- Full TypeScript rewrite of geometry, SDF, renderer, loss, optimizer modules
- Use `tf.tensor()`, `tf.grad()`, `tf.train.adam()` instead of PyTorch equivalents
- Use `tf.sigmoid()`, `tf.conv2d()`, etc.
- Canvas API or OffscreenCanvas for image display
- WebGPU backend for GPU acceleration

**Pros:**
- **GPU acceleration in browser** via WebGL/WebGPU — potentially faster than CPU PyTorch for this workload
- TensorFlow.js has full autograd (`tf.grad`, `tf.gradients`)
- Mature ecosystem, well-documented
- Small bundle size (~200 KB gzipped for core)
- Native browser integration (Canvas, DOM, etc.)
- No Python/Wasm overhead

**Cons:**
- Complete rewrite from Python to TypeScript
- TensorFlow.js API differs from PyTorch — not a mechanical translation
- Loses the "Python everywhere" value proposition
- TensorFlow.js has some missing ops compared to PyTorch (may need workarounds for specific SDF computations)
- Two codebases to maintain if you want both CLI and browser versions

**Feasibility: High.** TensorFlow.js is well-suited for this exact workload — small tensor operations with autograd and GPU acceleration. The algorithm is the differentiable rendering pipeline, which is mathematically straightforward to port.

### Option C: Hybrid — Python Orchestration in Pyodide, Computation in JS/WebGPU

**Description:** Run the high-level Python logic (shape management, greedy search, LLM integration) in Pyodide, but delegate the performance-critical differentiable rendering and optimization to a JavaScript/WebGPU library called from Python.

**What changes:**
- Keep shape definitions, greedy search logic, LLM integration in Python (Pyodide)
- Write a JS/TS differentiable renderer using WebGPU compute shaders or TensorFlow.js
- Use Pyodide's JavaScript FFI to call JS functions from Python
- Python manages the optimization loop; JS does the heavy tensor math

**Pros:**
- Keeps the algorithmic logic in Python
- Gets GPU acceleration for the hot path
- Pyodide's JS FFI is mature (JSPI enabled by default since 0.27.7)
- Best-of-both-worlds: Python readability + JS performance
- LLM integration (Anthropic API) can use `fetch()` from browser

**Cons:**
- Complex architecture spanning two languages
- Serialization overhead at Python/JS boundary (though Pyodide handles typed arrays efficiently)
- Harder to debug and test
- Requires writing a custom JS differentiable renderer anyway

**Feasibility: Medium-High.** Architecturally sound but complex. Worth considering if you want to keep the Python "story" while getting real GPU performance.

### Option D: Export Optimized Model to ONNX, Run Inference-Only in Browser

**Description:** Run the full optimization pipeline server-side in PyTorch, export the final optimized shapes as a static result (SVG, JSON, or ONNX model), and use the browser only for visualization and interactive exploration.

**What changes:**
- Server-side: Run optimization as-is (no changes)
- Browser: Load pre-computed results, display SVG, allow interactive parameter tweaking
- Optional: Export a small "refinement" ONNX model that takes shape parameters → rendered image, enabling limited browser-side gradient-free optimization (e.g., evolutionary strategies)

**Pros:**
- No Wasm complexity — PyTorch runs where it works best
- Browser is just a viewer
- Minimal development effort
- Could add interactive sliders for shape parameters without autograd

**Cons:**
- Not truly "browser-local computation" — requires server
- No live gradient-based optimization in browser
- Loses the educational/demo value of seeing optimization run live

**Feasibility: Very High.** This is the path of least resistance but doesn't achieve the goal of browser-local execution.

### Option E: WebGPU Compute Shaders (Pure JS/TS, No Framework)

**Description:** Implement the differentiable renderer directly using WebGPU compute shaders in WGSL. Write the SDF evaluation, coverage computation, and alpha compositing as GPU compute kernels. Implement a minimal autograd system in TypeScript that records operations and generates backward-pass compute shaders.

**What changes:**
- All rendering happens on GPU via compute shaders
- TypeScript manages the computation graph and parameter updates
- Custom Adam optimizer in TypeScript
- No Python, no TensorFlow.js dependency

**Pros:**
- Maximum performance — GPU-parallel SDF evaluation across all pixels
- Minimal dependencies (just WebGPU API)
- Could handle higher resolutions (512×512, 1024×1024) that would be impractical on CPU
- Educational value: demonstrates differentiable rendering from scratch

**Cons:**
- Significant engineering effort
- WebGPU not yet available in all browsers (Safari support is partial)
- WGSL shader debugging is primitive
- Custom autograd for GPU is a research-level problem
- No Python at all — complete departure from the original project

**Feasibility: Medium.** High performance ceiling but high development cost. Most relevant if the goal is a polished, high-performance browser demo.

---

## Feasibility Analysis for Differentiable-Pelican

### Dependency-by-Dependency Wasm Readiness

| Component | Required | Available in Wasm? | Alternative |
|---|---|---|---|
| Python 3.11+ | Yes | **Yes** (Pyodide 0.29.3 = CPython 3.13) | — |
| NumPy | Yes | **Yes** (Pyodide) | — |
| Pillow | Yes | **Yes** (Pyodide) | Canvas API |
| matplotlib | Yes (optional) | **Yes** (Pyodide, new backend) | Chart.js, D3 |
| SciPy | Useful | **Yes** (Pyodide) | — |
| Pydantic | Yes | **Yes** (Pyodide) | — |
| PyTorch autograd | **Critical** | **No** | HIPS/autograd, TF.js, custom |
| PyTorch nn.Module | **Critical** | **No** | Custom classes |
| PyTorch optimizers | **Critical** | **No** | Custom Adam (~30 LOC) |
| torch.conv2d | Used | **No** | scipy.signal.convolve2d |
| torch.sigmoid | Used | **No** | NumPy: `1/(1+np.exp(-x))` |
| Anthropic SDK | Optional | Partial (needs HTTP) | `fetch()` from JS |
| rich | CLI only | N/A | Browser UI |
| imageio | GIF export | Likely (pure Python) | Canvas recording |

### The Core Technical Barrier

**The single biggest barrier is replacing PyTorch's autograd engine.** Everything else either works in Pyodide already or has a straightforward alternative. The question is how to get automatic differentiation:

1. **HIPS/autograd** (pure Python): Most promising for a Pyodide-based approach. It wraps NumPy and provides `grad()` and `jacobian()`. The differentiable-pelican computation graph (SDFs → sigmoid → compositing → loss) should be within its capabilities. Needs testing in Pyodide.

2. **TensorFlow.js** (JavaScript): Provides `tf.grad()` with GPU acceleration. Requires a TypeScript rewrite but offers the best performance.

3. **Custom tape-based AD**: For this specific computation graph, a custom autograd is feasible. The operations are: basic arithmetic, sigmoid, softplus, conv2d, MSE, and Gaussian-windowed statistics (SSIM). A tape-based system recording these ops and computing vjps (vector-Jacobian products) would be ~300-500 lines of Python.

### Performance Estimates

For a 256×256 image with 35 shapes, one forward pass involves:
- 35 SDF evaluations × 65,536 pixels = ~2.3M operations
- Coverage computation (sigmoid): ~2.3M operations
- Alpha compositing: ~2.3M operations per shape = ~80M operations
- Loss computation: ~65K operations (MSE) + conv2d (Sobel, SSIM)

| Approach | Forward Pass | 500 Steps | Notes |
|---|---|---|---|
| PyTorch CPU (native) | ~5 ms | ~5 sec | Current baseline |
| NumPy in Pyodide | ~25-50 ms | ~25-50 sec | 5-10x slower |
| TensorFlow.js WebGPU | ~2-5 ms | ~2-5 sec | GPU parallel |
| WebGPU compute shaders | ~1-3 ms | ~1-3 sec | Optimal GPU use |

The greedy topology search (which runs multiple optimization rounds) would multiply these times by the number of shape proposals (typically 20-50).

### uv → Wasm Build Mapping

A hypothetical `uv-to-wasm` tool for this project would need to:

1. Parse `pyproject.toml` dependencies
2. For each dependency, check: (a) Is it in Pyodide's pre-built package list? → use Pyodide's version. (b) Is it a pure Python wheel? → install via micropip. (c) Does it have C extensions without a Pyodide build? → **flag as unsupported**
3. Generate a `micropip.install()` manifest or a Pyodide `loadPackage()` list
4. For PyTorch specifically: flag it as unsupported and suggest alternatives

For differentiable-pelican:
- ✅ Auto-mappable: `numpy`, `pillow`, `matplotlib`, `pydantic`, `python-dotenv`
- ⚠️ Needs testing: `imageio` (may work as pure Python)
- ❌ Not mappable: `torch`, `torchvision`
- 🔄 Needs replacement: `rich` → browser UI, `anthropic` → `fetch()` wrapper

**Nobody has built a general uv-to-Pyodide mapping tool.** Cloudflare's `pywrangler` is the closest, but it's proprietary and specific to Cloudflare Workers.

---

## Recommendations

### Recommended Path: Option B (TypeScript + TensorFlow.js) or Option C (Hybrid)

For a real browser-local differentiable rendering demo, **Option B** (TypeScript rewrite with TensorFlow.js) offers the best combination of feasibility, performance, and user experience. The algorithm is mathematically compact enough that a TypeScript port is tractable, and TensorFlow.js provides both autograd and GPU acceleration.

If preserving the Python codebase is important, **Option C** (hybrid Pyodide + JS) is viable but more complex. The Python layer handles shape management and the optimization loop, while a JS library handles the differentiable rendering math with GPU acceleration.

### If You Want to Stay Pure Python

**Option A** (Pyodide + NumPy + HIPS/autograd) is worth prototyping. The first step would be to verify that HIPS/autograd works in Pyodide — if it does, the port is relatively mechanical. Performance will be 5-10x slower than native PyTorch, meaning 25-50 seconds for 500 optimization steps at 256×256. This is acceptable for an interactive demo with a progress bar, but would make the greedy topology search (which runs many optimization rounds) slow.

### Key Insight

The differentiable-pelican computation is relatively simple compared to typical deep learning workloads. It's not a neural network — it's a handful of analytic SDFs composed through differentiable operations. This makes it an unusually good candidate for non-PyTorch autograd systems. The total parameter count is tiny (3-8 parameters per shape × 35 shapes ≈ 200 parameters). The computation graph is static and shallow. This is well within reach of lightweight autograd implementations.

### What Would Move the Needle

Several developments would significantly change the feasibility picture:

1. **PEP 783 approval + PyTorch Wasm wheels**: If PyTorch ever publishes Wasm wheels, everything changes. But this is unlikely in the near term due to the size and complexity of PyTorch's native code.
2. **Pyodide SIMD optimization**: The Pyodide team is exploring WebAssembly SIMD for NumPy/SciPy. This could cut the performance gap from 5-10x to 2-3x.
3. **WebGPU compute from Python**: If Pyodide gained bindings to WebGPU compute shaders, you could write GPU kernels callable from Python. This doesn't exist yet.
4. **A "PyTorch Lite for Wasm" project**: A minimal autograd + tensor library compiled to Wasm with just the operations needed for small-scale optimization. Nobody has built this yet.

---

## Next Steps

- [ ] Prototype: Test HIPS/autograd in Pyodide (does it install via micropip? does `grad()` work?)
- [ ] Prototype: Implement one SDF (circle) + sigmoid coverage + MSE loss in NumPy with HIPS/autograd, benchmark in Pyodide
- [ ] Evaluate: Try porting the `renderer.py` forward pass to TensorFlow.js, measure performance with WebGPU
- [ ] Explore: Could a minimal Python autograd library (purpose-built for this use case) be small enough to be fast in Pyodide?
- [ ] Research: Look into [AnyWidget](https://anywidget.dev/) as a bridge between Python (Pyodide/JupyterLite) and custom JS rendering
- [ ] Design: Sketch the hybrid architecture (Option C) — what crosses the Python/JS boundary?
- [ ] Community: Open a discussion on the Pyodide repo about lightweight autograd for scientific computing in browser

## References

### Python-in-Wasm Runtimes
- [Pyodide](https://pyodide.org/) — CPython for browser/Node.js via Wasm (14,217 stars)
- [Pyodide 0.28 Release Blog](https://blog.pyodide.org/posts/0.28-release/) — ABI stabilization, JSPI, Wasm exceptions
- [Pyodide 0.29 Release Blog](https://blog.pyodide.org/posts/0.29-release/) — cibuildwheel integration
- [PyScript](https://pyscript.net/) — User-facing framework on Pyodide/MicroPython
- [RustPython](https://github.com/RustPython/RustPython) — Python interpreter in Rust, compiles to Wasm
- [py2wasm](https://wasmer.io/posts/py2wasm-a-python-to-wasm-compiler) — AOT Python-to-Wasm via Nuitka

### Standards and PEPs
- [PEP 776](https://peps.python.org/pep-0776/) — Emscripten as Tier 3 CPython platform (Python 3.14)
- [PEP 783](https://peps.python.org/pep-0783/) — Emscripten/Pyodide wheels on PyPI (pending)
- [PEP 816](https://peps.python.org/pep-0816/) — WASI as Tier 2 CPython platform (Python 3.13)
- [PEP 818](https://peps.python.org/pep-0818/) — Upstream Python/JS FFI into CPython (proposed Jan 2026)

### ML in Browser
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) — ML inference with WebGPU/Wasm backends
- [Transformers.js v3](https://huggingface.co/docs/transformers.js/en/index) — Hugging Face models in browser
- [TensorFlow.js](https://www.tensorflow.org/js) — Full ML framework with autograd and WebGPU

### Package Management
- [micropip](https://micropip.pyodide.org/) — Pyodide's package manager
- [Cloudflare Python Workers](https://blog.cloudflare.com/python-workers-advancements/) — uv + Pyodide on edge
- [Pyodide Package List](https://pyodide.org/en/stable/usage/packages-in-pyodide.html) — What's available

### GitHub Ecosystem
- [jupyterlite](https://github.com/jupyterlite/jupyterlite) — Jupyter in browser (4,753 stars)
- [preswald](https://github.com/StructuredLabs/preswald) — Wasm packager for Python data apps (4,296 stars)
- [stlite](https://github.com/whitphx/stlite) — Streamlit in browser (1,597 stars)
- [brainchop](https://github.com/neuroneural/brainchop) — In-browser MRI segmentation (520 stars)
- [langchain-sandbox](https://github.com/langchain-ai/langchain-sandbox) — Sandboxed Python via Pyodide (234 stars)

### Autograd Libraries
- [HIPS/autograd](https://github.com/HIPS/autograd) — Automatic differentiation for NumPy
- [Pyodide Issue #1625](https://github.com/pyodide/pyodide/issues/1625) — PyTorch support request (open, unresolved)
