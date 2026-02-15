# Research: Browser-Local Differentiable Rendering via WebAssembly

**Date:** 2026-02-15

**Author:** Claude (with Joshua Levy)

**Status:** In Progress

## Overview

This research investigates the feasibility of running differentiable rendering algorithms in the browser via WebAssembly (Wasm). The motivating use case is the [differentiable-pelican](https://github.com/jlevy/differentiable-pelican) project — a gradient-based SVG optimization system that uses PyTorch for differentiable rendering, automatic differentiation, and optimization. We explore whether this kind of workload could be made entirely browser-local, examining not just Python-in-Wasm but also the Rust ML ecosystem as a compelling alternative path — Rust compiles natively to Wasm and has a rapidly maturing set of deep learning frameworks with autograd support.

## Questions to Answer

1. What is the current state of Python execution in WebAssembly (runtimes, maturity, performance)?
2. Can PyTorch or its core functionality run in the browser via Wasm?
3. What scientific Python packages work in Wasm today (NumPy, SciPy, etc.)?
4. How mature is package management for Python-in-Wasm, and could a `uv`-managed project be automatically mapped to a Wasm build?
5. What is the state of the Rust ML/deep learning ecosystem? Can Rust frameworks with autograd be compiled to Wasm as an alternative to porting Python?
6. What are the realistic options for getting differentiable rendering working in the browser?
7. What are the main technical barriers for a project like differentiable-pelican?

## Scope

**Included:**
- All major Python-in-Wasm runtimes (Pyodide, CPython-Wasm, MicroPython, RustPython, py2wasm)
- PyTorch and ML framework compatibility with Wasm
- Scientific Python stack in Wasm (NumPy, SciPy, Pillow, matplotlib)
- Python package management in Wasm (micropip, PEP 783, uv integration)
- Build processes (Emscripten, WASI)
- **Rust ML/deep learning ecosystem** — frameworks, autograd libraries, and their Wasm compilation story
- GitHub ecosystem analysis (trending and popular repositories)
- Alternative approaches (ONNX Runtime Web, Transformers.js, Rust-to-Wasm, pure JS/TS reimplementation)

**Excluded:**
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

### 7. The Rust ML Ecosystem: A Compelling Alternative Path

The inability to run PyTorch in Wasm highlights a fundamental tension: Python's ML ecosystem was never designed for portability. Rust, by contrast, compiles natively to `wasm32-unknown-unknown` and has a rapidly maturing deep learning ecosystem. Rather than trying to force Python into Wasm, the question becomes: **can we rewrite the differentiable computation in Rust and compile that to Wasm?**

#### Burn: The Most Complete Rust ML Framework (14,358 stars)

[Burn](https://github.com/tracel-ai/burn) (v0.20.1, released 2026-01-23) is a next-generation deep learning framework built in Rust by [Tracel Inc.](https://burn.dev) with 242 contributors. It is the strongest candidate for replacing PyTorch in a Wasm context because it has **both autograd and first-class Wasm support**.

**Architecture:** Burn's key design is a generic `Backend` trait. All model code is backend-agnostic — you write it once and it runs on any backend:

| Backend | Type | Wasm? | Notes |
|---|---|---|---|
| **NdArray** | CPU (pure Rust) | **Yes** | No native dependencies, `#![no_std]` compatible, `wasm32-unknown-unknown` |
| **WGPU** | GPU (WebGPU) | **Yes** | GPU acceleration in browsers with WebGPU support |
| **CUDA** | GPU (NVIDIA) | No | Native NVIDIA GPU |
| **ROCm** | GPU (AMD) | No | Native AMD GPU |
| **Metal** | GPU (Apple) | No | Native Apple GPU |
| **Vulkan** | GPU | No | Native Vulkan |
| **LibTorch** | CPU/GPU | No | tch-rs bindings to PyTorch C++ |
| **Candle** | CPU/GPU | Partial | HuggingFace's engine as a Burn backend |

**Autodiff:** Implemented as a backend *decorator* (`Autodiff<B>`). You wrap any base backend — e.g., `Autodiff<NdArray>` for CPU or `Autodiff<Wgpu>` for GPU — and it transparently adds reverse-mode automatic differentiation. Calling `.backward()` computes gradients through the full computation graph. This is a production-quality autograd engine.

**Wasm deployment is proven.** The repository includes working examples:
- `burn/examples/mnist-inference-web` — MNIST digit classification running entirely in the browser
- Live demo at [burn.dev/mnist_inference_web.html](https://burn.dev/mnist_inference_web.html)
- A compiled Burn + WGPU model ships as a **~2 MB `.wasm` file** (of which ~1.5 MB is model weights — the framework overhead is only ~357 KB)

**What Burn provides that maps to PyTorch concepts:**

| PyTorch | Burn Equivalent |
|---|---|
| `torch.Tensor` | `Tensor<B, D>` (generic over backend and dimensionality) |
| `torch.nn.Module` | `#[derive(Module)]` trait |
| `nn.Parameter` | Module fields are automatically tracked |
| `torch.autograd` | `Autodiff<B>` backend decorator |
| `torch.optim.Adam` | `burn::optim::Adam` |
| `torch.sigmoid` | `tensor.sigmoid()` (activation module) |
| `F.conv2d` | `burn::nn::conv::Conv2d` |
| `torch.nn.functional.mse_loss` | `burn::nn::loss::MseLoss` |
| Gradient clipping | `GradientsParams` with clipping config |
| Learning rate schedulers | Built-in LR schedulers |
| ONNX model import | `burn-onnx` crate |

**Current limitation:** Wasm demos to date are **inference only**. While Burn's `Autodiff<NdArray>` and `Autodiff<Wgpu>` backends technically support training in Wasm, nobody has published a browser-based training demo. Browser memory limits and WebGPU compute shader limitations are the practical constraints, not framework limitations. For the differentiable-pelican use case (tiny parameter count, simple computation graph), training in Wasm is likely feasible but unproven.

**CubeCL:** Burn includes [CubeCL](https://github.com/tracel-ai/cubecl), a Rust-native GPU programming language that compiles to CUDA, Metal, Vulkan, and WebGPU. This means custom GPU kernels (e.g., for SDF evaluation) could be written in Rust and run on WebGPU in the browser.

#### Candle: HuggingFace's Inference-Focused Framework (19,389 stars)

[Candle](https://github.com/huggingface/candle) (v0.9.2-alpha.2) is Hugging Face's minimalist ML framework in Rust, designed from the start with Wasm as a deployment target. It has the **most battle-tested Wasm demos** in the ecosystem.

**Wasm support is first-class.** The `candle-wasm-examples/` directory contains 11+ working browser demos:
- **Whisper** — speech-to-text transcription in browser
- **LLaMA2** — text generation (~120ms first-token latency on M2 MacBook)
- **Segment Anything** — interactive image segmentation
- **YOLOv8** — object detection and pose estimation
- **Phi-1.5/Phi-2** — small language model text generation
- **BERT** — semantic similarity search
- **BLIP** — image captioning
- **T5** — text generation

Live demos: [HuggingFace Candle Wasm collection](https://huggingface.co/collections/radames/candle-wasm-examples-650898dee13ff96230ce3e1f)

**Autograd:** Candle does have a native autograd engine. Tensors created as `Var` (variables) track operations, and `.backward()` computes gradients. The `candle-nn` crate provides layers, activations, optimizers (SGD, Adam, AdamW), and loss functions. However, Candle's autograd is **less mature than Burn's** — an open issue (#2674) asks about custom autograd functions, suggesting this is still evolving.

**API deliberately mirrors PyTorch** for easy porting — tensor creation, slicing, and operations look very similar.

**For differentiable-pelican:** Candle could work, but Burn is a better fit because Burn's autograd is more robust and its backend abstraction is more flexible. Candle excels at inference deployment, not gradient-based optimization loops.

#### Other Rust Frameworks

| Framework | Stars | Autograd | Wasm | Status | Notes |
|---|---|---|---|---|---|
| **[tch-rs](https://github.com/LaurentMazare/tch-rs)** | 5,280 | Yes (via libtorch) | **No** (C++ FFI) | Active | Rust bindings to PyTorch's C++ API. Cannot compile to Wasm. |
| **[linfa](https://github.com/rust-ml/linfa)** | 4,546 | No (classical ML) | Probably | Active | Rust's scikit-learn. SVM, trees, clustering. Pure Rust core likely compiles to Wasm but untested. |
| **[tract](https://github.com/sonos/tract)** | 2,772 | No (inference) | **Yes** (via [tractjs](https://github.com/bminixhofer/tractjs)) | Active | ONNX/NNEF inference in pure Rust. Proven Wasm deployment. |
| **[Luminal](https://github.com/luminal-ai/luminal)** | 2,766 | Yes | **No** | Active (YC S25) | 12-op kernel compiler. CUDA/Metal only. Pivoted to commercial GPU optimization. |
| **[dfdx](https://github.com/coreylowman/dfdx)** | 1,895 | Yes | Untested | **Dormant** (since Jul 2024) | Compile-time shape checking. Interesting design but abandoned. |
| **[WONNX](https://github.com/webonnx/wonnx)** | 1,744 | No (inference) | **Yes** (WebGPU) | Slow (last Jul 2024) | ONNX inference via WebGPU compute shaders. Pure Rust. |
| **[smartcore](https://github.com/smartcorelib/smartcore)** | 887 | No (classical ML) | **Yes** (explicit) | Active | Classical ML. Explicitly designed "WASM-first." |

#### Rust Autograd/Autodiff Libraries

Beyond full frameworks, several standalone Rust autograd libraries exist:

| Library | Stars | Mode | Wasm Likely? | Status |
|---|---|---|---|---|
| **[burn-autodiff](https://crates.io/crates/burn-autodiff)** | (part of Burn) | Reverse | **Yes (proven)** | Active |
| **[rust-autograd](https://github.com/raskr/rust-autograd)** | 500 | Reverse (lazy) | Probable (pure Rust + ndarray) | Dormant (2023) |
| **[scirs2-autograd](https://crates.io/crates/scirs2-autograd)** | N/A | Reverse | Probable (pure Rust) | Unknown |
| **[autodiff](https://crates.io/crates/autodiff)** | N/A | Forward | Probable (pure Rust) | Unknown |
| **[gad](https://docs.rs/gad)** | N/A | Reverse | Probable (pure Rust) | Unknown |
| **`std::autodiff`** (nightly) | N/A | Forward + Reverse | **Uncertain** | Experimental |

**`std::autodiff` (Rust nightly)** deserves special mention: this is an experimental feature being built into Rust's standard library itself, using [Enzyme](https://enzyme.mit.edu/) (an LLVM-based automatic differentiation framework). It supports both `#[autodiff_forward]` and `#[autodiff_reverse]` attributes. Available on nightly behind `-Zautodiff`. Actively being developed (GSoC 2025 work, RustWeek 2026 talk scheduled). If stabilized, this would make any Rust function automatically differentiable at the compiler level. Wasm compatibility is uncertain since Enzyme operates at the LLVM IR level and the wasm32 target uses a different LLVM backend.

#### The wasm-bindgen Toolchain

[wasm-bindgen](https://github.com/rustwasm/wasm-bindgen) (8,853 stars, updated 2026-02-14) is the standard for exposing Rust APIs to JavaScript when compiled to Wasm:

- Annotate structs/functions with `#[wasm_bindgen]` to export to JS
- Automatic TypeScript type definition generation
- Supports strings, numbers, classes, closures, JS objects
- Used with `wasm-pack` to produce npm-publishable packages
- Burn's MNIST web example uses this: a Rust `ImageClassifier` struct with a `classify` method gets `#[wasm_bindgen]`, and JavaScript calls it directly

**Key pattern:** JavaScript handles the UI (canvas, DOM, image preprocessing), Rust/Wasm handles the computation. Data crosses the boundary as typed arrays (efficient, minimal copying).

#### Performance: Rust-to-Wasm

| Comparison | Overhead | Source |
|---|---|---|
| Rust Wasm vs native Rust | **5-45% slower** (workload dependent) | 2025 benchmarks |
| Rust Wasm vs pure JavaScript | **8-10x faster** for compute-heavy work | 2025 benchmarks |
| Rust Wasm + SIMD vs JS | **10-15x faster** for parallelizable work | 2025 benchmarks |
| Wasmtime (Rust Wasm) vs native C | Within **5-10%** for compute-intensive tasks | 2025 benchmarks |

Rust has the **lowest Wasm overhead** of any popular language (0.003s overhead vs Go's 0.017s, Python's 0.02s in one benchmark). For the differentiable-pelican workload (arithmetic-heavy tensor operations), Rust-to-Wasm would be dramatically faster than Python-in-Pyodide and competitive with native execution.

#### Assessment: Rust-to-Wasm for Differentiable-Pelican

**Burn is the standout option.** It provides:
1. Full autograd (`Autodiff<B>`) that works with Wasm-compatible backends
2. NdArray backend for CPU Wasm (proven, `no_std`)
3. WGPU backend for GPU Wasm (WebGPU, proven for inference)
4. PyTorch-like API (`Module`, `Tensor`, optimizers, loss functions)
5. ~357 KB framework overhead in Wasm (tiny)
6. ONNX import for migrating existing models
7. Active development with 14K+ stars

The **gap** is that nobody has demonstrated Burn *training* (gradient-based optimization) in the browser. All existing Wasm demos are inference. But this is a demo gap, not a technical limitation — the `Autodiff<NdArray>` backend should work in Wasm. The differentiable-pelican use case (200 parameters, simple SDFs, 500 optimization steps) is a perfect test case to prove this out.

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

### Option F: Rewrite Core in Rust with Burn, Compile to Wasm

**Description:** Rewrite the differentiable rendering pipeline in Rust using the [Burn](https://github.com/tracel-ai/burn) deep learning framework. Burn provides autograd, tensor operations, optimizers, and loss functions with a PyTorch-like API — and compiles to Wasm via its NdArray (CPU) or WGPU (WebGPU) backends. The Rust code compiles to a `.wasm` module exposed to JavaScript via `wasm-bindgen`.

**What changes:**
- Rewrite SDF computation, soft coverage, alpha compositing in Rust with Burn tensors
- Use `Autodiff<NdArray>` (CPU) or `Autodiff<Wgpu>` (GPU) for automatic differentiation
- Use `burn::optim::Adam` with LR scheduling
- Implement loss functions (MSE, SSIM, edge) using Burn's tensor ops
- Expose API to JavaScript via `#[wasm_bindgen]`: `create_optimizer(config) → handle`, `step() → rendered_image`, `get_svg() → string`
- JavaScript handles: UI, image upload, progress display, SVG rendering
- Ship as npm package via `wasm-pack`

**API mapping from PyTorch to Burn:**

```
# PyTorch                          → Burn (Rust)
torch.Tensor                       → Tensor<B, D>
torch.nn.Module                    → #[derive(Module)] struct
nn.Parameter                       → Module fields (auto-tracked)
torch.autograd                     → Autodiff<B> backend
F.sigmoid(x)                       → activation::sigmoid(x)
F.softplus(x)                      → activation::softplus(x, beta)
F.mse_loss(a, b)                   → MseLoss::new().forward(a, b)
F.conv2d(x, kernel)                → Conv2d::forward(x)
torch.optim.Adam(lr=0.01)          → AdamConfig::new().with_lr(0.01)
torch.clamp(x, min, max)           → tensor.clamp(min, max)
```

**Pros:**
- **Near-native performance in Wasm.** Rust-to-Wasm adds only 5-45% overhead vs native. For arithmetic-heavy SDF computation, this is dramatically faster than Python-in-Pyodide (which is 3-5x slower than native CPython, which is already 10-100x slower than Rust).
- **GPU acceleration via WebGPU.** `Autodiff<Wgpu>` gives GPU-accelerated gradient computation in the browser.
- **Tiny bundle size.** Burn framework overhead is ~357 KB in Wasm. Total with a model: ~2 MB. Compare to Pyodide at ~11 MB before any packages.
- **Full autograd.** Burn's `Autodiff` decorator is production-quality reverse-mode AD. No need for custom autograd code.
- **PyTorch-like API.** The translation from PyTorch to Burn is more mechanical than translating to TensorFlow.js — similar concepts (Module, Tensor, backward, optimizers).
- **No Python runtime in the browser.** Eliminates the entire Pyodide/Python-in-Wasm complexity.
- **Reusable beyond browser.** The Rust code also runs natively (CLI, server) by swapping backends. One codebase, multiple deployment targets.
- **Type safety.** Rust's type system catches tensor shape mismatches at compile time.

**Cons:**
- **Rust learning curve.** Porting Python to Rust is a bigger cognitive shift than Python to TypeScript.
- **Unproven for training in Wasm.** Nobody has demonstrated Burn's autograd running in a browser Wasm context. This would be a first. May hit unexpected issues with Wasm memory limits, WebGPU compute shader limitations, or `Autodiff<Wgpu>` edge cases.
- **Two language ecosystems.** Maintaining Rust + JavaScript (for the browser UI) requires two toolchains.
- **Burn is pre-1.0.** API may have breaking changes. Though it's actively maintained (v0.20.1, 14K stars), it's not as battle-tested as PyTorch.
- **ONNX import is partial.** If the goal is to port a trained PyTorch model, not all ops may be supported in `burn-onnx`.
- **No equivalent to `rich`, `anthropic`, `pydantic`.** The Python orchestration layer (LLM-guided topology search, CLI UI) would need a separate solution — either kept in Python (server-side) or rewritten in TypeScript.

**Feasibility: Medium-High.** The core differentiable rendering algorithm maps cleanly to Burn's API. The main risk is being the first to run `Autodiff` training in browser Wasm — but the workload is small enough (200 parameters, simple graph) that it's a reasonable bet. This would be a compelling demo for the Burn project itself.

### Option G: Hybrid — Burn (Rust/Wasm) Core + TypeScript UI

**Description:** Combine the best of Option F with a clean separation of concerns. The differentiable renderer is implemented in Rust/Burn, compiled to Wasm, and published as an npm package. A TypeScript application handles the browser UI, image handling, and orchestration. The Rust/Wasm module is a pure computation engine with a clean API boundary.

**Architecture:**
```
┌─────────────────────────────────────────────┐
│  Browser                                     │
│                                              │
│  ┌──────────────────┐  ┌──────────────────┐ │
│  │  TypeScript UI    │  │  Rust/Burn Wasm  │ │
│  │                   │  │                  │ │
│  │  - Image upload   │◄─┤  - SDF eval      │ │
│  │  - Progress bar   │  │  - Autograd      │ │
│  │  - SVG display    │─►│  - Adam optim    │ │
│  │  - Parameter UI   │  │  - Loss compute  │ │
│  │  - LLM API calls  │  │  - Rendering     │ │
│  └──────────────────┘  └──────────────────┘ │
│         ▲                      ▲             │
│         │  wasm-bindgen        │ WebGPU      │
│         │  typed arrays        │             │
└─────────────────────────────────────────────┘
```

**Pros:**
- All pros of Option F, plus clean separation of concerns
- TypeScript handles what it's good at (DOM, UI, fetch for LLM APIs)
- Rust handles what it's good at (fast computation, autograd, Wasm)
- The Rust/Wasm package is reusable in any JS context (React, Svelte, plain HTML)
- Could publish as an npm package: `@differentiable-pelican/core`

**Cons:**
- Two-language project (Rust + TypeScript)
- `wasm-bindgen` boundary requires careful API design
- Build toolchain complexity (cargo + wasm-pack + npm/vite)

**Feasibility: Medium-High.** This is arguably the most architecturally sound option for a production browser application. The two-language split follows the grain of each language's strengths.

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
| PyTorch autograd | **Critical** | **No** | **Burn `Autodiff<B>`**, TF.js `tf.grad`, HIPS/autograd |
| PyTorch nn.Module | **Critical** | **No** | **Burn `#[derive(Module)]`**, TF.js layers, custom classes |
| PyTorch optimizers | **Critical** | **No** | **Burn `Adam`**, TF.js `tf.train.adam`, custom (~30 LOC) |
| torch.conv2d | Used | **No** | **Burn `Conv2d`**, scipy.signal.convolve2d |
| torch.sigmoid | Used | **No** | **Burn `activation::sigmoid`**, NumPy `1/(1+exp(-x))` |
| Anthropic SDK | Optional | Partial (needs HTTP) | `fetch()` from JS |
| rich | CLI only | N/A | Browser UI |
| imageio | GIF export | Likely (pure Python) | Canvas recording |

### The Core Technical Barrier

**The single biggest barrier is replacing PyTorch's autograd engine.** Everything else either works in Pyodide already or has a straightforward alternative. The question is how to get automatic differentiation:

1. **Burn's `Autodiff<B>`** (Rust → Wasm): The most promising option. Production-quality reverse-mode AD with a PyTorch-like API, compiling to ~357 KB Wasm. The `Autodiff<NdArray>` backend should work in browser Wasm, though nobody has demonstrated training (vs inference) in this context yet. Burn's `Autodiff<Wgpu>` could additionally provide GPU-accelerated gradients via WebGPU.

2. **TensorFlow.js** (JavaScript): Provides `tf.grad()` with GPU acceleration. Requires a TypeScript rewrite but is proven and well-documented.

3. **HIPS/autograd** (pure Python in Pyodide): Wraps NumPy and provides `grad()` and `jacobian()`. The differentiable-pelican computation graph (SDFs → sigmoid → compositing → loss) should be within its capabilities. Needs testing in Pyodide. Performance will be significantly slower.

4. **Candle** (Rust → Wasm): Has autograd (`Var` + `.backward()`), first-class Wasm support, and HuggingFace backing. Less mature autograd than Burn but still viable.

5. **Custom tape-based AD**: For this specific computation graph, a custom autograd is feasible in any language. The operations are: basic arithmetic, sigmoid, softplus, conv2d, MSE, and Gaussian-windowed statistics (SSIM). A tape-based system recording these ops and computing vjps would be ~200-500 lines of code.

### Performance Estimates

For a 256×256 image with 35 shapes, one forward pass involves:
- 35 SDF evaluations × 65,536 pixels = ~2.3M operations
- Coverage computation (sigmoid): ~2.3M operations
- Alpha compositing: ~2.3M operations per shape = ~80M operations
- Loss computation: ~65K operations (MSE) + conv2d (Sobel, SSIM)

| Approach | Forward Pass | 500 Steps | Notes |
|---|---|---|---|
| PyTorch CPU (native) | ~5 ms | ~5 sec | Current baseline |
| **Burn NdArray (Rust→Wasm CPU)** | **~5-10 ms** | **~5-10 sec** | **5-45% Wasm overhead vs native Rust** |
| **Burn WGPU (Rust→Wasm WebGPU)** | **~2-5 ms** | **~2-5 sec** | **GPU parallel, WebGPU in browser** |
| NumPy in Pyodide | ~25-50 ms | ~25-50 sec | 5-10x slower than native CPython |
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

### Summary of All Options

| Option | Approach | Feasibility | Performance | Bundle Size | Effort |
|---|---|---|---|---|---|
| **A** | Pyodide + NumPy + HIPS/autograd | Medium | Slow (25-50s) | ~15 MB+ | Moderate rewrite |
| **B** | TypeScript + TensorFlow.js | High | Fast (2-5s GPU) | ~200 KB gzip | Full rewrite |
| **C** | Hybrid Pyodide + JS/WebGPU | Medium-High | Good | ~15 MB+ | Complex |
| **D** | Server-side + browser viewer | Very High | N/A | Minimal | Minimal |
| **E** | Pure WebGPU compute shaders | Medium | Fastest | Minimal | Research-level |
| **F** | **Rust/Burn → Wasm** | **Medium-High** | **Fast (5-10s CPU, 2-5s GPU)** | **~2 MB** | **Full rewrite** |
| **G** | **Rust/Burn Wasm + TypeScript UI** | **Medium-High** | **Fast** | **~2 MB** | **Full rewrite** |

### Recommended Path: Option F/G (Rust/Burn → Wasm) or Option B (TypeScript + TensorFlow.js)

**Option F/G (Rust/Burn)** is the most exciting path. Burn provides a production-quality autograd engine that compiles to a ~2 MB Wasm module — dramatically smaller and faster than any Python-in-Wasm approach. The API maps closely to PyTorch, making the port more mechanical than it might seem. The main risk is being the first to run Burn's autograd in browser Wasm for training (vs. inference), but the workload is simple enough that this is a reasonable bet. If successful, this would also be a compelling contribution to the Burn ecosystem and a proof-of-concept for "PyTorch-level differentiable programming in the browser."

**Option B (TypeScript + TensorFlow.js)** remains the safest high-performance option. TF.js autograd is proven in the browser, GPU-accelerated, and well-documented. The trade-off is that TypeScript is less ergonomic for numerical computing than Rust (with Burn) or Python.

**Choosing between them:**
- If you want the best performance/size ratio and are comfortable with Rust → **Option F/G**
- If you want the most proven, lowest-risk browser path → **Option B**
- If you want to stay in Python and accept slower performance → **Option A**

### Key Insight

The differentiable-pelican computation is relatively simple compared to typical deep learning workloads. It's not a neural network — it's a handful of analytic SDFs composed through differentiable operations. This makes it an unusually good candidate for non-PyTorch autograd systems. The total parameter count is tiny (3-8 parameters per shape × 35 shapes ≈ 200 parameters). The computation graph is static and shallow. This is well within reach of lightweight autograd implementations in any language.

This simplicity is what makes the Burn/Rust path viable despite being unproven: you don't need Burn's full training infrastructure (data loaders, distributed training, checkpointing). You need tensors, autograd, and an Adam optimizer — the core primitives that are best-tested in any framework.

### What Would Move the Needle

Several developments would significantly change the feasibility picture:

1. **Burn training demo in Wasm**: If someone demonstrates Burn's `Autodiff<NdArray>` or `Autodiff<Wgpu>` running a training loop in the browser, it would de-risk Option F/G entirely. The differentiable-pelican use case is arguably the perfect first demo.
2. **`std::autodiff` stabilization in Rust**: If Rust's experimental Enzyme-based autodiff becomes stable and works with wasm32 targets, any Rust numeric code becomes differentiable by default — no framework needed.
3. **PEP 783 approval + PyTorch Wasm wheels**: If PyTorch ever publishes Wasm wheels, everything changes. But this is unlikely in the near term due to the size and complexity of PyTorch's native code.
4. **Pyodide SIMD optimization**: Could cut the Python-in-Wasm performance gap from 5-10x to 2-3x.
5. **WebGPU maturity**: As WebGPU reaches full support in Safari and Firefox, GPU-accelerated Wasm options (Burn WGPU, TF.js WebGPU) become universally available.

---

## Next Steps

### Highest priority (Rust/Burn path)
- [ ] Spike: Set up a minimal Burn project with `Autodiff<NdArray>`, compile to Wasm with wasm-pack, verify `.backward()` works in browser
- [ ] Prototype: Implement one SDF (circle) + sigmoid coverage + MSE loss in Burn, run 100 Adam optimization steps in browser Wasm
- [ ] Benchmark: Measure forward pass and backward pass times for Burn NdArray Wasm vs. PyTorch CPU (native) on the same SDF computation
- [ ] Evaluate: Try `Autodiff<Wgpu>` in browser — does WebGPU-accelerated training work?
- [ ] Design: Define the wasm-bindgen API boundary — what goes in Rust vs. TypeScript?

### Secondary (alternative paths)
- [ ] Prototype: Test HIPS/autograd in Pyodide (does it install via micropip? does `grad()` work?)
- [ ] Evaluate: Try porting the `renderer.py` forward pass to TensorFlow.js, measure performance with WebGPU
- [ ] Research: Look into [AnyWidget](https://anywidget.dev/) as a bridge between Python (Pyodide/JupyterLite) and custom JS rendering

### Community engagement
- [ ] Community: If the Burn Wasm training spike succeeds, write it up as a blog post / example PR for the Burn project
- [ ] Community: Open a discussion on the Pyodide repo about lightweight autograd for scientific computing in browser

## References

### Rust ML Frameworks
- [Burn](https://github.com/tracel-ai/burn) — Deep learning framework in Rust with autograd + Wasm support (14,358 stars)
- [Burn MNIST Web Example](https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web) — Proven Burn-to-Wasm deployment
- [Candle](https://github.com/huggingface/candle) — Minimalist ML framework by HuggingFace with first-class Wasm (19,389 stars)
- [Candle Wasm Examples](https://huggingface.co/collections/radames/candle-wasm-examples-650898dee13ff96230ce3e1f) — Live browser demos (Whisper, LLaMA, SAM, etc.)
- [tch-rs](https://github.com/LaurentMazare/tch-rs) — Rust bindings to libtorch (5,280 stars, no Wasm)
- [linfa](https://github.com/rust-ml/linfa) — Classical ML in Rust (4,546 stars)
- [tract](https://github.com/sonos/tract) — ONNX inference in pure Rust (2,772 stars)
- [Luminal](https://github.com/luminal-ai/luminal) — GPU kernel compiler (2,766 stars, no Wasm)
- [WONNX](https://github.com/webonnx/wonnx) — WebGPU ONNX runtime in Rust (1,744 stars)
- [dfdx](https://github.com/coreylowman/dfdx) — Shape-checked DL in Rust (1,895 stars, dormant)
- [smartcore](https://github.com/smartcorelib/smartcore) — Classical ML, WASM-first (887 stars)

### Rust Autograd/Autodiff
- [burn-autodiff](https://crates.io/crates/burn-autodiff) — Burn's autodiff backend decorator (most mature, Wasm-proven)
- [rust-autograd](https://github.com/raskr/rust-autograd) — TensorFlow-style autograd in pure Rust (500 stars)
- [std::autodiff (nightly)](https://doc.rust-lang.org/nightly/std/autodiff/index.html) — Experimental Enzyme-based AD in Rust stdlib
- [Enzyme GSoC 2025](https://blog.karanjanthe.me/posts/enzyme-autodiff-rust-gsoc/) — Stabilization progress
- [std::autodiff at RustWeek 2026](https://2026.rustweek.org/talks/manuel_drehwald/) — Upcoming talk on stabilization

### Rust-to-Wasm Tooling
- [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen) — Rust/JS FFI for Wasm (8,853 stars)
- [wasm-pack](https://github.com/nicktomlin/autograd) — Build Rust Wasm packages for npm
- [CubeCL](https://github.com/tracel-ai/cubecl) — Burn's GPU programming language (compiles to WebGPU)

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
