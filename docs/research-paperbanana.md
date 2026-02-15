# Research Brief: PaperBanana

> Automated Academic Illustration for AI Scientists

**Source**: [github.com/llmsresearch/paperbanana](https://github.com/llmsresearch/paperbanana) (v0.1.2, MIT)
**Paper**: arXiv:2601.23265, Zhu et al. (unofficial community implementation)
**Local clone**: `attic/paperbanana/`

---

## Summary

PaperBanana is a multi-agent pipeline that generates publication-quality academic diagrams and statistical plots from natural-language descriptions of paper methodology sections. It uses Google Gemini for both VLM reasoning (planning, critique) and image generation, orchestrating 5 specialized agents across a two-phase architecture.

Key design decisions:
- **Textual description as intermediate representation** -- the system never generates images directly from source text; instead it produces a detailed natural-language "blueprint" that is iteratively refined before rendering.
- **Visual in-context learning** -- reference diagram images are passed directly to the VLM alongside text, enabling style transfer.
- **Dual rendering strategy** -- methodology diagrams use text-to-image generation (Gemini 3 Pro Image), while statistical plots generate executable matplotlib code.

---

## Architecture Overview

```
Phase 1 -- Linear Planning (sequential)
  Retriever -> Planner -> Stylist

Phase 2 -- Iterative Refinement (loop, up to N rounds)
  Visualizer -> Critic -> [revised description] -> Visualizer -> ...
```

The pipeline is orchestrated by `PaperBananaPipeline` in `paperbanana/core/pipeline.py`. All agents inherit from `BaseAgent` (`paperbanana/agents/base.py`), which provides prompt loading (`prompts/{diagram|plot}/{agent_name}.txt`) and formatting via Python `str.format()`.

---

## Phase 1: Linear Planning

### Step 1 -- Retriever (`paperbanana/agents/retriever.py`)

**Purpose**: Select the top-N most relevant reference examples from a curated pool of 13 academic papers for few-shot learning.

**Inputs**:
- `source_context` -- methodology section text
- `caption` -- communicative intent / figure caption
- `candidates` -- all 13 reference examples (formatted as Paper ID + Caption + truncated methodology)
- `num_examples` -- how many to select (default: 10 for diagrams, 5 for plots)

**Process**:
1. If `len(candidates) <= num_examples`, returns all candidates immediately (short circuit).
2. Formats candidates into a text block with ID, caption, and first 300 chars of methodology.
3. Loads `prompts/diagram/retriever.txt` (or `plot/`), fills placeholders.
4. Calls `vlm.generate()` at **temperature 0.3**, requesting JSON response.
5. Parses `{"selected_ids": [...]}` (also accepts `top_10_papers` / `top_10_plots` for paper-format compatibility).
6. Maps IDs back to `ReferenceExample` objects. Falls back to returning all candidates on JSON parse failure.

**Selection logic** (from prompt):
- For **diagrams**: Matches on research domain (Agent/Reasoning, Vision/Perception, Generative/Learning, Science/Applications) and visual intent (Framework, Pipeline, Detailed Module, Performance Chart). Ranking: Same Topic AND Same Visual Intent > Same Visual Intent only > avoid Different Visual Intent.
- For **plots**: Matches on data characteristics (categorical vs numerical, dimensionality) and plot type (bar, scatter, line, pie, heatmap, radar). Ranking: Same Data Type AND Same Plot Type > Same Plot Type with compatible data.

**Key code path**: `retriever.py:30-92`

### Step 2 -- Planner (`paperbanana/agents/planner.py`)

**Purpose**: Generate a comprehensive textual description of the target diagram using in-context learning. Corresponds to paper equation 4: `P = VLM_plan(S, C, {(S_i, C_i, I_i)})`.

**Inputs**:
- `source_context`, `caption`, `examples` (from Retriever)

**Process**:
1. Formats examples as text blocks: Caption + Source Context (first 500 chars) + image reference.
2. **Loads actual JPG images** from the reference set (`_load_example_images`) and passes them as PIL Image objects alongside the text prompt -- this is the visual in-context learning mechanism.
3. Calls `vlm.generate()` at **temperature 0.7**, max 4096 tokens, text response.

**Description coverage** (7 dimensions from prompt):
1. Overall layout (flow direction, sections)
2. Components (boxes, modules, labels)
3. Connections (arrows, data flows)
4. Groupings (colored regions, dashed borders)
5. Labels and annotations (text, math notations)
6. Input/Output
7. Styling (background fills, color palettes)

**Critical difference between diagram and plot planners**:
- Diagram planner: natural language colors ONLY ("soft sky blue"), NEVER hex codes
- Plot planner: specific color codes, font sizes, line widths ARE allowed
- Plot planner: must enumerate every raw data point coordinate

**Key code path**: `planner.py:32-80`

### Step 3 -- Stylist (`paperbanana/agents/stylist.py`)

**Purpose**: Refine the Planner's output for publication-quality aesthetics, preserving content.

**Inputs**:
- `description` (from Planner), `guidelines` (loaded NeurIPS style guide), `source_context`, `caption`

**Process**:
1. Loads domain-appropriate guidelines (methodology or plot style guide from `data/guidelines/`).
2. Falls back to hardcoded default guidelines if none provided.
3. Calls `vlm.generate()` at **temperature 0.5**, max 4096 tokens.

**Diagram stylist rules** (5 instructions from prompt):
1. Preserve Aesthetics -- natural language colors, NEVER hex codes/pixel dimensions/CSS
2. Intervene Only When Necessary -- minimal edits if already good
3. Respect Diversity -- adapt to different diagram styles
4. Enrich Details -- add specifics where vague (e.g., "a rounded rectangle with soft blue fill")
5. Preserve Content -- NO adding/removing/modifying components

**Plot stylist** is simpler (3 instructions): Enrich Details, Preserve Content, Context Awareness.

**Key code path**: `stylist.py:34-77`

---

## Phase 2: Iterative Refinement

The pipeline enters a loop of up to `refinement_iterations` rounds (default: 3). The loop is in `pipeline.py:258-331`.

### Step 4 -- Visualizer (`paperbanana/agents/visualizer.py`)

**Purpose**: Render the textual description into an image.

**Branching logic** (`visualizer.py:67-70`):
- `DiagramType.METHODOLOGY` -> `_generate_diagram()` -- text-to-image
- `DiagramType.STATISTICAL_PLOT` -> `_generate_plot()` -- code generation + execution

#### Diagram path (`_generate_diagram`, line 72-97):

1. Loads `prompts/diagram/visualizer.txt`, fills `{description}`.
2. Calls `image_gen.generate()` with **width=1792, height=1024**.
3. The prompt instructs Gemini 3 Pro Image to act as "an expert scientific diagram illustrator" and emphasizes clear, readable English labels.
4. Saves result as PNG.

#### Plot path (`_generate_plot`, line 99-140):

1. Appends raw data as JSON to the description.
2. Loads `prompts/plot/visualizer.txt`, fills `{description}`.
3. Calls `vlm.generate()` at **temperature 0.3**, max 4096 tokens -- generates Python code (matplotlib/seaborn).
4. Extracts code from markdown fences (`_extract_code`).
5. **Strips any VLM-generated `OUTPUT_PATH` assignments** via regex (line 160) and injects the real output path.
6. Writes code to a temp file and runs it in a **subprocess with 60-second timeout** (`_execute_plot_code`).
7. On failure, creates a blank white placeholder image (1024x768).

**Key code path**: `visualizer.py:45-187`

### Step 5 -- Critic (`paperbanana/agents/critic.py`)

**Purpose**: Evaluate the generated image against the source context and provide feedback.

**Inputs**:
- `image_path` (from Visualizer), `description`, `source_context`, `caption`

**Process**:
1. Loads the generated image as a PIL Image.
2. Loads `prompts/{diagram|plot}/critic.txt`, fills placeholders.
3. Calls `vlm.generate()` at **temperature 0.3**, max 4096 tokens, JSON response, **with the image passed as visual input**.
4. Parses response into `CritiqueResult`:
   - `critic_suggestions`: list of identified issues
   - `revised_description`: updated description if revision needed (null if publication-ready)
   - `needs_revision` property: `True` if any suggestions exist

**Critique dimensions** (from prompt):

For **diagrams**:
- Content: Fidelity/alignment with methodology, text QA (typos, garbled text, hex codes rendered as text), factual correctness, no caption in image
- Presentation: Clarity/readability, legend management

For **plots** (additional checks):
- Data Fidelity: all quantitative values correct, no data hallucinated/omitted
- Validation of Values: numerical values, axis scales, data points
- Overlap/Layout: overlapping text labels, hatching, pie chart labels
- Code execution failure handling: if "[SYSTEM NOTICE]" appears, analyze logical errors

**Key code path**: `critic.py:31-95`

### Refinement loop termination (`pipeline.py:318-331`):

```python
if critique.needs_revision and critique.revised_description:
    current_description = critique.revised_description
    # continues to next iteration
else:
    break  # early exit -- image is publication-ready
```

The loop also naturally terminates after `refinement_iterations` rounds.

---

## Evaluation System

The evaluation system (`paperbanana/evaluation/judge.py`) implements **VLM-as-Judge referenced comparison** from paper Section 4.2.

### Four dimensions:

| Dimension | Primary? | Inputs | Key criterion |
|-----------|----------|--------|---------------|
| **Faithfulness** | Yes | source_context, caption, both images | Technical alignment; veto on hallucination, logical contradiction, scope violation, gibberish |
| **Readability** | Yes | caption, both images | Visual flow and legibility; veto on noise, occlusion, spaghetti routing, illegible fonts, low contrast, black background. Default is "Both are good" |
| **Conciseness** | No | source_context, caption, both images | Visual signal-to-noise ratio; veto on textual overload (>15 words in boxes), literal copying, math dump |
| **Aesthetics** | No | caption, both images | Visual polish; veto on artifacts, neon colors, black background |

Each dimension returns a winner: `Model | Human | Both are good | Both are bad`.

### Hierarchical aggregation (`judge.py:136-162`):

1. Aggregate primary dimensions (Faithfulness + Readability):
   - If both agree on a side -> that side wins
   - If one wins + one ties -> winner wins
   - If split or both tie -> inconclusive
2. If primary is inconclusive, aggregate secondary dimensions (Conciseness + Aesthetics) using the same logic.
3. If still inconclusive -> "Both are good" (default tie).

**Scoring**: Model wins = 100.0, Human wins = 0.0, Tie = 50.0.

**Judge model**: configured as `gpt-4o` in `configs/eval/vlm_judge.yaml` (at **temperature 0.1**, max 1024 tokens).

---

## Reference Data

### Curated reference set (`data/reference_sets/index.json`)

13 verified methodology diagrams from recent arXiv papers, organized in 4 categories:

| Category | Count | Examples |
|----------|-------|----------|
| `agent_reasoning` | 5 | GlimpRouter (entropy-based LLM/SLM routing), SEEM (memory agents), Dr. Zero (self-evolution), MAXS (LLM rollout) |
| `generative_learning` | 5 | ReasonMark (watermarking), X-Coder (code features), Codified Foreshadow-Payoff (story gen), Flexibility Trap (AR vs diffusion), Stable-DiffCoder |
| `vision_perception` | 2 | Fast-ThinkAct (robotics distillation), HERMES (streaming video QA) |
| `science_applications` | 1 | StructMAE (graph masking) |

Each example includes: `id`, `source_context` (full methodology section text), `caption`, `image_path` (JPG), `category`, `aspect_ratio`.

### Style guidelines (`data/guidelines/`)

**Methodology style guide** (`methodology_style_guide.md`, 176 lines):
- "NeurIPS 2025 Method Diagram Aesthetics Guide"
- Color philosophy: soft tech/scientific pastels, 10-15% opacity backgrounds, medium saturation for active modules, high saturation reserved for highlights
- Shapes: rounded rectangles (80% dominant), 3D cuboids for tensors, cylinders exclusively for databases
- Arrows: orthogonal for architectures, curved for system logic, solid for forward pass, dashed for gradients
- Domain-specific styles: Agent/LLM (cartoony, chat bubbles), CV (spatial, dense), Theoretical (minimalist), Generative (dynamic, noise metaphors)

**Plot style guide** (`plot_style_guide.md`, 149 lines):
- White backgrounds, Viridis/Magma for sequential, never Jet/Rainbow
- Type-specific: bar charts (black outlines), line charts (always include markers), pie/donut (white separators), scatter (shape + color coding), heatmaps (values inside cells)

---

## Provider Architecture

### Provider registry (`paperbanana/providers/registry.py`)

Factory pattern with two provider types:

| Provider Type | Implementations | API Key |
|---------------|----------------|---------|
| **VLM** | `gemini` (Gemini VLM), `openrouter` (OpenRouter) | `GOOGLE_API_KEY` or `OPENROUTER_API_KEY` |
| **Image Gen** | `google_imagen` (Gemini 3 Pro Image), `openrouter_imagen` | Same |

### Agent temperature settings

| Agent | Temperature | Max Tokens | Response Format |
|-------|------------|------------|-----------------|
| Retriever | 0.3 | default | JSON |
| Planner | 0.7 | 4096 | text |
| Stylist | 0.5 | 4096 | text |
| Visualizer (diagram) | N/A (image gen) | N/A | image (1792x1024) |
| Visualizer (plot) | 0.3 | 4096 | text (code) |
| Critic | 0.3 | 4096 | JSON |
| Evaluation Judge | 0.1 | 1024 | JSON |

Temperature rationale: creative agents (Planner, Stylist) run warmer; selection and evaluation agents (Retriever, Critic, Judge) run cold for consistency.

---

## Interfaces

### CLI (Typer)

```bash
paperbanana generate --input method.txt --caption "..." [--iterations 3]
paperbanana plot --data results.csv --intent "..."
paperbanana evaluate --generated img.png --reference ref.png --context method.txt --caption "..."
paperbanana setup  # interactive API key wizard
```

### Python API

```python
pipeline = PaperBananaPipeline(settings=Settings(...))
result = await pipeline.generate(GenerationInput(
    source_context="...",
    communicative_intent="...",
    diagram_type=DiagramType.METHODOLOGY,
))
```

### MCP Server (`mcp_server/`)

Three tools via FastMCP: `generate_diagram`, `generate_plot`, `evaluate_diagram`. Installable via `uvx --from paperbanana[mcp] paperbanana-mcp`.

### Claude Code Skills

`/generate-diagram`, `/generate-plot`, `/evaluate-diagram` -- shipped in `.claude/skills/`.

---

## Dependencies

Core: `pydantic>=2.0`, `pydantic-settings>=2.0`, `google-genai>=1.0`, `pillow>=10.0`, `typer>=0.12`, `rich>=13.0`, `httpx>=0.27`, `matplotlib>=3.8`, `pandas>=2.0`, `structlog>=24.0`, `tenacity>=8.0`.

Optional: `fastmcp>=2.0` (MCP), `pymupdf>=1.24` (PDF input).

Build: `hatchling`. Linting: `ruff`. Testing: `pytest` + `pytest-asyncio`.

---

## Key Design Observations

1. **Description-as-blueprint pattern**: The entire pipeline revolves around an intermediate textual description. The Planner creates it, the Stylist refines it, the Visualizer renders it, and the Critic revises it. This avoids the "single-shot image generation" failure mode where text-to-image models hallucinate or miss components.

2. **Asymmetric rendering**: Diagrams use native image generation (Gemini 3 Pro Image at 1792x1024) while plots generate executable Python code. This is pragmatic -- text-to-image models are poor at precise data visualization, while matplotlib excels at it.

3. **Visual few-shot learning**: The Planner receives actual reference images (not just text descriptions), enabling style transfer. This is the `I_i` component in the paper's equation.

4. **Conservative critic**: The Critic's fallback on JSON parse failure is to return empty suggestions (no revision needed), which means the pipeline errs toward keeping the current image rather than entering a broken revision loop.

5. **Minimal prompt engineering for the Visualizer**: The diagram Visualizer prompt is the shortest in the system -- just a role statement and the description. This pushes complexity into the description itself (created by Planner+Stylist), keeping the image generation prompt clean.

6. **Plot code sandboxing**: VLM-generated OUTPUT_PATH assignments are stripped via regex before injection, preventing path traversal. The code runs in a subprocess with a 60-second timeout.

7. **Evaluation is separate from generation**: The VLM-as-Judge system is not used during the generation loop (the Critic fills that role). Evaluation is a standalone CLI command for benchmarking against human references.
