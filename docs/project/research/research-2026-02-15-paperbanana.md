# Research: PaperBanana -- Multi-Agent Academic Illustration Generation

**Date:** 2026-02-15

**Author:** Claude (automated research)

**Status:** Complete

## Overview

PaperBanana is a multi-agent pipeline that generates publication-quality academic
diagrams and statistical plots from natural-language descriptions of paper methodology
sections. It uses Google Gemini for both VLM reasoning (planning, critique) and image
generation, orchestrating 5 specialized agents across a two-phase architecture.

We are researching this project to understand its multi-agent orchestration patterns,
VLM-as-judge evaluation methodology, and iterative refinement approach -- all of which
may inform our own SVG optimization pipeline design.

**Source:** [github.com/llmsresearch/paperbanana](https://github.com/llmsresearch/paperbanana) (v0.1.2, MIT)
**Paper:** arXiv:2601.23265, Zhu et al. (unofficial community implementation)
**Local clone:** `attic/paperbanana/`

## Questions to Answer

1. How does PaperBanana orchestrate multiple LLM agents for diagram generation?
2. What is the iterative refinement loop and how does the Critic agent decide when
   to stop revising?
3. How does the VLM-as-Judge evaluation system work, and what dimensions does it
   measure?
4. What prompt engineering patterns are used across the 5 agents?
5. What can we learn from PaperBanana's approach for our own SVG optimization work?

## Scope

**Included:** Full code-level analysis of the pipeline architecture, all 5 agents,
prompt engineering, evaluation system, provider architecture, reference data, and
interface layer (CLI, MCP, Python API).

**Excluded:** Running the pipeline end-to-end (requires Google API key), performance
benchmarking, comparison against the original paper's results.

## Findings

### Architecture: Two-Phase Pipeline

```
Phase 1 -- Linear Planning (sequential)
  Retriever -> Planner -> Stylist

Phase 2 -- Iterative Refinement (loop, up to N rounds)
  Visualizer -> Critic -> [revised description] -> Visualizer -> ...
```

The pipeline is orchestrated by `PaperBananaPipeline` in
`paperbanana/core/pipeline.py`. All agents inherit from `BaseAgent`
(`paperbanana/agents/base.py`), which provides prompt loading
(`prompts/{diagram|plot}/{agent_name}.txt`) and formatting via Python `str.format()`.

Three key design decisions:
- **Textual description as intermediate representation** -- the system never generates
  images directly from source text; it produces a detailed natural-language "blueprint"
  that is iteratively refined before rendering.
- **Visual in-context learning** -- reference diagram images are passed directly to the
  VLM alongside text, enabling style transfer.
- **Dual rendering strategy** -- methodology diagrams use text-to-image generation
  (Gemini 3 Pro Image), while statistical plots generate executable matplotlib code.

### Phase 1: Linear Planning

#### Retriever (`paperbanana/agents/retriever.py`)

Selects the top-N most relevant reference examples from a curated pool of 13 academic
papers for few-shot learning.

**Inputs:** `source_context` (methodology section text), `caption` (communicative
intent), `candidates` (all 13 reference examples), `num_examples` (default: 10 for
diagrams, 5 for plots).

**Process:**
1. Short circuit: if `len(candidates) <= num_examples`, returns all immediately.
2. Formats candidates as text blocks (ID, caption, first 300 chars of methodology).
3. Loads `prompts/diagram/retriever.txt` (or `plot/`), fills placeholders.
4. Calls `vlm.generate()` at **temperature 0.3**, requesting JSON response.
5. Parses `{"selected_ids": [...]}` (also accepts `top_10_papers`/`top_10_plots` for
   paper-format compatibility).
6. Maps IDs back to `ReferenceExample` objects. Falls back to returning all candidates
   on JSON parse failure.

**Selection logic (from prompt):**
- **Diagrams:** Matches on research domain (Agent/Reasoning, Vision/Perception,
  Generative/Learning, Science/Applications) and visual intent (Framework, Pipeline,
  Detailed Module, Performance Chart). Ranking: Same Topic AND Same Visual Intent >
  Same Visual Intent only > avoid Different Visual Intent.
- **Plots:** Matches on data characteristics (categorical vs numerical, dimensionality)
  and plot type (bar, scatter, line, pie, heatmap, radar). Ranking: Same Data Type AND
  Same Plot Type > Same Plot Type with compatible data.

#### Planner (`paperbanana/agents/planner.py`)

Generates a comprehensive textual description of the target diagram using in-context
learning. Corresponds to paper equation 4: `P = VLM_plan(S, C, {(S_i, C_i, I_i)})`.

**Process:**
1. Formats examples as text blocks: Caption + Source Context (first 500 chars) + image
   reference.
2. **Loads actual JPG images** from the reference set (`_load_example_images`) and
   passes them as PIL Image objects alongside the text prompt -- this is the visual
   in-context learning mechanism.
3. Calls `vlm.generate()` at **temperature 0.7**, max 4096 tokens, text response.

**Description coverage (7 dimensions):**
1. Overall layout (flow direction, sections)
2. Components (boxes, modules, labels)
3. Connections (arrows, data flows)
4. Groupings (colored regions, dashed borders)
5. Labels and annotations (text, math notations)
6. Input/Output
7. Styling (background fills, color palettes)

**Critical distinction:** Diagram planner uses natural language colors ONLY ("soft sky
blue"), NEVER hex codes. Plot planner allows specific color codes, font sizes, line
widths, and must enumerate every raw data point coordinate.

#### Stylist (`paperbanana/agents/stylist.py`)

Refines the Planner's output for publication-quality aesthetics, preserving content.

**Inputs:** `description` (from Planner), `guidelines` (loaded NeurIPS style guide),
`source_context`, `caption`.

**Process:**
1. Loads domain-appropriate guidelines (methodology or plot from `data/guidelines/`).
2. Falls back to hardcoded default guidelines if none provided.
3. Calls `vlm.generate()` at **temperature 0.5**, max 4096 tokens.

**Diagram stylist rules (5 instructions):**
1. Preserve Aesthetics -- natural language colors, NEVER hex codes/pixel dimensions/CSS
2. Intervene Only When Necessary -- minimal edits if already good
3. Respect Diversity -- adapt to different diagram styles
4. Enrich Details -- add specifics where vague (e.g., "a rounded rectangle with soft
   blue fill")
5. Preserve Content -- NO adding/removing/modifying components

### Phase 2: Iterative Refinement

The pipeline enters a loop of up to `refinement_iterations` rounds (default: 3). The
loop is in `pipeline.py:258-331`.

#### Visualizer (`paperbanana/agents/visualizer.py`)

Renders the textual description into an image.

**Branching logic:**
- `DiagramType.METHODOLOGY` -> `_generate_diagram()` -- text-to-image model
- `DiagramType.STATISTICAL_PLOT` -> `_generate_plot()` -- code generation + execution

**Diagram path:**
1. Loads prompt, calls `image_gen.generate()` with **width=1792, height=1024**.
2. Saves result as PNG.

**Plot path:**
1. Appends raw data as JSON to the description.
2. Calls `vlm.generate()` at **temperature 0.3** -- generates matplotlib/seaborn code.
3. Strips VLM-generated `OUTPUT_PATH` assignments via regex, injects the real path.
4. Runs code in a **subprocess with 60-second timeout**.
5. On failure, creates a blank white placeholder image (1024x768).

#### Critic (`paperbanana/agents/critic.py`)

Evaluates the generated image against the source context and provides feedback.

**Process:**
1. Loads the generated image as a PIL Image.
2. Calls `vlm.generate()` at **temperature 0.3**, JSON response, **with the image as
   visual input**.
3. Parses into `CritiqueResult`: `critic_suggestions` (issue list),
   `revised_description` (updated description or null if publication-ready),
   `needs_revision` (True if any suggestions exist).

**Critique dimensions for diagrams:** Content fidelity/alignment, text QA
(typos, garbled text, hex codes rendered as text), factual correctness, no caption in
image; clarity/readability, legend management.

**Additional plot checks:** Data fidelity, numerical value validation, axis scales,
overlapping text labels, code execution failure handling.

**Refinement termination:**
```python
if critique.needs_revision and critique.revised_description:
    current_description = critique.revised_description  # continue loop
else:
    break  # early exit -- image is publication-ready
```

### Evaluation System

The evaluation system (`paperbanana/evaluation/judge.py`) implements **VLM-as-Judge
referenced comparison** from paper Section 4.2. This is separate from the generation
loop (the Critic fills that role during generation).

**Four dimensions:**

| Dimension | Primary? | Key criterion |
|-----------|----------|---------------|
| **Faithfulness** | Yes | Technical alignment; veto on hallucination, contradiction, scope violation |
| **Readability** | Yes | Visual flow/legibility; veto on noise, occlusion, illegible fonts |
| **Conciseness** | No | Signal-to-noise; veto on textual overload (>15 words in boxes), math dump |
| **Aesthetics** | No | Visual polish; veto on artifacts, neon colors, black background |

Each dimension returns: `Model | Human | Both are good | Both are bad`.

**Hierarchical aggregation:**
1. Aggregate primary pair (Faithfulness + Readability): if both agree -> that side
   wins; if one wins + one ties -> winner wins; otherwise inconclusive.
2. If primary inconclusive, aggregate secondary pair (Conciseness + Aesthetics) same
   way.
3. If still inconclusive -> "Both are good" (default tie).

**Scoring:** Model wins = 100.0, Human wins = 0.0, Tie = 50.0.
**Judge model:** `gpt-4o` at **temperature 0.1**, max 1024 tokens.

### Reference Data

**Curated reference set** (`data/reference_sets/index.json`): 13 verified methodology
diagrams from recent arXiv papers in 4 categories:

| Category | Count | Examples |
|----------|-------|----------|
| `agent_reasoning` | 5 | GlimpRouter, SEEM, Dr. Zero, MAXS |
| `generative_learning` | 5 | ReasonMark, X-Coder, Codified Foreshadow-Payoff, Flexibility Trap, Stable-DiffCoder |
| `vision_perception` | 2 | Fast-ThinkAct, HERMES |
| `science_applications` | 1 | StructMAE |

**Methodology style guide** (176 lines, NeurIPS 2025 aesthetic standards): soft pastels,
10-15% opacity backgrounds, rounded rectangles (80% dominant), 3D cuboids for tensors,
orthogonal arrows for architectures, domain-specific sub-styles.

**Plot style guide** (149 lines): white backgrounds, Viridis/Magma for sequential,
type-specific rules for bar/line/pie/scatter/heatmap charts.

### Provider Architecture

Factory pattern via `ProviderRegistry` with two provider types:

| Provider Type | Implementations | API Key |
|---------------|----------------|---------|
| **VLM** | `gemini`, `openrouter` | `GOOGLE_API_KEY` or `OPENROUTER_API_KEY` |
| **Image Gen** | `google_imagen`, `openrouter_imagen` | Same |

**Agent temperature settings:**

| Agent | Temperature | Max Tokens | Response Format |
|-------|------------|------------|-----------------|
| Retriever | 0.3 | default | JSON |
| Planner | 0.7 | 4096 | text |
| Stylist | 0.5 | 4096 | text |
| Visualizer (diagram) | N/A (image gen) | N/A | image (1792x1024) |
| Visualizer (plot) | 0.3 | 4096 | text (code) |
| Critic | 0.3 | 4096 | JSON |
| Evaluation Judge | 0.1 | 1024 | JSON |

Rationale: creative agents (Planner, Stylist) run warmer; selection and evaluation
agents (Retriever, Critic, Judge) run cold for consistency.

### Interfaces

**CLI** (Typer): `paperbanana generate`, `paperbanana plot`, `paperbanana evaluate`,
`paperbanana setup`.

**Python API:**
```python
pipeline = PaperBananaPipeline(settings=Settings(...))
result = await pipeline.generate(GenerationInput(
    source_context="...",
    communicative_intent="...",
    diagram_type=DiagramType.METHODOLOGY,
))
```

**MCP Server** (`mcp_server/`): 3 tools via FastMCP (`generate_diagram`,
`generate_plot`, `evaluate_diagram`). Installable via `uvx --from paperbanana[mcp]
paperbanana-mcp`.

**Claude Code Skills:** `/generate-diagram`, `/generate-plot`, `/evaluate-diagram`.

**Dependencies:** `pydantic>=2.0`, `google-genai>=1.0`, `pillow>=10.0`, `typer>=0.12`,
`matplotlib>=3.8`, `structlog>=24.0`, `tenacity>=8.0`. Optional: `fastmcp>=2.0` (MCP),
`pymupdf>=1.24` (PDF).

## Options Considered

### Option A: Adopt PaperBanana's multi-agent pattern for SVG optimization

**Description:** Apply the same Planner->Stylist->Visualizer->Critic loop to SVG
generation, using the textual-description-as-blueprint pattern.

**Pros:**
- Proven iterative refinement approach with built-in quality gate (Critic)
- Description-as-blueprint decouples semantic intent from rendering
- Temperature tuning per agent role is a sound practice we could replicate

**Cons:**
- PaperBanana generates raster images (PNG); our SVG pipeline is fundamentally
  different (structured XML, not pixel generation)
- Multiple VLM calls per iteration is expensive; SVG optimization may benefit from
  more direct manipulation (gradient-based or rule-based)
- The Critic pattern assumes a VLM can visually evaluate the output, which works for
  images but may not be the right feedback signal for SVG structure

### Option B: Cherry-pick specific patterns (evaluation system, prompt engineering)

**Description:** Adopt the VLM-as-Judge hierarchical evaluation and prompt engineering
patterns, but use our own rendering and optimization approach.

**Pros:**
- The 4-dimension evaluation with hierarchical aggregation is well-designed and
  could serve as a quality benchmark for our SVG outputs
- Prompt engineering patterns (natural-language color descriptions, 7-dimension
  coverage) are directly applicable
- Lower cost: evaluate once at the end rather than on every iteration

**Cons:**
- Doesn't give us the iterative refinement benefit
- VLM-as-Judge still requires image rendering for evaluation

### Option C: Use PaperBanana as a complementary tool (not a pattern to replicate)

**Description:** Treat PaperBanana as a separate tool for raster diagram generation,
keep our SVG pipeline independent. Reference it as prior art but don't adopt its
patterns.

**Pros:**
- Simplest path; no architectural changes needed
- PaperBanana's strength is raster methodology diagrams, which is orthogonal to
  SVG optimization

**Cons:**
- Misses the opportunity to learn from their multi-agent orchestration patterns

## Recommendations

Based on the findings, we recommend **Option B: Cherry-pick specific patterns**.

Key takeaways to apply to our SVG optimization work:

1. **Hierarchical evaluation dimensions.** PaperBanana's 4-dimension evaluation
   (Faithfulness, Readability, Conciseness, Aesthetics) with primary/secondary
   weighting is a well-structured quality framework. We should define analogous
   dimensions for SVG quality (e.g., semantic accuracy, visual fidelity, file size,
   accessibility).

2. **Temperature stratification by agent role.** The pattern of running creative
   generation at higher temperatures (0.5-0.7) and evaluation/selection at lower
   temperatures (0.1-0.3) is sound and applicable to any multi-step LLM pipeline.

3. **Description-as-blueprint pattern.** While we won't replicate the full pipeline,
   the idea of producing a structured intermediate representation before rendering is
   powerful. For SVG, this could mean generating a semantic description of the desired
   optimization before applying transforms.

4. **Conservative failure handling.** The Critic's "no revision on parse failure"
   approach prevents runaway loops. Any iterative refinement we build should have
   similar circuit breakers.

5. **Prompt engineering for visual outputs.** The distinction between natural-language
   color descriptions (for image generation) vs. precise specifications (for code
   generation) is a useful heuristic for designing prompts in different rendering
   contexts.

## Next Steps

- [ ] Define SVG quality dimensions analogous to PaperBanana's 4-dimension evaluation
- [ ] Prototype a VLM-based SVG quality evaluator using the hierarchical aggregation
      pattern
- [ ] Consider whether an iterative refinement loop (with circuit breaker) would
      improve our SVG optimization results

## References

- [PaperBanana GitHub](https://github.com/llmsresearch/paperbanana) -- source code
  (v0.1.2, MIT license)
- arXiv:2601.23265, Zhu et al. -- original research paper
- Local clone: `attic/paperbanana/`
