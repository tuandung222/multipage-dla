# Multi-Page Document Layout Analysis Pipeline

A **multi-phase, MLLM-powered pipeline** for analyzing complex multi-page documents (ISO standards, quality manuals, government procedures). The system decomposes the hard problem of document layout analysis into focused phases, each with a specialized prompt — producing significantly better results than end-to-end single-prompt approaches.

---

## 1. Problem Formulation

### 1.1 Traditional DLA (Single-Page)

Traditional Document Layout Analysis learns a mapping from a single document image to a set of layout entities:

$$\mathcal{F}(\mathcal{I}) \rightarrow \mathcal{Y} = \bigl\{ (b_i,\; c_i) \bigr\}_{i=1}^{N}$$

where:
- $b_i = [x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ — **Bounding box** coordinates
- $c_i \in \mathcal{C}$ — **Category label** (Title, Text, Table, Figure, Formula, ...)

This formulation treats each page independently and ignores inter-element relationships.

### 1.2 Extended Formulation: Graph-Based DLA

Modern approaches extend DLA to capture **structural relationships** between elements, formulated as a document graph:

$$\mathcal{Y} = (\mathcal{V},\; \mathcal{E})$$

where $\mathcal{V}$ is the set of layout regions (nodes) and $\mathcal{E}$ is the set of relationships (edges) encoding reading order, hierarchical structure (parent-child), and logical grouping.

### 1.3 Our Extension: Cross-Page Multi-Page DLA

We extend the problem further to **multi-page documents** where analyzing a single page requires context from adjacent pages:

$$\mathcal{F}\bigl(\mathcal{I}_t \;\big|\; \mathcal{I}_{t-1},\; \mathcal{S}_{t-1}\bigr) \;\rightarrow\; \bigl(\mathcal{Y}_t,\; \mathcal{S}_t\bigr)$$

where:
- $\mathcal{I}_t$ — Current page image
- $\mathcal{I}_{t-1}$ — Previous page image (adjacent context)
- $\mathcal{S}_{t-1}$ — **Trailing state** from all previously analyzed pages (accumulated context)
- $\mathcal{Y}_t = (\mathcal{V}_t,\; \mathcal{E}_t)$ — Detected layout graph for page $t$, including:
  - $\mathcal{V}_t = \bigl\{ (b_i,\; c_i,\; \text{id}_i,\; \text{parent}_i) \bigr\}$ — Nodes with bounding boxes, categories, unique IDs, and parent links
  - $\mathcal{E}_t$ — Edges encoding hierarchy, reading order, and **cross-page continuations**
- $\mathcal{S}_t$ — Updated trailing state passed to page $t+1$

The trailing state $\mathcal{S}_t$ encodes:
- `last_assigned_node_id` — Global ID counter for continuity
- `active_parent_stack` — Hierarchical context (e.g., Section 1 > 1.1 > 1.1.1)
- `element_cut_off_at_bottom` — Detects text/tables split at page boundaries
- `suggested_new_categories` — Evolving category vocabulary

**Key distinction from traditional DLA:**
- A single page **cannot** be fully analyzed in isolation — category assignment and hierarchy depend on context from preceding pages
- The output is not a flat set of boxes but a **hierarchical document tree** with cross-page links
- The category vocabulary is **document-specific** and discovered dynamically, not fixed a priori

### 1.4 Target Domain

This pipeline targets documents with **extremely complex layouts** that are far from well-structured scientific papers or books:

- **ISO standards & quality manuals** — Deep nested sections, definition lists spanning multiple pages
- **Financial reports** — Multi-page tables, footnotes, cross-references
- **Government procedures** — Mixed visual styles, regulatory formatting, nested numbered lists
- **Technical specifications** — Diagrams interspersed with text, appendices, revision tables

Existing benchmarks (PubLayNet, DocLayNet, DocBank) focus primarily on scientific papers and simple documents — they do not cover these complex, multi-page, cross-referencing layouts.

---

## 2. Motivation & Research Gap

### 2.1 Why Cross-Page Context Matters

Real-world industrial documents have:

- **Cross-page structures**: A section heading on page 5 governs paragraphs on pages 5-8.
- **Page-boundary artifacts**: Text cut mid-sentence, tables split across pages, lists that span multiple pages.
- **Context-dependent categories**: Whether a block is a "definition list" or a "numbered procedure" often depends on the section header 3 pages earlier.

### 2.2 Challenges with Existing Approaches

**Vision-based detection models** (YOLO, RT-DETR, DocLayout-YOLO):
- High bounding box accuracy and fast inference on trained categories
- **Cannot** handle categories outside their training set
- **Cannot** reason about cross-page context or dynamic category assignment

**SOTA MLLM in end-to-end mode** (including 235B-parameter reasoning models):
- Strong semantic reasoning capability
- **Grounding accuracy degrades** as the number of elements increases — the more objects requested, the more coordinates drift
- When asked to do everything in one prompt (discover categories + localize + build hierarchy), they either **over-fragment** or **under-detect**

**SOTA document parsing pipelines** (PaddleOCR, MinerU):
- Work well for simple, single-page documents
- **Fail** on complex multi-page layouts with nested hierarchies and cross-page continuations

### 2.3 Benchmark & Training Data Gap

- No existing benchmark covers multi-page, cross-page DLA with hierarchical annotations
- Creating ground truth requires **expensive human annotation**
- This pipeline serves as an **automated labeling tool** to generate high-quality annotations

### 2.4 What We Tried Before This Pipeline

We extensively experimented with existing approaches:

- **PaddleOCR, MinerU**: Fail on complex multi-page layouts — no cross-page awareness, no hierarchical output
- **Qwen3-VL-235B-Thinking, Gemini family, and other large MLLMs**: Even 235B-parameter reasoning models produce poor results in end-to-end mode. The models have the capability, but the **task formulation** was wrong.

The breakthrough came from **decomposing the problem**: the same MLLM that failed in end-to-end mode produces excellent results when given a narrower, well-defined task with the right context.

---

## 3. Pipeline Architecture

```mermaid
flowchart TB
    INPUT["INPUT: PDF Document\nRendered to page images at 220 DPI"]

    subgraph P1 [" Phase 1 : Category Discovery "]
        direction TB
        P1A["Sliding window of k pages\nsent to SOTA MLLM"]
        P1B["MLLM reasons about\nmacro-level layout categories"]
        P1C["Output: consolidated, deduplicated\ncategory vocabulary"]
        P1A --> P1B --> P1C
    end

    subgraph P2 [" Phase 2 : Structural Parsing "]
        direction TB
        subgraph P2_1 [" Phase 2.1 : Initial Analysis "]
            P21A["Analyze first page\nwith category definitions"]
            P21B["Establish initial trailing state:\n- last_assigned_node_id\n- active_parent_stack\n- element_cut_off_at_bottom"]
            P21A --> P21B
        end
        subgraph P2_2 [" Phase 2.2 : Incremental Loop "]
            P22A["For each subsequent page:"]
            P22B["Input: prev_page + current_page\n+ trailing state + categories"]
            P22C["Output: bounding boxes,\nnode IDs, parent-child links"]
            P22D["Detect cross-page continuations\nMaintain hierarchy across pages\nFlag new categories if needed"]
            P22E["Pass updated trailing state\nto next iteration"]
            P22A --> P22B --> P22C --> P22D --> P22E
            P22E -.->|"loop until\nlast page"| P22A
        end
        P2_1 --> P2_2
    end

    subgraph P3 [" Phase 3 : Document Tree Assembly "]
        direction TB
        P3A["Merge continuation nodes\nacross page boundaries"]
        P3B["Build hierarchical tree\nfrom parent_node_id links"]
        P3C["Export: JSON tree,\ntext outline, Mermaid diagram"]
        P3D["Rule-based, deterministic\nNo MLLM calls needed"]
        P3A --> P3B --> P3C
        P3D
    end

    INPUT -->|"page images"| P1
    P1 -->|"categories.json"| P2
    P2 -->|"phase2_incremental.json"| P3

    style INPUT fill:#E3F2FD,stroke:#1565C0,color:#0D47A1,font-weight:bold
    style P1 fill:#FFF3E0,stroke:#E65100,color:#333
    style P2 fill:#E8F5E9,stroke:#2E7D32,color:#333
    style P2_1 fill:#C8E6C9,stroke:#388E3C,color:#333
    style P2_2 fill:#A5D6A7,stroke:#2E7D32,color:#333
    style P3 fill:#F3E5F5,stroke:#6A1B9A,color:#333
```

**Scripts:** `phase1_category_discovery.py` | `phase2_structural_parsing.py` | `phase3_tree_assembly.py`

---

## 4. Algorithm (Pseudocode)

```
Algorithm: Multi-Page Document Layout Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:  D = {I_1, I_2, ..., I_n}    // n page images from PDF (rendered at 220 DPI)
Output: T = Document Tree             // hierarchical tree with bounding boxes, categories, relations

═══════════════════════════════════════
  PHASE 1: Category Discovery
═══════════════════════════════════════

  categories ← ∅

  for w = 1 to ⌈n/k⌉ do                           // sliding window, k pages per window
      window ← {I_{(w-1)k+1}, ..., I_{min(wk, n)}}
      cats_w ← MLLM(window, CATEGORY_DISCOVERY_PROMPT)
      categories ← MERGE_AND_DEDUPLICATE(categories, cats_w)
  end for

  // categories = {(class_name, description, downstream_purpose), ...}

═══════════════════════════════════════
  PHASE 2.1: Initial Analysis
═══════════════════════════════════════

  elements   ← []
  state_0    ← INIT_TRAILING_STATE()               // {node_id_counter: 0, parent_stack: [], cut_off: null}

  (Y_1, S_1) ← MLLM(I_1, null, state_0, categories, STRUCTURAL_PARSING_PROMPT)
  elements   ← elements ∪ Y_1

═══════════════════════════════════════
  PHASE 2.2: Incremental Loop
═══════════════════════════════════════

  for t = 2 to n do
      (Y_t, S_t) ← MLLM(I_t, I_{t-1}, S_{t-1}, categories, STRUCTURAL_PARSING_PROMPT)

      // Y_t contains for each detected element:
      //   - node_id (globally unique, continuing from S_{t-1})
      //   - category ∈ categories
      //   - box_2d = [ymin, xmin, ymax, xmax] normalized to 1000×1000
      //   - parent_node_id (hierarchy link, possibly to node on earlier page)
      //   - continues_previous_node (cross-page continuation link)
      //   - content_snippet

      elements ← elements ∪ Y_t

      // S_t carries forward:
      //   - last_assigned_node_id
      //   - active_parent_stack (e.g., [Section 4, Subsection 4.1])
      //   - element_cut_off_at_bottom
      //   - suggested_new_categories → merged into categories if valid
  end for

═══════════════════════════════════════
  PHASE 3: Document Tree Assembly
  (Rule-based, no MLLM)
═══════════════════════════════════════

  // Step 3a: Merge continuation nodes
  for each element e where e.continues_previous_node ≠ null do
      target ← RESOLVE_CHAIN(e.continues_previous_node)
      target.snippet ← target.snippet + " " + e.snippet
      REMAP_REFERENCES(e.node_id → target.node_id)
      REMOVE(e)
  end for

  // Step 3b: Build hierarchical tree
  T ← EMPTY_FOREST()
  for each element e do
      if e.parent_node_id = null then
          T.add_root(e)
      else if e.parent_node_id ∈ elements then
          PARENT(e.parent_node_id).add_child(e)
      else
          T.add_root(e)        // orphan → attach to root
      end if
  end for

  // Step 3c: Export
  EXPORT_JSON(T)                // phase3_document_tree.json
  EXPORT_OUTLINE(T)             // phase3_outline.txt
  EXPORT_MERMAID(T)             // phase3_mermaid.md

  return T
```

---

## 5. Baseline Experiment: E2E vs Multi-Phase

We ran both approaches on the same 3 pages of an ISO 9001 Quality Manual using the same SOTA MLLM (temperature=0):

| Metric | Single-Page E2E | Multi-Page E2E (3p) | **Our Pipeline** |
|---|---|---|---|
| Total elements detected | 161 | 26 | **21** |
| Categories discovered | 10 | 5 | **13** |
| Valid bounding boxes | 161 | 26 | **21** |

**Key findings:**

- **Single-page E2E** over-fragments massively (161 elements for 3 pages — it detects individual `table_cell` and `table_row` instead of whole tables). Category vocabulary is ad-hoc and inconsistent across pages.
- **Multi-page E2E** under-detects (26 elements) with only 5 generic categories — loses fine-grained structure.
- **Our pipeline** produces the right granularity (21 elements) with 13 pragmatic, consistent categories that have clear downstream purpose.

Visual comparison available in `outputs/baseline_experiment/`.

---

## 6. Example Documents

### Document 1: ISO 9001:2015 Quality Manual

- **Pages analyzed**: 11 (of 42)
- **Results**: `outputs/`
- **Highlights**:
  - 136 nodes detected, 13 categories
  - Cross-page ToC continuation detected (pages 3->4)
  - 3-level hierarchy: Section -> Subsection -> Content blocks
  - Section 3 "Terms and Definitions" correctly grouped 64 definition entries under one parent across 5 pages

### Document 2: Generic Manual on ISO 9001 Six Mandatory Procedures

- **Pages analyzed**: 7 (cover, ToC, introduction, first procedure)
- **Results**: `outputs_iso_generic/`
- **Highlights**:
  - 81 nodes, 12 categories, max depth 3
  - Cross-page list continuation detected
  - Handles mixed visual styles (blue headings, italic guidelines, nested tables)

---

## 7. Research Roadmap

This pipeline is the first step toward a larger research agenda:

### 7.1 Proposed Labeling Pipeline

Combine **vision-based detectors** (SOTA bounding box detection) with **SOTA MLLMs** (category assignment + relation reasoning):
- Detectors provide accurate bounding boxes (what they do best)
- MLLMs rewrite categories and resolve hierarchical relations (what they do best)
- Result: high-quality benchmark + training dataset without expensive human annotation

### 7.2 Planned Experiments

1. **Inference-based approaches**: Test SOTA parsing pipelines (PaddleOCR, MinerU) combined with MLLM reasoning for category/relation assignment
2. **Prompt-based end-to-end MLLM**: Send page image sequences + instructions, output layout analysis directly
3. **Fine-tuning approaches**:
   - *Direction A*: Fine-tune a small VLM to correct categories and resolve relations from detector outputs
   - *Direction B*: Fine-tune an end-to-end VLM that takes PDF image sequences and outputs full layout analysis
4. **Model optimization**: If fine-tuning succeeds, apply compression/distillation for production deployment

### 7.3 Data Goals

- Collect PDFs with complex layouts requiring cross-page context for analysis
- Build automated labeling pipeline using this system
- Target: **2K benchmark samples** (with ground truth) + **20K training samples**

---

## 8. Key Design Decisions

### Phase 1: Why separate category discovery?

Letting the MLLM **see multiple pages first** and **reason about categories** before localizing anything produces a vocabulary that is:
- **Pragmatic**: Macro-level categories (e.g., `List_Block` instead of `bulleted_list_item` + `numbered_list_item`)
- **Downstream-aware**: Each category has a documented purpose (e.g., "Passed to specialized table parser")
- **Consistent**: Same vocabulary applied uniformly across all pages

### Phase 2: Why split into Initial + Incremental Loop?

**Phase 2.1 (Initial Analysis)** establishes the baseline state from the first page — creating the initial node IDs, parent stack, and category assignments. This bootstraps the trailing state that drives all subsequent analysis.

**Phase 2.2 (Incremental Loop)** then processes each remaining page sequentially, carrying forward the **trailing state mechanism**:
- `last_assigned_node_id` -> global ID continuity without post-hoc renumbering
- `active_parent_stack` -> hierarchical parenting across page boundaries
- `element_cut_off_at_bottom` / `continues_previous_node` -> text merge at page breaks
- `suggested_new_categories` -> vocabulary evolution as new page types appear

### Phase 3: Why rule-based?

All information needed for tree construction is already in the Phase 2 JSON (`parent_node_id`, `continues_previous_node`). Using an MLLM here would be:
- **Non-deterministic**: Same input -> different trees
- **Wasteful**: Paying for API calls to do what a `dict` lookup can do
- **Slower**: Network latency for a CPU-instant operation

---

## 9. Project Structure

```
multipage_dla/
├── .env.example              # Template for API key
├── .gitignore
├── requirements.txt
│
├── config.py                 # Centralized configuration
├── llm_client.py             # MLLM API wrapper (model-agnostic)
├── visualize.py              # Shared bounding box visualization
│
├── phase1_category_discovery.py    # Phase 1: Category reasoning
├── phase2_structural_parsing.py    # Phase 2: Incremental structural parsing
├── phase2_window_based.py          # Phase 2 alternative: sliding window
├── phase3_tree_assembly.py         # Phase 3: Rule-based tree construction
├── baseline_e2e_experiment.py      # Baseline comparison experiment
│
├── example_images/                 # Quality Manual pages (42 pages)
├── iso_generic_pages/              # ISO Generic Manual pages (7 pages)
│
├── outputs/                        # Quality Manual results
│   ├── phase1_categories.json
│   ├── phase2_incremental.json
│   ├── phase2_trailing_states.json
│   ├── phase2_incremental_vis/     # Visualized bounding boxes per page
│   ├── phase3_document_tree.json
│   ├── phase3_outline.txt
│   ├── phase3_mermaid.md           # Mermaid diagram (renders on GitHub)
│   └── baseline_experiment/        # E2E vs pipeline comparison
│
└── outputs_iso_generic/            # ISO Generic Manual results
    ├── phase1_categories.json
    ├── phase2_incremental.json
    ├── phase3_document_tree.json
    ├── phase3_outline.txt
    └── phase3_mermaid.md
```

## 10. Quick Start

### Setup

```bash
cp .env.example .env
# Edit .env and add your API key

pip install -r requirements.txt
```

### Prepare page images

Render your PDF to page images (220 DPI recommended):

```python
import fitz  # PyMuPDF
doc = fitz.open("your_document.pdf")
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=220)
    pix.save(f"your_pages/page_{i+1:03d}.png")
```

### Run the pipeline

```bash
# Phase 1: Discover layout categories
python phase1_category_discovery.py --image-dir your_pages/

# Phase 2: Incremental structural parsing
python phase2_structural_parsing.py --image-dir your_pages/

# Phase 3: Build document tree
python phase3_tree_assembly.py
```

### Read the results

- **`outputs/phase1_categories.json`** — Category vocabulary with descriptions and downstream purpose
- **`outputs/phase2_incremental.json`** — All detected elements with bounding boxes, hierarchy, and cross-page links
- **`outputs/phase2_incremental_vis/`** — Visual inspection: bounding boxes drawn on each page image
- **`outputs/phase3_outline.txt`** — Human-readable document structure at a glance
- **`outputs/phase3_mermaid.md`** — Interactive tree diagram (renders natively on GitHub)
- **`outputs/phase2_trailing_states.json`** — Debug: inspect the trailing state chain across pages

## Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | *(configurable)* | SOTA MLLM model identifier |
| `CONTEXT_WINDOW_SIZE` | `3` | Pages per window in Phase 1 |
| `PHASE2_MAX_TOKENS` | `16384` | Max output tokens for Phase 2 |
| `TEMPERATURE` | `0.0` | Deterministic outputs |

## Use Cases

- **Automated Labeling**: Generate high-quality layout annotations for documents that are too complex for existing tools
- **Benchmark Creation**: Build ground-truth datasets for cross-page DLA evaluation
- **Training Data**: Create bounding-box + category + hierarchy labels for fine-tuning specialized models
- **MLLM-as-a-Judge**: Compare detector outputs against MLLM-generated ground truth
- **Document Understanding**: Feed the structured tree into downstream RAG pipelines

## License

Research use. See individual document sources for their respective licenses.
