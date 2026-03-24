"""
Phase 2 — Incremental Structural Parsing.

Performs page-by-page layout detection, hierarchical structuring,
and cross-page continuity analysis using a trailing-state mechanism.

Capabilities:
    - Bounding-box detection for every layout element.
    - Global node_id continuity across all pages.
    - Parent-child hierarchy inference (via active_parent_stack).
    - Cross-page text continuation detection (continues_previous_node).
    - New-category discovery: if the model encounters an element that
      does NOT fit any Phase 1 category, it flags a suggestion so the
      pipeline can evolve the category vocabulary over time.

How it works:
    1. Load Phase 1 categories.
    2. For each page (sequentially):
       a. Send previous page image (visual context) + current page image
          + trailing state + category definitions to the MLLM.
       b. MLLM returns detected elements, trailing state for next page,
          and optionally suggested new categories.
       c. Pipeline validates, collects, and feeds the state forward.
    3. Merge all per-page results into a document-level JSON.
    4. Report any suggested new categories for human review.
    5. Save visualisations.

Usage:
    python phase2_structural_parsing.py                     # all pages
    python phase2_structural_parsing.py --start 1 --end 10  # pages 1-10
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image

from config import (
    IMAGE_DIR,
    MODEL_NAME,
    OUTPUT_DIR,
    PHASE2_MAX_TOKENS,
    TEMPERATURE,
)
from llm_client import chat_completion, make_image_content, make_text_content
from visualize import draw_elements_on_page

# =====================================================================
#  Prompt template
# =====================================================================

PHASE2_SYSTEM_PROMPT = r"""# [Role]
You are an Industrial-Grade Visual Spatial Analyzer and Structural Logic Agent operating within a Sequential Document Parsing Pipeline.

# [Objective]
Analyze the CURRENT page image, locate every instance of the defined layout categories, assign globally sequential node IDs, and infer logical & hierarchical relationships — using the Trailing State from the previous page to maintain cross-page continuity.

# [Layout Categories]
Locate and classify elements strictly into one of the following categories:

{category_definitions}

You MUST use these categories for all recognized elements.

**However**, if you encounter a visually distinct element that genuinely does NOT fit ANY of the above categories, you MUST still detect it (assign a node_id and box_2d) using the closest existing category, AND report it in the `suggested_new_categories` section of your output. This ensures no element is ever skipped.

# [Rules & Constraints]

## 1. Physical Bounding Boxes (`box_2d`)
- Format: `[ymin, xmin, ymax, xmax]`.
- Normalized to a **1000×1000 grid** for the CURRENT page image only.
- Boxes must be **tight** around each element.
- Do NOT skip any visible element. Each visually distinct block gets its own node.

## 2. Global ID Sequencing (`node_id`)
- Resume counting from `last_assigned_node_id` + 1.
  (e.g., if last was "node_045", start with "node_046".)
- Follow **reading order**: top→bottom, left→right within the page.

## 3. Hierarchical Parenting (`parent_node_id`)
- If an element at the top of this page logically belongs to an open section from the previous page, set `parent_node_id` to the matching entry in `active_parent_stack`.
- For elements with a parent on the current page, use that parent's node_id.
- Top-level elements (e.g., Header_Footer) → `null`.

## 4. Cross-Page Continuation (`continues_previous_node`)
- If the **very first content element** on this page is a direct continuation of text/list cut off at the bottom of the previous page (see `element_cut_off_at_bottom` in the Trailing State), set this field to that node's ID.
- Otherwise → `null`.
- This flag tells the downstream pipeline to concatenate the two nodes' content.

## 5. Content Snippet (`content_snippet`)
- First 100 characters only — for grounding/verification, NOT full extraction.
- For Tables/Figures with little text, give a brief visual description.

## 6. Producing the Trailing State for the Next Page
At the end of your response, you MUST output a `trailing_state_for_next` object:
- `last_assigned_node_id`: the highest node_id you assigned on this page.
- `active_parent_stack`: a list (outermost→innermost) of section headings that remain "open" at the bottom of this page. Include `node_id`, `category`, and `snippet` for each. This should reflect the **hierarchical nesting** of the document (e.g., [Section 1, Subsection 1.1, Sub-sub 1.1.1]).
- `element_cut_off_at_bottom`: if the last element on this page appears to be truncated (text stops mid-sentence, list continues, table rows cut), provide its `node_id`, `category`, and `snippet`. Otherwise → `null`.

## 7. New Category Suggestions (`suggested_new_categories`)
- If you detect an element that does NOT fit well into any defined category, still classify it with the closest match AND add an entry to `suggested_new_categories`.
- Each suggestion must include: `proposed_class_name`, `description`, `encountered_on_node_id`, and `reason` (why existing categories are insufficient).
- If all elements fit well → output an empty list `[]`.

# [Expected JSON Output — respond with ONLY this JSON, nothing else]
```
{{
  "page_metadata": {{
    "current_image_index": "integer",
    "total_nodes_detected_on_page": "integer"
  }},
  "detected_elements": [
    {{
      "node_id": "string",
      "image_index": "integer",
      "category": "string",
      "box_2d": [ymin, xmin, ymax, xmax],
      "content_snippet": "string (max 100 chars)",
      "parent_node_id": "string | null",
      "continues_previous_node": "string | null",
      "reasoning": "Brief 1-sentence justification."
    }}
  ],
  "trailing_state_for_next": {{
    "last_assigned_node_id": "string",
    "active_parent_stack": [
      {{
        "node_id": "string",
        "category": "string",
        "snippet": "string"
      }}
    ],
    "element_cut_off_at_bottom": {{
      "node_id": "string",
      "category": "string",
      "snippet": "string"
    }} or null
  }},
  "suggested_new_categories": [
    {{
      "proposed_class_name": "string",
      "description": "string",
      "encountered_on_node_id": "string",
      "reason": "string (why existing categories are insufficient)"
    }}
  ]
}}
```"""


# =====================================================================
#  Trailing state management
# =====================================================================

def make_initial_trailing_state() -> dict:
    """Create the trailing state for the very first page (no history)."""
    return {
        "previous_page_index": None,
        "last_assigned_node_id": "node_000",
        "active_parent_stack": [],
        "element_cut_off_at_bottom": None,
    }


def extract_trailing_state(result: dict, page_index: int) -> dict:
    """
    Extract the trailing state from a page's MLLM output.

    The model is asked to produce `trailing_state_for_next`.
    If it's missing or malformed, fall back to heuristics.
    """
    ts = result.get("trailing_state_for_next", {})

    # Validate / fill defaults.
    last_id = ts.get("last_assigned_node_id")
    if not last_id:
        # Fallback: find the highest node_id in detected_elements.
        elements = result.get("detected_elements", [])
        if elements:
            last_id = elements[-1].get("node_id", "node_000")
        else:
            last_id = "node_000"

    parent_stack = ts.get("active_parent_stack", [])
    cut_off = ts.get("element_cut_off_at_bottom")

    return {
        "previous_page_index": page_index,
        "last_assigned_node_id": last_id,
        "active_parent_stack": parent_stack,
        "element_cut_off_at_bottom": cut_off,
    }


def validate_node_ids(
    elements: list[dict],
    expected_start: str,
) -> list[str]:
    """
    Check that node_ids start from expected_start and are sequential.
    Returns a list of warnings (empty if all good).
    """
    warnings: list[str] = []
    if not elements:
        return warnings

    # Parse expected start number.
    match = re.search(r"(\d+)", expected_start)
    expected_num = int(match.group(1)) + 1 if match else 1

    first_id = elements[0].get("node_id", "")
    first_match = re.search(r"(\d+)", first_id)
    if first_match:
        actual_num = int(first_match.group(1))
        if actual_num != expected_num:
            warnings.append(
                f"ID mismatch: expected node_{expected_num:03d}, "
                f"got {first_id}"
            )

    return warnings


# =====================================================================
#  Helpers
# =====================================================================

def load_phase1_categories(path: Path) -> list[dict]:
    """Load the merged category list from Phase 1 output."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    categories = data.get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in {path}")
    return categories


def format_category_definitions(categories: list[dict]) -> str:
    """Format categories for prompt injection."""
    lines: list[str] = []
    for i, cat in enumerate(categories, 1):
        name = cat["class_name"]
        desc = cat.get("description", "")
        purpose = cat.get("downstream_purpose", "")
        lines.append(f"- `{name}`: {desc}")
        if purpose:
            lines.append(f"  Purpose: {purpose}")
    return "\n".join(lines)


def load_page_images(
    image_dir: Path,
    start: int | None = None,
    end: int | None = None,
) -> list[tuple[int, Image.Image]]:
    """Load page images sorted by filename. Returns (page_number, image)."""
    paths = sorted(image_dir.glob("page_*.png"))
    if not paths:
        raise FileNotFoundError(f"No page_*.png found in {image_dir}")

    pages: list[tuple[int, Image.Image]] = []
    for p in paths:
        num = int(re.search(r"(\d+)", p.stem).group(1))
        if start and num < start:
            continue
        if end and num > end:
            continue
        pages.append((num, Image.open(p).convert("RGB")))

    print(f"Loaded {len(pages)} pages from {image_dir}")
    return pages


def parse_json_response(raw: str) -> dict:
    """Extract the first JSON object from model response."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError(f"Could not extract JSON:\n{raw[:500]}")


# =====================================================================
#  Per-page analysis
# =====================================================================

def build_page_message(
    prev_page: tuple[int, Image.Image] | None,
    current_page: tuple[int, Image.Image],
    current_index: int,
    trailing_state: dict,
    category_text: str,
) -> list[dict]:
    """
    Build the chat message for one page.

    Sends up to 2 images:
      - Previous page (visual context, if available).
      - Current page (analysis target).
    Plus the trailing state and the system prompt with categories.
    """
    content: list[dict] = []

    # Previous page as visual context.
    if prev_page is not None:
        prev_num, prev_img = prev_page
        content.append(make_text_content(
            f"[CONTEXT — Previous Page {prev_num} — for visual reference ONLY, "
            f"do NOT detect elements on this page]"
        ))
        content.append(make_image_content(prev_img))

    # Current page = analysis target.
    cur_num, cur_img = current_page
    content.append(make_text_content(
        f"[TARGET — Current Page {cur_num} — Analyze THIS page]"
    ))
    content.append(make_image_content(cur_img))

    # Trailing state.
    state_json = json.dumps(trailing_state, indent=2, ensure_ascii=False)
    content.append(make_text_content(
        f"# [Trailing State from Previous Page]\n```json\n{state_json}\n```"
    ))

    # Prompt with categories.
    prompt = PHASE2_SYSTEM_PROMPT.format(
        category_definitions=category_text,
    )
    content.append(make_text_content(prompt))

    return [{"role": "user", "content": content}]


def analyze_page(
    prev_page: tuple[int, Image.Image] | None,
    current_page: tuple[int, Image.Image],
    current_index: int,
    trailing_state: dict,
    category_text: str,
) -> dict:
    """Analyze a single page and return detected elements + new trailing state."""
    cur_num = current_page[0]
    prev_label = f"prev={prev_page[0]}" if prev_page else "no prev"
    print(
        f"  Page {cur_num} (index {current_index}, {prev_label}) ... ",
        end="", flush=True,
    )

    messages = build_page_message(
        prev_page, current_page, current_index, trailing_state, category_text,
    )

    t0 = time.time()
    raw = chat_completion(
        messages,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=PHASE2_MAX_TOKENS,
    )
    elapsed = time.time() - t0

    result = parse_json_response(raw)

    # Validate node IDs.
    elements = result.get("detected_elements", [])
    warnings = validate_node_ids(elements, trailing_state["last_assigned_node_id"])
    n_elems = len(elements)

    status = f"done ({elapsed:.1f}s, {n_elems} elements)"
    if warnings:
        status += f"  ⚠ {'; '.join(warnings)}"
    print(status)

    return result


# =====================================================================
#  Merge all pages
# =====================================================================

def merge_all_pages(per_page_results: list[dict]) -> dict:
    """Combine per-page results into a single document-level output."""
    all_elements: list[dict] = []
    continuations: list[dict] = []
    all_new_category_suggestions: list[dict] = []

    for pr in per_page_results:
        for elem in pr.get("detected_elements", []):
            all_elements.append(elem)
            if elem.get("continues_previous_node"):
                continuations.append({
                    "node_id": elem["node_id"],
                    "continues": elem["continues_previous_node"],
                    "page": elem.get("image_index"),
                })

        # Collect new category suggestions.
        for suggestion in pr.get("suggested_new_categories", []):
            all_new_category_suggestions.append(suggestion)

    # Deduplicate suggestions by proposed_class_name.
    seen_names: set[str] = set()
    unique_suggestions: list[dict] = []
    for s in all_new_category_suggestions:
        name = s.get("proposed_class_name", "").strip().lower()
        if name and name not in seen_names:
            seen_names.add(name)
            unique_suggestions.append(s)

    pages_analyzed = len(per_page_results)
    return {
        "document_summary": {
            "total_pages_analyzed": pages_analyzed,
            "total_nodes_detected": len(all_elements),
            "cross_page_continuations": continuations,
            "suggested_new_categories": unique_suggestions,
        },
        "detected_elements": all_elements,
    }


# =====================================================================
#  Visualisation
# =====================================================================

def save_visualisations(
    merged: dict,
    pages: list[tuple[int, Image.Image]],
    out_dir: Path,
) -> None:
    """Draw bounding boxes on each page and save as JPG."""
    page_map = {pn: img for pn, img in pages}

    by_page: dict[int, list[dict]] = {}
    for elem in merged["detected_elements"]:
        pn = elem.get("page_number", elem.get("image_index"))
        by_page.setdefault(pn, []).append(elem)

    for pn, img in sorted(page_map.items()):
        elements = by_page.get(pn, [])
        vis = draw_elements_on_page(img, elements, expand_px=8)
        out_path = out_dir / f"phase2_page_{pn:03d}.jpg"
        vis.save(out_path, "JPEG", quality=92)
        print(f"    {out_path.name}: {len(elements)} elements")


# =====================================================================
#  Main
# =====================================================================

def run(
    image_dir: Path = IMAGE_DIR,
    phase1_path: Path | None = None,
    start: int | None = None,
    end: int | None = None,
) -> dict:
    """Execute Phase 2 incremental analysis."""
    if phase1_path is None:
        phase1_path = OUTPUT_DIR / "phase1_categories.json"

    # Load inputs.
    categories = load_phase1_categories(phase1_path)
    category_text = format_category_definitions(categories)
    print(f"Phase 1 categories: {len(categories)} classes")
    for cat in categories:
        print(f"  - {cat['class_name']}")

    pages = load_page_images(image_dir, start=start, end=end)
    if not pages:
        raise ValueError("No pages to process.")

    print(f"\nPhase 2 (Structural Parsing): {len(pages)} pages, incremental mode")
    print(f"Model: {MODEL_NAME}\n")

    # Sequential page-by-page analysis.
    trailing_state = make_initial_trailing_state()
    per_page_results: list[dict] = []
    all_trailing_states: list[dict] = [trailing_state]

    for i, (page_num, page_img) in enumerate(pages):
        prev_page = pages[i - 1] if i > 0 else None
        current_page = (page_num, page_img)

        result = analyze_page(
            prev_page=prev_page,
            current_page=current_page,
            current_index=i,
            trailing_state=trailing_state,
            category_text=category_text,
        )

        # Stamp actual page_number on each element.
        for elem in result.get("detected_elements", []):
            elem["page_number"] = page_num

        per_page_results.append(result)

        # Extract trailing state for next page.
        trailing_state = extract_trailing_state(result, page_index=i)
        all_trailing_states.append(trailing_state)

    # Merge.
    merged = merge_all_pages(per_page_results)

    # Save outputs.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUTPUT_DIR / "phase2_incremental.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    states_path = OUTPUT_DIR / "phase2_trailing_states.json"
    with open(states_path, "w", encoding="utf-8") as f:
        json.dump(all_trailing_states, f, indent=2, ensure_ascii=False)

    # Visualise.
    print("\nVisualising...")
    vis_dir = OUTPUT_DIR / "phase2_incremental_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_visualisations(merged, pages, vis_dir)

    # Summary.
    summary = merged["document_summary"]
    print(f"\nPhase 2 (Structural Parsing) complete.")
    print(f"  Pages:          {summary['total_pages_analyzed']}")
    print(f"  Total nodes:    {summary['total_nodes_detected']}")
    print(f"  Continuations:  {len(summary['cross_page_continuations'])}")
    print(f"  New cat. suggestions: {len(summary['suggested_new_categories'])}")
    print(f"  Output:         {out_path}")
    print(f"  Trailing states:{states_path}")

    if summary["cross_page_continuations"]:
        print(f"\n  Cross-page continuations:")
        for c in summary["cross_page_continuations"]:
            print(f"    {c['node_id']} continues {c['continues']}")

    if summary["suggested_new_categories"]:
        print(f"\n  Suggested new categories (for human review):")
        for s in summary["suggested_new_categories"]:
            print(f"    * {s['proposed_class_name']}: {s.get('reason', '')[:80]}")

    cat_counts: dict[str, int] = {}
    for elem in merged["detected_elements"]:
        c = elem.get("category", "unknown")
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print(f"\n  Category distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:25s} {count:4d}")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 (Incremental): page-by-page layout analysis with trailing state."
    )
    parser.add_argument(
        "--image-dir", type=Path, default=IMAGE_DIR,
        help="Directory containing page_*.png files.",
    )
    parser.add_argument(
        "--phase1", type=Path, default=None,
        help="Path to Phase 1 output JSON.",
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="First page number (1-indexed, inclusive).",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="Last page number (1-indexed, inclusive).",
    )

    args = parser.parse_args()
    run(
        image_dir=args.image_dir,
        phase1_path=args.phase1,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
