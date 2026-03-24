"""
Phase 2 — Physical Layout Analysis (Element Detection & Hierarchy).

Goal:
    Using the category vocabulary discovered in Phase 1, locate every
    layout element in the document with bounding boxes, assign globally
    unique node IDs, and infer parent-child relationships.

How it works:
    1. Load the Phase 1 category list from outputs/phase1_categories.json.
    2. Load page images.
    3. Slide a window of PHASE2_WINDOW_SIZE pages across the document.
    4. For each window, build a prompt that includes:
       - The category definitions (from Phase 1).
       - The page images.
       - Instructions for bounding-box detection + hierarchy inference.
    5. Collect per-window results.
    6. Merge across windows: deduplicate overlapping pages, re-assign
       globally sequential node IDs, and fix parent_node_id references.
    7. Save the consolidated result + per-page visualisations.

Usage:
    python phase2_physical_analysis.py                     # all pages
    python phase2_physical_analysis.py --start 1 --end 6   # pages 1-6
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
    MAX_TOKENS,
    MODEL_NAME,
    OUTPUT_DIR,
    PHASE2_MAX_TOKENS,
    PHASE2_WINDOW_SIZE,
    TEMPERATURE,
)
from llm_client import chat_completion, make_image_content, make_text_content
from visualize import draw_elements_on_page

# =====================================================================
#  Prompt template
# =====================================================================

PHASE2_PROMPT_TEMPLATE = r"""[Role]
You are an expert Visual Spatial Analyzer and Structural Logic Agent for multi-page document processing pipelines.

[Context]
You are provided with a sequence of {n_pages} page images from a multi-page document.
In the previous phase, the following pragmatic layout categories were defined:

{category_definitions}

You MUST use ONLY these categories. Do NOT invent new ones.

[Objective]
1. Analyze the provided image sequence as a single logical document.
2. Locate every distinct instance of the defined categories, drawing bounding boxes.
3. Assign a unique, globally sequential ID (`node_id`) across all pages.
4. Correctly identify which page each element belongs to using an `image_index`.
5. Infer hierarchical relationships (Parent-Child) between elements.

[Spatial, Logic, & Content Constraints]
1. image_index: Zero-based integer index representing the position of the image in the input sequence (e.g., 0 for the first image, 1 for the second, etc.).
2. Coordinates: Output bounding boxes strictly in format [ymin, xmin, ymax, xmax], normalized to a 1000x1000 grid for EACH page.
3. node_id: Unique string ID across the entire multi-page document (e.g., "node_001", "node_002"). Numbering must follow logical reading order across pages.
4. content_snippet LIMITATIONS:
   - Provide only the first 100 characters of text found within the box.
   - If the element is a Table or Figure with limited text, provide a brief, concise visual description (e.g., "Table with 3 columns", "Flowchart showing SWOT").
   - This snippet is ONLY for grounding/verifying the box location, NOT for full extraction.
5. Relationship Inference: Identify Parent-Child relationships based on layout and context (e.g., Lists/Paragraphs belonging to a Section_Heading). Use `parent_node_id`.

[CRITICAL RULES]
- Do NOT skip any visible element on any page.
- Do NOT merge distinct elements into one box. Each visually separate block gets its own node.
- Ensure bounding boxes are TIGHT around the element, not loose.
- For elements that span near the full page width, xmin should be close to the left margin and xmax close to the right margin.
- Reading order: top-to-bottom, left-to-right within each page, then next page.

[Expected JSON Output Format]
{{
  "document_summary": {{
    "total_pages_analyzed": {n_pages},
    "total_nodes_detected": "integer"
  }},
  "detected_elements": [
    {{
      "node_id": "string (global unique)",
      "image_index": "integer (0-based index in THIS image sequence)",
      "category": "string (one of the defined categories)",
      "box_2d": [ymin, xmin, ymax, xmax],
      "content_snippet": "string (STRICTLY LIMITED to first 100 chars or concise visual desc)",
      "parent_node_id": "string | null",
      "reasoning": "string (brief justification for category and parent assignment)"
    }}
  ]
}}

Respond with ONLY the JSON. No additional text."""


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
    """
    Format categories into a readable block for prompt injection.

    Example output:
        1. Header — The block at the top of the page ...
           Purpose: Provides consistent context ...
    """
    lines: list[str] = []
    for i, cat in enumerate(categories, 1):
        name = cat["class_name"]
        desc = cat.get("description", "")
        purpose = cat.get("downstream_purpose", "")
        lines.append(f"{i}. {name} — {desc}")
        if purpose:
            lines.append(f"   Purpose: {purpose}")
    return "\n".join(lines)


def load_page_images(
    image_dir: Path,
    start: int | None = None,
    end: int | None = None,
) -> list[tuple[int, Image.Image]]:
    """Load page images sorted by filename. Returns (page_number, image) tuples."""
    paths = sorted(image_dir.glob("page_*.png"))
    if not paths:
        raise FileNotFoundError(f"No page_*.png files found in {image_dir}")

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
    """Extract the first JSON object from the model response."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError(f"Could not extract JSON:\n{raw[:500]}")


# =====================================================================
#  Window processing
# =====================================================================

def build_phase2_message(
    window: list[tuple[int, Image.Image]],
    category_text: str,
) -> list[dict]:
    """
    Build the chat message for one Phase 2 window.

    Images are sent first (so the model sees them), followed by
    the analysis prompt with category definitions injected.
    """
    content: list[dict] = []

    for idx, (page_num, img) in enumerate(window):
        content.append(make_text_content(
            f"--- Image index {idx} (Document page {page_num}) ---"
        ))
        content.append(make_image_content(img))

    prompt = PHASE2_PROMPT_TEMPLATE.format(
        n_pages=len(window),
        category_definitions=category_text,
    )
    content.append(make_text_content(prompt))

    return [{"role": "user", "content": content}]


def analyze_window(
    window: list[tuple[int, Image.Image]],
    category_text: str,
    window_idx: int,
) -> dict:
    """Send one window to the MLLM and return parsed detection results."""
    page_nums = [p[0] for p in window]
    print(f"  Window {window_idx}: pages {page_nums} ... ", end="", flush=True)

    messages = build_phase2_message(window, category_text)
    t0 = time.time()
    raw = chat_completion(
        messages,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=PHASE2_MAX_TOKENS,
    )
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")

    result = parse_json_response(raw)

    # Attach metadata for merging.
    result["_window_idx"] = window_idx
    result["_window_pages"] = page_nums
    # Map image_index → actual page_number for this window.
    result["_index_to_page"] = {idx: pn for idx, (pn, _) in enumerate(window)}

    return result


# =====================================================================
#  Merging across windows
# =====================================================================

def merge_window_results(
    window_results: list[dict],
) -> dict:
    """
    Merge detected elements from all windows into a single document.

    Strategy:
    - Each window "owns" certain pages (non-overlapping assignment).
      For overlapping pages, the window where the page appeared earlier
      (lower image_index = more central context) is preferred.
    - After collecting owned elements, re-assign sequential node_ids
      and fix parent_node_id references.
    """
    # Step 1: Decide which window owns which page.
    # For each page, pick the window where it has the most context
    # (i.e., is NOT at the boundary). Tie-break: first window wins.
    page_owner: dict[int, int] = {}  # page_num → window_idx

    for wr in window_results:
        widx = wr["_window_idx"]
        pages = wr["_window_pages"]
        for page_num in pages:
            if page_num not in page_owner:
                page_owner[page_num] = widx

    # Step 2: Collect elements from owned pages only.
    all_elements: list[dict] = []

    for wr in window_results:
        widx = wr["_window_idx"]
        idx_to_page = wr["_index_to_page"]

        for elem in wr.get("detected_elements", []):
            img_idx = elem.get("image_index", 0)
            page_num = idx_to_page.get(img_idx, idx_to_page.get(str(img_idx)))

            if page_num is None:
                continue

            # Only keep if this window owns this page.
            if page_owner.get(page_num) != widx:
                continue

            # Store the actual page number on the element.
            elem["page_number"] = page_num
            elem["_original_node_id"] = elem.get("node_id", "")
            all_elements.append(elem)

    # Step 3: Sort by (page_number, reading order within page).
    all_elements.sort(key=lambda e: (
        e["page_number"],
        e.get("box_2d", [0, 0, 0, 0])[0],  # ymin
        e.get("box_2d", [0, 0, 0, 0])[1],  # xmin
    ))

    # Step 4: Re-assign globally sequential node_ids.
    old_to_new: dict[str, str] = {}
    for i, elem in enumerate(all_elements, 1):
        new_id = f"node_{i:03d}"
        old_to_new[elem["_original_node_id"]] = new_id
        elem["node_id"] = new_id

    # Step 5: Fix parent_node_id references.
    for elem in all_elements:
        old_parent = elem.get("parent_node_id")
        if old_parent and old_parent in old_to_new:
            elem["parent_node_id"] = old_to_new[old_parent]
        elif old_parent:
            # Parent was from a different window or not found — set null.
            elem["parent_node_id"] = None

        # Clean up internal fields.
        elem.pop("_original_node_id", None)

    # Build summary.
    pages_analyzed = sorted(page_owner.keys())
    return {
        "document_summary": {
            "total_pages_analyzed": len(pages_analyzed),
            "total_nodes_detected": len(all_elements),
            "page_range": [pages_analyzed[0], pages_analyzed[-1]] if pages_analyzed else [],
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

    # Group elements by page_number.
    by_page: dict[int, list[dict]] = {}
    for elem in merged["detected_elements"]:
        pn = elem["page_number"]
        by_page.setdefault(pn, []).append(elem)

    for pn, img in sorted(page_map.items()):
        elements = by_page.get(pn, [])
        vis = draw_elements_on_page(img, elements)
        out_path = out_dir / f"phase2_page_{pn:03d}.jpg"
        vis.save(out_path, "JPEG", quality=92)
        print(f"  Saved visualisation: {out_path.name} ({len(elements)} elements)")


# =====================================================================
#  Main
# =====================================================================

def run(
    image_dir: Path = IMAGE_DIR,
    phase1_path: Path | None = None,
    start: int | None = None,
    end: int | None = None,
    window_size: int = PHASE2_WINDOW_SIZE,
) -> dict:
    """Execute Phase 2 physical analysis."""
    # Resolve Phase 1 output.
    if phase1_path is None:
        phase1_path = OUTPUT_DIR / "phase1_categories.json"

    # Load inputs.
    categories = load_phase1_categories(phase1_path)
    category_text = format_category_definitions(categories)
    print(f"Phase 1 categories loaded: {len(categories)} classes")
    for cat in categories:
        print(f"  - {cat['class_name']}")

    pages = load_page_images(image_dir, start=start, end=end)
    if not pages:
        raise ValueError("No pages to process.")

    # Build sliding windows.
    windows: list[list[tuple[int, Image.Image]]] = []
    step = max(1, window_size - 1)  # overlap by 1 page
    for i in range(0, len(pages), step):
        window = pages[i : i + window_size]
        windows.append(window)

    print(f"\nPhase 2: {len(windows)} window(s), window_size={window_size}")
    print(f"Model: {MODEL_NAME}\n")

    # Process each window sequentially.
    window_results: list[dict] = []
    for idx, window in enumerate(windows):
        result = analyze_window(window, category_text, idx)
        window_results.append(result)

    # Merge.
    merged = merge_window_results(window_results)

    # Save JSON.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "phase2_layout.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    # Save per-window raw results for debugging.
    raw_path = OUTPUT_DIR / "phase2_raw_windows.json"
    # Strip images from raw results before saving.
    raw_safe = []
    for wr in window_results:
        wr_copy = {k: v for k, v in wr.items() if not k.startswith("_")}
        wr_copy["_window_pages"] = wr["_window_pages"]
        raw_safe.append(wr_copy)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_safe, f, indent=2, ensure_ascii=False)

    # Visualise.
    print("\nSaving visualisations...")
    vis_dir = OUTPUT_DIR / "phase2_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_visualisations(merged, pages, vis_dir)

    # Summary.
    summary = merged["document_summary"]
    print(f"\nPhase 2 complete.")
    print(f"  Pages analysed:  {summary['total_pages_analyzed']}")
    print(f"  Nodes detected:  {summary['total_nodes_detected']}")
    print(f"  Output JSON:     {out_path}")
    print(f"  Visualisations:  {vis_dir}/")

    # Per-category counts.
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
        description="Phase 2: Physical layout analysis — detect elements & hierarchy."
    )
    parser.add_argument(
        "--image-dir", type=Path, default=IMAGE_DIR,
        help="Directory containing page_*.png files.",
    )
    parser.add_argument(
        "--phase1", type=Path, default=None,
        help="Path to Phase 1 output JSON (default: outputs/phase1_categories.json).",
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="First page number to process (1-indexed, inclusive).",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="Last page number to process (1-indexed, inclusive).",
    )
    parser.add_argument(
        "--window-size", type=int, default=PHASE2_WINDOW_SIZE,
        help="Number of consecutive pages per analysis window.",
    )

    args = parser.parse_args()
    run(
        image_dir=args.image_dir,
        phase1_path=args.phase1,
        start=args.start,
        end=args.end,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()
