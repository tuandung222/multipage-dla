"""
Phase 1 — Category Discovery via Multi-Page Context.

Goal:
    Given a multi-page document (as page images), use a powerful MLLM
    to observe sliding windows of consecutive pages and reason about
    which layout categories exist. The output is a consolidated,
    deduplicated list of categories that Phase 2 will use for
    component-level bounding-box extraction.

How it works:
    1. Load all page images from the configured directory.
    2. Slide a window of CONTEXT_WINDOW_SIZE pages across the document.
    3. For each window, send the images + a reasoning prompt to the MLLM.
    4. Collect per-window category proposals.
    5. Merge and deduplicate into a single canonical category list.
    6. Save the result to outputs/.

Usage:
    python phase1_category_discovery.py                     # all pages
    python phase1_category_discovery.py --start 1 --end 10  # pages 1-10
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image

from config import (
    CONTEXT_WINDOW_SIZE,
    IMAGE_DIR,
    MAX_TOKENS,
    MODEL_NAME,
    OUTPUT_DIR,
    TEMPERATURE,
)
from llm_client import chat_completion, make_image_content, make_text_content

# =====================================================================
#  Prompt
# =====================================================================

PHASE1_PROMPT = r"""[Role]
You are an Industrial-Grade Document Layout Analysis (DLA) Agent. Your primary function is to visually analyze multi-page document inputs and define a pragmatic, highly generalizable set of layout categories (labels) for an automated processing pipeline.

[Objective]
Observe the provided multi-page document, reason about its visual and logical hierarchy, and output a consolidated, practical list of layout categories.

[Design Philosophy & Constraints]
- Be Pragmatic & Robust: Do NOT over-complicate or become overly granular. In a production pipeline, too many micro-categories cause fragmentation, chunking issues, and logic failures.
- Focus on Macro-Structures: Group elements by their logical boundaries and downstream extraction purpose. (e.g., Instead of creating separate labels for "bulleted_list_item", "numbered_list_item", or individual lines, use a unified macro-level "List_Block" if it serves the same parsing purpose).
- Ignore Micro-Noise: Do not create separate categories for minor visual variations unless they fundamentally change how the text should be read or processed.
- Downstream-Aware: Every defined category MUST have a clear justification for WHY it needs to be isolated (e.g., 'Requires Table Structure Recognition', 'Acts as a semantic boundary for text chunking', 'Safe to discard as noise').

[Task Instructions]
1. Holistically scan all provided document pages.
2. Identify the recurring, structurally significant components based on the design philosophy.
3. Reason about how these components should be grouped for a clean, efficient data extraction pipeline.
4. Output your final reasoning and category list in the exact JSON format below.

[Expected JSON Output Format]
{
  "reasoning_process": "A brief, 2-3 sentence explanation of your observation across the pages and why you chose to group certain elements together.",
  "categories": [
    {
      "class_name": "StandardizedName (e.g., Title, Text_Block, Table, Image, Header_Footer)",
      "description": "Clear, concise definition of the visual and logical boundaries of this category.",
      "downstream_purpose": "Practical reason for this category in a data pipeline (e.g., 'Discarded to reduce context noise', 'Passed to specialized Vision-Language Model')."
    }
  ]
}"""


# =====================================================================
#  Core logic
# =====================================================================

def load_page_images(
    image_dir: Path,
    start: int | None = None,
    end: int | None = None,
) -> list[tuple[int, Image.Image]]:
    """
    Load page images sorted by filename.

    Returns a list of (page_number, PIL.Image) tuples.
    `start` and `end` are 1-indexed, inclusive.
    """
    paths = sorted(image_dir.glob("page_*.png"))
    if not paths:
        raise FileNotFoundError(f"No page_*.png files found in {image_dir}")

    pages: list[tuple[int, Image.Image]] = []
    for p in paths:
        # Extract page number from filename like page_001.png
        num = int(re.search(r"(\d+)", p.stem).group(1))
        if start and num < start:
            continue
        if end and num > end:
            continue
        pages.append((num, Image.open(p).convert("RGB")))

    print(f"Loaded {len(pages)} pages from {image_dir}")
    return pages


def build_window_message(
    window: list[tuple[int, Image.Image]],
) -> list[dict]:
    """
    Build the chat message for one sliding window.

    The message contains:
      - One image content block per page (in order).
      - A text block with page labels + the reasoning prompt.
    """
    content: list[dict] = []

    # Add images first so the model "sees" them before reading the prompt.
    for page_num, img in window:
        content.append(make_text_content(f"--- Page {page_num} ---"))
        content.append(make_image_content(img))

    # Add the analysis prompt last.
    content.append(make_text_content(PHASE1_PROMPT))

    return [{"role": "user", "content": content}]


def parse_json_response(raw: str) -> dict:
    """
    Extract the first JSON object from the model response.

    Handles responses wrapped in ```json ... ``` markdown fences.
    """
    # Try to find JSON inside markdown code fences first.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    # Fallback: find the outermost { ... }.
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise ValueError(f"Could not extract JSON from model response:\n{raw[:500]}")


def analyze_window(
    window: list[tuple[int, Image.Image]],
    window_idx: int,
) -> dict:
    """Send one window to the MLLM and return the parsed JSON result."""
    page_nums = [p[0] for p in window]
    print(f"  Window {window_idx}: pages {page_nums} ... ", end="", flush=True)

    messages = build_window_message(window)
    t0 = time.time()
    raw = chat_completion(
        messages,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")

    result = parse_json_response(raw)
    result["_window_pages"] = page_nums
    return result


def merge_categories(window_results: list[dict]) -> dict:
    """
    Merge category proposals from all windows into a single canonical list.

    Strategy: deduplicate by class_name (case-insensitive), keeping the
    description and downstream_purpose from the first occurrence.
    """
    seen: dict[str, dict] = {}  # lower(class_name) -> category dict

    for wr in window_results:
        for cat in wr.get("categories", []):
            key = cat["class_name"].strip().lower()
            if key not in seen:
                seen[key] = cat

    merged = list(seen.values())
    return {
        "total_windows_analyzed": len(window_results),
        "merged_category_count": len(merged),
        "categories": merged,
        "per_window_details": window_results,
    }


# =====================================================================
#  Main
# =====================================================================

def run(
    image_dir: Path = IMAGE_DIR,
    start: int | None = None,
    end: int | None = None,
    window_size: int = CONTEXT_WINDOW_SIZE,
) -> dict:
    """Execute Phase 1 category discovery on the given page range."""
    pages = load_page_images(image_dir, start=start, end=end)

    if len(pages) == 0:
        raise ValueError("No pages to process.")

    # Build sliding windows (with overlap).
    windows: list[list[tuple[int, Image.Image]]] = []
    step = max(1, window_size - 1)  # overlap by 1 page
    for i in range(0, len(pages), step):
        window = pages[i : i + window_size]
        windows.append(window)

    print(f"Phase 1: {len(windows)} window(s), window_size={window_size}")
    print(f"Model: {MODEL_NAME}\n")

    # Process each window sequentially (to respect rate limits).
    window_results: list[dict] = []
    for idx, window in enumerate(windows):
        result = analyze_window(window, idx)
        window_results.append(result)

    # Merge all window proposals.
    merged = merge_categories(window_results)

    # Save output.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "phase1_categories.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\nPhase 1 complete.")
    print(f"  Merged categories: {merged['merged_category_count']}")
    print(f"  Output saved to:   {out_path}")

    # Pretty-print the category names.
    for cat in merged["categories"]:
        print(f"    - {cat['class_name']}: {cat['description'][:80]}")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Multi-page category discovery for DLA."
    )
    parser.add_argument(
        "--image-dir", type=Path, default=IMAGE_DIR,
        help="Directory containing page_*.png files.",
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
        "--window-size", type=int, default=CONTEXT_WINDOW_SIZE,
        help="Number of consecutive pages per analysis window.",
    )

    args = parser.parse_args()
    run(
        image_dir=args.image_dir,
        start=args.start,
        end=args.end,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()
