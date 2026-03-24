"""
Baseline Experiment — End-to-End (No Phase Separation).

Purpose:
    Demonstrate that asking the MLLM to do everything in a single prompt
    (discover categories + localize bounding boxes + infer hierarchy)
    produces worse results than our multi-phase pipeline.

Experiments:
    A) Single-page e2e: send 1 page, ask for full analysis.
    B) Multi-page e2e:  send 3 pages, ask for full analysis.

Both use the SAME model and temperature as our pipeline, but WITHOUT
the Phase 1 category reasoning step. The MLLM must invent categories
AND draw boxes AND infer hierarchy all at once.

Usage:
    python baseline_e2e_experiment.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from PIL import Image

from config import IMAGE_DIR, MODEL_NAME, OUTPUT_DIR, TEMPERATURE
from llm_client import chat_completion, make_image_content, make_text_content
from visualize import draw_elements_on_page

# =====================================================================
#  E2E Prompt — everything in one shot, no prior category reasoning
# =====================================================================

E2E_PROMPT_SINGLE = r"""[Role]
You are a Document Layout Analysis agent.

[Task]
Analyze this document page image. You must:
1. Identify all layout categories present (you decide what categories exist).
2. Locate every element with a bounding box.
3. Assign hierarchical parent-child relationships.
4. Assign a unique node_id to each element.

[Constraints]
- Bounding boxes: [ymin, xmin, ymax, xmax] normalized to 1000x1000 grid.
- content_snippet: first 100 characters only.
- Be thorough — do not skip any visible element.

[Output Format — JSON only]
{
  "categories_discovered": ["list of category names you identified"],
  "detected_elements": [
    {
      "node_id": "string",
      "category": "string",
      "box_2d": [ymin, xmin, ymax, xmax],
      "content_snippet": "string (max 100 chars)",
      "parent_node_id": "string | null"
    }
  ]
}"""

E2E_PROMPT_MULTI = r"""[Role]
You are a Document Layout Analysis agent.

[Task]
Analyze these 3 consecutive document page images. You must:
1. Identify all layout categories present (you decide what categories exist).
2. Locate every element on every page with a bounding box.
3. Assign hierarchical parent-child relationships across pages.
4. Assign globally unique node_ids across all pages.
5. Detect elements that continue across page boundaries.

[Constraints]
- image_index: 0-based index of the page in the sequence.
- Bounding boxes: [ymin, xmin, ymax, xmax] normalized to 1000x1000 grid for EACH page.
- content_snippet: first 100 characters only.
- Be thorough — do not skip any visible element on any page.

[Output Format — JSON only]
{
  "categories_discovered": ["list of category names you identified"],
  "detected_elements": [
    {
      "node_id": "string",
      "image_index": integer,
      "category": "string",
      "box_2d": [ymin, xmin, ymax, xmax],
      "content_snippet": "string (max 100 chars)",
      "parent_node_id": "string | null"
    }
  ]
}"""


# =====================================================================
#  Helpers
# =====================================================================

def load_pages(start: int, end: int) -> list[tuple[int, Image.Image]]:
    paths = sorted(IMAGE_DIR.glob("page_*.png"))
    pages = []
    for p in paths:
        num = int(re.search(r"(\d+)", p.stem).group(1))
        if start <= num <= end:
            pages.append((num, Image.open(p).convert("RGB")))
    return pages


def parse_json(raw: str) -> dict:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError(f"Could not extract JSON:\n{raw[:500]}")


# =====================================================================
#  Experiment A: Single-page E2E
# =====================================================================

def run_single_page_e2e(pages: list[tuple[int, Image.Image]]) -> list[dict]:
    """Analyze each page independently with a single e2e prompt."""
    results = []
    for page_num, img in pages:
        print(f"  [Single E2E] Page {page_num} ... ", end="", flush=True)
        content = [
            make_text_content(f"Document page {page_num}:"),
            make_image_content(img),
            make_text_content(E2E_PROMPT_SINGLE),
        ]
        t0 = time.time()
        raw = chat_completion(
            [{"role": "user", "content": content}],
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=16384,
        )
        elapsed = time.time() - t0
        result = parse_json(raw)
        n = len(result.get("detected_elements", []))
        print(f"done ({elapsed:.1f}s, {n} elements)")

        # Tag elements with page_number.
        for e in result.get("detected_elements", []):
            e["page_number"] = page_num

        result["_page_number"] = page_num
        results.append(result)

    return results


# =====================================================================
#  Experiment B: Multi-page E2E (3 pages at once)
# =====================================================================

def run_multi_page_e2e(pages: list[tuple[int, Image.Image]]) -> dict:
    """Send all 3 pages at once with a single e2e prompt."""
    print(f"  [Multi E2E] Pages {[p[0] for p in pages]} ... ", end="", flush=True)

    content = []
    for idx, (page_num, img) in enumerate(pages):
        content.append(make_text_content(f"--- Page index {idx} (page {page_num}) ---"))
        content.append(make_image_content(img))
    content.append(make_text_content(E2E_PROMPT_MULTI))

    t0 = time.time()
    raw = chat_completion(
        [{"role": "user", "content": content}],
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=16384,
    )
    elapsed = time.time() - t0
    result = parse_json(raw)
    n = len(result.get("detected_elements", []))
    print(f"done ({elapsed:.1f}s, {n} elements)")

    # Map image_index → page_number.
    idx_to_page = {idx: pn for idx, (pn, _) in enumerate(pages)}
    for e in result.get("detected_elements", []):
        e["page_number"] = idx_to_page.get(e.get("image_index", 0), pages[0][0])

    return result


# =====================================================================
#  Comparison metrics
# =====================================================================

def compute_metrics(result: dict | list, label: str) -> dict:
    """Compute summary metrics for comparison."""
    if isinstance(result, list):
        elements = []
        cats = set()
        for r in result:
            elements.extend(r.get("detected_elements", []))
            cats.update(r.get("categories_discovered", []))
    else:
        elements = result.get("detected_elements", [])
        cats = set(result.get("categories_discovered", []))

    # Count per page.
    per_page: dict[int, int] = {}
    for e in elements:
        pn = e.get("page_number", 0)
        per_page[pn] = per_page.get(pn, 0) + 1

    # Check bbox quality: how many have valid box_2d?
    valid_boxes = 0
    degenerate_boxes = 0
    for e in elements:
        box = e.get("box_2d", [])
        if len(box) == 4:
            ymin, xmin, ymax, xmax = box
            if ymin < ymax and xmin < xmax and 0 <= ymin <= 1000 and 0 <= xmax <= 1000:
                valid_boxes += 1
            else:
                degenerate_boxes += 1

    # Count elements with parent.
    with_parent = sum(1 for e in elements if e.get("parent_node_id"))

    return {
        "label": label,
        "total_elements": len(elements),
        "categories_count": len(cats),
        "categories": sorted(cats),
        "per_page_counts": per_page,
        "valid_boxes": valid_boxes,
        "degenerate_boxes": degenerate_boxes,
        "elements_with_parent": with_parent,
    }


# =====================================================================
#  Visualisation
# =====================================================================

def save_vis(
    elements: list[dict],
    pages: list[tuple[int, Image.Image]],
    out_dir: Path,
    prefix: str,
) -> None:
    page_map = {pn: img for pn, img in pages}
    by_page: dict[int, list[dict]] = {}
    for e in elements:
        pn = e.get("page_number", 0)
        by_page.setdefault(pn, []).append(e)

    for pn, img in sorted(page_map.items()):
        elems = by_page.get(pn, [])
        vis = draw_elements_on_page(img, elems, expand_px=8)
        path = out_dir / f"{prefix}_page_{pn:03d}.jpg"
        vis.save(path, "JPEG", quality=92)


# =====================================================================
#  Main
# =====================================================================

def run():
    pages = load_pages(1, 3)
    print(f"Baseline E2E Experiment — {len(pages)} pages")
    print(f"Model: {MODEL_NAME}\n")

    # Run experiments.
    print("Experiment A: Single-page E2E (no prior category reasoning)")
    single_results = run_single_page_e2e(pages)

    print("\nExperiment B: Multi-page E2E (3 pages, no prior category reasoning)")
    multi_result = run_multi_page_e2e(pages)

    # Load our pipeline results for comparison.
    pipeline_path = OUTPUT_DIR / "phase2_incremental.json"
    with open(pipeline_path) as f:
        pipeline_data = json.load(f)
    pipeline_elements = [
        e for e in pipeline_data["detected_elements"]
        if e.get("page_number", 0) <= 3
    ]
    pipeline_cats_path = OUTPUT_DIR / "phase1_categories.json"
    with open(pipeline_cats_path) as f:
        pipeline_cats = json.load(f)

    # Metrics.
    m_single = compute_metrics(single_results, "Single-Page E2E")
    m_multi = compute_metrics(multi_result, "Multi-Page E2E (3 pages)")
    m_pipeline = compute_metrics(
        {"detected_elements": pipeline_elements,
         "categories_discovered": [c["class_name"] for c in pipeline_cats["categories"]]},
        "Our Pipeline (Phase1 + Phase2)"
    )

    # Save outputs.
    exp_dir = OUTPUT_DIR / "baseline_experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results.
    with open(exp_dir / "single_page_e2e.json", "w") as f:
        json.dump(single_results, f, indent=2, ensure_ascii=False)
    with open(exp_dir / "multi_page_e2e.json", "w") as f:
        json.dump(multi_result, f, indent=2, ensure_ascii=False)

    # Save visualisations.
    all_single_elems = []
    for r in single_results:
        all_single_elems.extend(r.get("detected_elements", []))
    save_vis(all_single_elems, pages, exp_dir, "single_e2e")
    save_vis(multi_result.get("detected_elements", []), pages, exp_dir, "multi_e2e")
    # Pipeline vis already exists, copy reference.
    save_vis(pipeline_elements, pages, exp_dir, "pipeline")

    # Print comparison table.
    print(f"\n{'='*70}")
    print(f"  COMPARISON: E2E Baselines vs Our Multi-Phase Pipeline")
    print(f"  (Pages 1-3 of Quality Manual)")
    print(f"{'='*70}\n")

    header = f"{'Metric':<35} {'Single E2E':>12} {'Multi E2E':>12} {'Pipeline':>12}"
    print(header)
    print("-" * len(header))

    metrics_list = [m_single, m_multi, m_pipeline]
    rows = [
        ("Total elements detected", "total_elements"),
        ("Categories discovered", "categories_count"),
        ("Valid bounding boxes", "valid_boxes"),
        ("Degenerate/invalid boxes", "degenerate_boxes"),
        ("Elements with parent (hierarchy)", "elements_with_parent"),
    ]
    for label, key in rows:
        vals = [str(m[key]) for m in metrics_list]
        print(f"{label:<35} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # Per-page breakdown.
    print(f"\n  Per-page element counts:")
    for pn in [1, 2, 3]:
        vals = [str(m["per_page_counts"].get(pn, 0)) for m in metrics_list]
        print(f"    Page {pn:<30} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # Categories discovered.
    print(f"\n  Categories discovered:")
    for m in metrics_list:
        print(f"    {m['label']}:")
        for cat in m["categories"]:
            print(f"      - {cat}")

    # Save comparison report.
    report = {
        "experiment": "Baseline E2E vs Multi-Phase Pipeline",
        "pages_analyzed": [1, 2, 3],
        "model": MODEL_NAME,
        "results": {
            "single_page_e2e": m_single,
            "multi_page_e2e": m_multi,
            "pipeline": m_pipeline,
        },
    }
    with open(exp_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Outputs saved to: {exp_dir}/")
    print(f"    - single_e2e_page_*.jpg  (single-page E2E visualisations)")
    print(f"    - multi_e2e_page_*.jpg   (multi-page E2E visualisations)")
    print(f"    - pipeline_page_*.jpg    (our pipeline visualisations)")
    print(f"    - comparison_report.json")


if __name__ == "__main__":
    run()
