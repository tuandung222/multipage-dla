"""
Phase 3 — Document Tree Assembly (Rule-Based).

Builds a hierarchical document tree from Phase 2 detected elements.
No MLLM needed — all information is already in the JSON:
  - parent_node_id  → tree edges
  - continues_previous_node → merge split nodes
  - content_snippet + category → node labels

Operations:
  1. Merge continuation nodes (text split across pages).
  2. Build parent-child tree from parent_node_id.
  3. Attach orphan nodes (missing parent) to nearest ancestor.
  4. Export: JSON tree, Mermaid diagram, indented text outline.

Usage:
    python phase3_tree_assembly.py
    python phase3_tree_assembly.py --input outputs/phase2_incremental.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from config import OUTPUT_DIR

# =====================================================================
#  Data model
# =====================================================================

@dataclass
class TreeNode:
    """A single node in the document tree."""
    node_id: str
    category: str
    snippet: str
    page_number: int
    box_2d: list[int] | None = None
    parent_id: str | None = None
    merged_from: list[str] = field(default_factory=list)
    children: list[TreeNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Recursively convert to a JSON-serialisable dict."""
        d = {
            "node_id": self.node_id,
            "category": self.category,
            "snippet": self.snippet,
            "page_number": self.page_number,
        }
        if self.merged_from:
            d["merged_from"] = self.merged_from
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


# =====================================================================
#  Step 1: Merge continuation nodes
# =====================================================================

def merge_continuations(elements: list[dict]) -> list[dict]:
    """
    Merge nodes where continues_previous_node is set.

    When node B continues node A:
      - Append B's snippet to A's snippet.
      - Record B's node_id in A's merged_from.
      - Remove B from the list.
      - Update any references to B → A (parent_node_id).
    """
    # Build lookup.
    by_id: dict[str, dict] = {e["node_id"]: e for e in elements}

    # Find all continuation pairs.
    continuations: dict[str, str] = {}  # child_id → parent_id
    for e in elements:
        cont = e.get("continues_previous_node")
        if cont and cont in by_id:
            continuations[e["node_id"]] = cont

    # Resolve chains: if A→B→C, resolve C→A.
    def resolve(nid: str) -> str:
        visited = set()
        while nid in continuations and nid not in visited:
            visited.add(nid)
            nid = continuations[nid]
        return nid

    # Merge.
    merged_ids: set[str] = set()
    for child_id, _ in continuations.items():
        target_id = resolve(child_id)
        target = by_id[target_id]
        child = by_id[child_id]

        # Append snippet.
        target.setdefault("_merged_snippets", [target["content_snippet"]])
        target["_merged_snippets"].append(child["content_snippet"])
        target["content_snippet"] = " ".join(target["_merged_snippets"])

        # Track merge.
        target.setdefault("merged_from", [])
        target["merged_from"].append(child_id)

        merged_ids.add(child_id)

    # Rewrite parent references: any node pointing to a merged child → point to target.
    id_remap = {cid: resolve(cid) for cid in continuations}
    for e in elements:
        pid = e.get("parent_node_id")
        if pid and pid in id_remap:
            e["parent_node_id"] = id_remap[pid]

    # Filter out merged children.
    result = [e for e in elements if e["node_id"] not in merged_ids]

    if merged_ids:
        print(f"  Merged {len(merged_ids)} continuation node(s): "
              f"{', '.join(sorted(merged_ids))}")

    return result


# =====================================================================
#  Step 2: Build tree
# =====================================================================

def build_tree(elements: list[dict]) -> tuple[list[TreeNode], list[TreeNode]]:
    """
    Build a forest (list of root nodes) from parent_node_id links.

    Returns (roots, orphans) where orphans had a parent_id that
    doesn't exist in the element list.
    """
    # Create TreeNode for each element.
    nodes: dict[str, TreeNode] = {}
    for e in elements:
        nodes[e["node_id"]] = TreeNode(
            node_id=e["node_id"],
            category=e["category"],
            snippet=e["content_snippet"][:120],
            page_number=e.get("page_number", 0),
            box_2d=e.get("box_2d"),
            parent_id=e.get("parent_node_id"),
            merged_from=e.get("merged_from", []),
        )

    # Link children to parents.
    roots: list[TreeNode] = []
    orphans: list[TreeNode] = []

    for node in nodes.values():
        if node.parent_id is None:
            roots.append(node)
        elif node.parent_id in nodes:
            nodes[node.parent_id].children.append(node)
        else:
            # Parent not found → orphan (attach to root level).
            orphans.append(node)
            roots.append(node)

    if orphans:
        print(f"  {len(orphans)} orphan node(s) attached to root level: "
              f"{', '.join(o.node_id for o in orphans)}")

    return roots, orphans


# =====================================================================
#  Step 3: Export formats
# =====================================================================

def tree_to_json(roots: list[TreeNode]) -> list[dict]:
    """Convert the tree forest to a JSON-serialisable list."""
    return [r.to_dict() for r in roots]


def tree_to_outline(roots: list[TreeNode], indent: int = 0) -> str:
    """
    Render the tree as an indented text outline.

    Example:
        [Section_Heading] 1. Scope (p.5)
          [Section_Header] 1.1 Introduction (p.5)
            [Text_Block] 1.1.1 It is our policy... (p.5)
            [Text_Block] 1.1.2 CCA has developed... (p.5)
    """
    lines: list[str] = []
    for node in roots:
        prefix = "  " * indent
        snippet_short = node.snippet[:60].replace("\n", " ")
        merge_tag = f" [merged:{','.join(node.merged_from)}]" if node.merged_from else ""
        lines.append(
            f"{prefix}[{node.category}] {snippet_short}  (p.{node.page_number}){merge_tag}"
        )
        if node.children:
            lines.append(tree_to_outline(node.children, indent + 1))
    return "\n".join(lines)


def tree_to_mermaid(
    roots: list[TreeNode],
    continuations: list[dict] | None = None,
) -> str:
    """
    Render the document tree as a vertical Mermaid flowchart.

    Design choices for GitHub readability:
      - Top-down (TB) layout for vertical reading.
      - Edge labels show the relationship type (contains, defines, etc.).
      - Continuation (cross-page merge) shown as dashed "continues" edges.
      - Nodes grouped by page using subgraphs.
      - Category-specific shapes for visual distinction.
      - Styling via classDef for category colours.
    """
    lines: list[str] = ["graph TB"]

    # ── Category → shape mapping ─────────────────────────────────
    shape_map = {
        "Title":           ('([', "])"),    # stadium
        "Section_Heading": ("[[", "]]"),    # subroutine
        "Section_Header":  ("[[", "]]"),
        "Table":           ("[/", "/]"),    # parallelogram
        "Table_of_Contents": ("[/", "/]"),
        "Diagram":         ("{{", "}}"),    # hexagon
        "Image":           ("{{", "}}"),
        "Logo":            ("{{", "}}"),
        "List_Block":      ("[(", ")]"),    # cylindrical
    }
    default_shape = ("[", "]")

    # ── Relation label based on parent→child category pair ───────
    def relation_label(parent: TreeNode, child: TreeNode) -> str:
        p, c = parent.category, child.category
        if p in ("Section_Heading", "Section_Header"):
            if c in ("Section_Header",):
                return "has_subsection"
            if c in ("Text_Block",):
                return "contains_text"
            if c in ("Table",):
                return "contains_table"
            if c in ("List_Block",):
                return "contains_list"
            if c in ("Diagram", "Image"):
                return "contains_figure"
            if c in ("Table_of_Contents",):
                return "contains_toc"
            return "contains"
        return "contains"

    def sanitize(text: str) -> str:
        """Escape chars that break Mermaid syntax."""
        return (text
                .replace('"', "'")
                .replace("\n", " ")
                .replace("(", "&#40;")
                .replace(")", "&#41;")
                .replace("[", "&#91;")
                .replace("]", "&#93;")
                [:45])

    # ── Collect all nodes + group by page ────────────────────────
    all_nodes: list[TreeNode] = []
    edges: list[str] = []

    def collect(node: TreeNode, parent: TreeNode | None) -> None:
        all_nodes.append(node)
        if parent is not None:
            rel = relation_label(parent, node)
            edges.append(f"    {parent.node_id} -->|{rel}| {node.node_id}")
        for child in node.children:
            collect(child, node)

    for root in roots:
        collect(root, None)

    # Group nodes by page.
    by_page: dict[int, list[TreeNode]] = {}
    for n in all_nodes:
        by_page.setdefault(n.page_number, []).append(n)

    # ── Emit subgraphs per page ──────────────────────────────────
    for page_num in sorted(by_page):
        nodes_on_page = by_page[page_num]
        lines.append(f"")
        lines.append(f"    subgraph page_{page_num} [Page {page_num}]")
        lines.append(f"        direction TB")
        for node in nodes_on_page:
            open_b, close_b = shape_map.get(node.category, default_shape)
            snippet = sanitize(node.snippet)
            label = f"{node.node_id}\\n{node.category}\\n{snippet}"
            lines.append(f'        {node.node_id}{open_b}"{label}"{close_b}')
        lines.append(f"    end")

    # ── Emit edges (after subgraphs so Mermaid resolves IDs) ─────
    lines.append("")
    lines.append("    %% Parent-child relationships")
    lines.extend(edges)

    # ── Continuation edges (cross-page merge) ────────────────────
    if continuations:
        lines.append("")
        lines.append("    %% Cross-page continuations")
        for c in continuations:
            child_id = c["node_id"]
            parent_id = c["continues"]
            lines.append(f"    {parent_id} -.->|continues| {child_id}")

    # ── Styling ──────────────────────────────────────────────────
    lines.append("")
    lines.append("    %% Styles")
    lines.append('    classDef heading fill:#4A90D9,stroke:#2C5F8A,color:#fff,font-weight:bold')
    lines.append('    classDef text fill:#F5F5F5,stroke:#999,color:#333')
    lines.append('    classDef table fill:#FFE0B2,stroke:#E65100,color:#333')
    lines.append('    classDef figure fill:#C8E6C9,stroke:#2E7D32,color:#333')
    lines.append('    classDef meta fill:#E1BEE7,stroke:#6A1B9A,color:#333')
    lines.append('    classDef boilerplate fill:#EEEEEE,stroke:#BDBDBD,color:#666,stroke-dasharray:5')

    # Apply styles.
    style_map = {
        "Section_Heading": "heading",
        "Section_Header": "heading",
        "Title": "heading",
        "Text_Block": "text",
        "Table": "table",
        "Table_of_Contents": "table",
        "Diagram": "figure",
        "Image": "figure",
        "List_Block": "text",
        "Logo": "meta",
        "Metadata_Block": "meta",
        "Header": "boilerplate",
        "Footer": "boilerplate",
    }
    for cls_name, style_name in style_map.items():
        ids = [n.node_id for n in all_nodes if n.category == cls_name]
        if ids:
            lines.append(f"    class {','.join(ids)} {style_name}")

    return "\n".join(lines)


# =====================================================================
#  Statistics
# =====================================================================

def compute_stats(roots: list[TreeNode]) -> dict:
    """Compute tree statistics."""
    total = 0
    max_depth = 0
    category_counts: dict[str, int] = {}
    leaf_count = 0

    def walk(node: TreeNode, depth: int) -> None:
        nonlocal total, max_depth, leaf_count
        total += 1
        max_depth = max(max_depth, depth)
        category_counts[node.category] = category_counts.get(node.category, 0) + 1
        if not node.children:
            leaf_count += 1
        for child in node.children:
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)

    return {
        "total_nodes": total,
        "root_nodes": len(roots),
        "leaf_nodes": leaf_count,
        "max_depth": max_depth,
        "category_counts": category_counts,
    }


# =====================================================================
#  Main
# =====================================================================

def run(input_path: Path | None = None) -> dict:
    """Execute Phase 3: build document tree from Phase 2 output."""
    if input_path is None:
        input_path = OUTPUT_DIR / "phase2_incremental.json"

    print(f"Phase 3: Document Tree Assembly")
    print(f"  Input: {input_path}\n")

    # Load Phase 2 output.
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    elements = data["detected_elements"]
    continuations = data.get("document_summary", {}).get("cross_page_continuations", [])
    print(f"  Loaded {len(elements)} elements, {len(continuations)} continuation(s)")

    # Step 1: Merge continuations.
    elements = merge_continuations(elements)
    print(f"  After merge: {len(elements)} elements")

    # Step 2: Build tree.
    roots, orphans = build_tree(elements)

    # Step 3: Compute stats.
    stats = compute_stats(roots)

    # Step 4: Export.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON tree.
    tree_json = tree_to_json(roots)
    json_path = OUTPUT_DIR / "phase3_document_tree.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "tree": tree_json}, f, indent=2, ensure_ascii=False)

    # Text outline.
    outline = tree_to_outline(roots)
    outline_path = OUTPUT_DIR / "phase3_outline.txt"
    with open(outline_path, "w", encoding="utf-8") as f:
        f.write(outline)

    # Mermaid diagram.
    mermaid = tree_to_mermaid(roots, continuations=continuations)
    mermaid_path = OUTPUT_DIR / "phase3_mermaid.md"
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(f"```mermaid\n{mermaid}\n```\n")

    # Print summary.
    print(f"\n{'='*60}")
    print(f"Phase 3 complete.")
    print(f"  Total nodes:  {stats['total_nodes']}")
    print(f"  Root nodes:   {stats['root_nodes']}")
    print(f"  Leaf nodes:   {stats['leaf_nodes']}")
    print(f"  Max depth:    {stats['max_depth']}")
    print(f"\n  Outputs:")
    print(f"    Tree JSON:  {json_path}")
    print(f"    Outline:    {outline_path}")
    print(f"    Mermaid:    {mermaid_path}")

    print(f"\n{'='*60}")
    print(f"  Document Outline:\n")
    print(outline)

    return {"stats": stats, "tree": tree_json}


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Assemble document tree from Phase 2 detected elements."
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to Phase 2 output JSON (default: outputs/phase2_incremental.json).",
    )
    args = parser.parse_args()
    run(input_path=args.input)


if __name__ == "__main__":
    main()
