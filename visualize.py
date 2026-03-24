"""
Shared visualization utilities for the DLA pipeline.

Draws bounding boxes and labels on page images so that
detection results can be inspected visually.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# ── Color palette (deterministic per category) ───────────────────
_PALETTE = [
    "#FF4136", "#2ECC40", "#0074D9", "#FF851B", "#B10DC9",
    "#FFDC00", "#39CCCC", "#F012BE", "#01FF70", "#85144b",
    "#7FDBFF", "#3D9970", "#111111", "#AAAAAA", "#E65100",
]


def _color_for(category: str) -> str:
    """Return a deterministic hex color for a category name."""
    idx = hash(category) % len(_PALETTE)
    return _PALETTE[idx]


def draw_elements_on_page(
    image: Image.Image,
    elements: list[dict],
    box_key: str = "box_2d",
    label_key: str = "category",
    id_key: str = "node_id",
    expand_px: int = 5,
) -> Image.Image:
    """
    Draw bounding boxes and labels on a copy of the page image.

    Parameters
    ----------
    image : PIL.Image
        The original page image (will NOT be modified).
    elements : list[dict]
        Each dict must contain `box_key` ([ymin, xmin, ymax, xmax] in 0-1000)
        and `label_key` (category string).
    box_key, label_key, id_key : str
        Keys to look up in each element dict.
    expand_px : int
        Pixels to expand each box outward (guards against model boxes
        that are not perfectly tight). Set to 0 to disable.

    Returns
    -------
    PIL.Image with boxes and labels drawn on top.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Try to load a reasonable font; fall back to default.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for elem in elements:
        box = elem.get(box_key)
        if not box or len(box) != 4:
            continue

        ymin, xmin, ymax, xmax = box
        # Convert from 0-1000 normalised coords to pixel coords.
        x0 = xmin / 1000.0 * w
        y0 = ymin / 1000.0 * h
        x1 = xmax / 1000.0 * w
        y1 = ymax / 1000.0 * h

        # Expand box outward (clamped to image bounds).
        if expand_px:
            x0 = max(0, x0 - expand_px)
            y0 = max(0, y0 - expand_px)
            x1 = min(w, x1 + expand_px)
            y1 = min(h, y1 + expand_px)

        category = elem.get(label_key, "unknown")
        node_id = elem.get(id_key, "")
        color = _color_for(category)

        # Draw rectangle.
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        # Draw label background + text.
        label = f"{node_id} [{category}]"
        bbox_text = draw.textbbox((x0, y0), label, font=font)
        draw.rectangle(
            [bbox_text[0] - 1, bbox_text[1] - 1, bbox_text[2] + 1, bbox_text[3] + 1],
            fill=color,
        )
        draw.text((x0, y0), label, fill="white", font=font)

    return img
