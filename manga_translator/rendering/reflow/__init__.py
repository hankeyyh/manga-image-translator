from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from ...utils import BASE_PATH, TextBlock, get_logger
from ...utils.generic2 import color_difference
from ..text_render import (
    add_color,
    get_font_path,
    get_string_width,
    put_char_horizontal,
    set_font,
)
from .constraint import SafeBox, estimate_balloon_mask, load_panels, resolve_safe_box, shared_balloon_flags
from .layout import Layout, layout_in_box

logger = get_logger("render")


def _measure(font_size: int, text: str) -> float:
    if not text:
        return 0.0
    return float(get_string_width(font_size, text))


def _font_bounds(
    img: np.ndarray,
    region: TextBlock,
    box: SafeBox,
    font_size_fixed: Optional[int],
    font_size_offset: int,
    font_size_minimum: int,
) -> tuple[int, int]:
    img_h, img_w = img.shape[:2]
    if font_size_minimum == -1:
        font_size_minimum = round((img_h + img_w) / 200)
    min_size = max(1, int(font_size_minimum))
    if font_size_fixed is not None:
        size = max(int(font_size_fixed), min_size)
        return min_size, size
    base = region.font_size if region.font_size and region.font_size > 0 else min_size
    base = max(base + int(font_size_offset or 0), min_size)
    # Allow growing past OCR size so text can fill a large bubble.
    fill_cap = max(base, int(min(box.w, box.h) * 0.45))
    max_size = min(fill_cap, max(min_size, min(box.w, box.h)))
    return min_size, max(max_size, min_size)


def _layout_collides(layout: Layout, box: SafeBox) -> bool:
    if box.mask is None or layout.block_w <= 0 or layout.block_h <= 0:
        return False
    origin_x = box.x + (box.w - layout.block_w) / 2.0
    origin_y = box.y + (box.h - layout.block_h) / 2.0
    y = origin_y
    mask = box.mask
    mh, mw = mask.shape[:2]
    spacing = layout.line_height - layout.font_size if layout.line_height else 0
    for width in layout.line_widths or [layout.block_w]:
        x = origin_x + (layout.block_w - width) / 2.0
        x1 = max(0, int(np.floor(x)))
        y1 = max(0, int(np.floor(y)))
        x2 = min(mw, int(np.ceil(x + width)))
        y2 = min(mh, int(np.ceil(y + layout.font_size)))
        if x2 <= x1 or y2 <= y1:
            y += layout.font_size + spacing
            continue
        roi = mask[y1:y2, x1:x2]
        if roi.size and np.mean(roi == 0) > 0.02:
            return True
        y += layout.font_size + spacing
    return False


def _render_lines(
    layout: Layout,
    fg,
    bg,
    line_spacing: float,
) -> Optional[np.ndarray]:
    if not layout.lines:
        return None
    font_size = layout.font_size
    bg_size = int(max(font_size * 0.07, 1)) if bg is not None else 0
    spacing_y = int(font_size * (line_spacing or 0.01))
    line_widths = layout.line_widths or [_measure(font_size, line) for line in layout.lines]
    max_w = int(max(line_widths)) if line_widths else 0
    canvas_w = max_w + (font_size + bg_size) * 2
    canvas_h = font_size * len(layout.lines) + spacing_y * max(len(layout.lines) - 1, 0) + (font_size + bg_size) * 2
    if canvas_w <= 0 or canvas_h <= 0:
        return None
    canvas_text = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_border = canvas_text.copy()
    pen_y = font_size + bg_size
    for line, width in zip(layout.lines, line_widths):
        pen_x = font_size + bg_size + int((max_w - width) // 2)
        for ch in line:
            offset_x = put_char_horizontal(font_size, ch, [pen_x, pen_y], canvas_text, canvas_border, border_size=bg_size)
            pen_x += offset_x
        pen_y += spacing_y + font_size
    canvas_border = np.clip(canvas_border, 0, 255)
    line_box = add_color(canvas_text, fg, canvas_border, bg)
    x, y, w, h = cv2.boundingRect(canvas_border)
    if w <= 0 or h <= 0:
        return None
    return line_box[y:y + h, x:x + w]


def _paste_rgba(
    img: np.ndarray,
    rgba: np.ndarray,
    x: int,
    y: int,
    clip_xyxy: Optional[tuple[int, int, int, int]] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    ih, iw = img.shape[:2]
    rh, rw = rgba.shape[:2]
    x1, y1, x2, y2 = x, y, x + rw, y + rh
    if clip_xyxy is not None:
        cx1, cy1, cx2, cy2 = clip_xyxy
        x1, y1 = max(x1, cx1), max(y1, cy1)
        x2, y2 = min(x2, cx2), min(y2, cy2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)
    if x2 <= x1 or y2 <= y1:
        return img
    src_x1 = x1 - x
    src_y1 = y1 - y
    patch = rgba[src_y1:src_y1 + (y2 - y1), src_x1:src_x1 + (x2 - x1)]
    if patch.size == 0:
        return img
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    if mask is not None:
        mh, mw = mask.shape[:2]
        if y2 <= mh and x2 <= mw:
            alpha *= (mask[y1:y2, x1:x2] > 0).astype(np.float32)[..., None]
    color = patch[:, :, :3].astype(np.float32)
    dst = img[y1:y2, x1:x2].astype(np.float32)
    img[y1:y2, x1:x2] = np.clip(dst * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    return img


def _fg_bg(region: TextBlock, disable_font_border: bool):
    fg, bg = region.get_font_colors()
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if np.mean(fg) <= 127 else (0, 0, 0)
    if disable_font_border:
        bg = None
    return fg, bg


async def dispatch_reflow(
    img_canvas: np.ndarray,
    original_img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = "",
    font_name: str = "",
    line_spacing: Optional[float] = None,
    disable_font_border: bool = False,
    font_size_fixed: Optional[int] = None,
    font_size_offset: int = 0,
    font_size_minimum: int = -1,
    hyphenate: bool = True,
    rtl: bool = True,
) -> np.ndarray:
    """Reflow English-style horizontal text into a safe box (bubble interior ∩ panel, or padded OCR)."""
    if not text_regions:
        return img_canvas

    default_font_path = os.path.join(BASE_PATH, "fonts/comic shanns 2.ttf")
    if font_path:
        set_font(font_path)
    elif font_name:
        resolved = get_font_path(font_name) or default_font_path
        set_font(resolved)
    else:
        set_font(default_font_path)

    regions = [r for r in text_regions if r.translation]
    if not regions:
        return img_canvas

    spacing = 0.01 if line_spacing is None else float(line_spacing)
    panels = load_panels(original_img, rtl=rtl)
    balloon_masks = [estimate_balloon_mask(original_img, r) for r in regions]
    shared = shared_balloon_flags(balloon_masks)
    img = img_canvas

    for i, region in enumerate(tqdm(regions, "[reflow]")):
        others = [r for j, r in enumerate(regions) if j != i]
        box = resolve_safe_box(
            original_img,
            region,
            others,
            panels,
            balloon_mask=balloon_masks[i],
            shared=shared[i],
        )
        text = " ".join((region.get_translation_for_rendering() or "").replace("...", "…").split())
        if not text.strip():
            continue
        min_size, max_size = _font_bounds(img, region, box, font_size_fixed, font_size_offset, font_size_minimum)
        collides = (lambda layout, b=box: _layout_collides(layout, b)) if box.mask is not None and not box.fallback else None
        layout = layout_in_box(
            text,
            box.w,
            box.h,
            _measure,
            min_size,
            max_size,
            line_spacing=spacing,
            hyphenate=hyphenate,
            language="en_US" if (getattr(region, "target_lang", "ENG") or "ENG") == "ENG" else region.target_lang,
            collides=collides,
        )
        if layout.font_size <= min_size and (layout.block_h > box.h or layout.block_w > box.w):
            logger.info("reflow min font still overflows box for %r", text[:40])
        fg, bg = _fg_bg(region, disable_font_border)
        rgba = _render_lines(layout, fg, bg, spacing)
        if rgba is None:
            continue
        rh, rw = rgba.shape[:2]
        fit = min(box.w / max(rw, 1), box.h / max(rh, 1), 1.0)
        if fit < 0.99:
            rgba = cv2.resize(
                rgba,
                (max(1, int(rw * fit)), max(1, int(rh * fit))),
                interpolation=cv2.INTER_AREA,
            )
        cx = box.x + box.w / 2.0
        cy = box.y + box.h / 2.0
        paste_x = int(round(cx - rgba.shape[1] / 2.0))
        paste_y = int(round(cy - rgba.shape[0] / 2.0))
        img = _paste_rgba(img, rgba, paste_x, paste_y, clip_xyxy=box.xyxy, mask=box.mask)
    return img
