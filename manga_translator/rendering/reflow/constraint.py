from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..ballon_extractor import extract_ballon_region
from ...utils import TextBlock, get_logger

logger = get_logger("render")

# Keep balloon estimate close to the OCR box. manga2eng's 3x enlarge is what leaks across panels.
ENLARGE_RATIO = 1.35
OCR_PAD_RATIO = 0.08
MIN_SAFE_AREA_RATIO = 0.8
LEAK_AREA_RATIO = 4.0
PANEL_FILL_LEAK = 0.5
SHARED_MASK_IOU = 0.35
NEIGHBOR_PAD_RATIO = 0.08

Panel = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class SafeBox:
    x: int
    y: int
    w: int
    h: int
    mask: Optional[np.ndarray] = None
    fallback: bool = False
    panel: Panel = (0, 0, 0, 0)

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


def padding_px(img_shape: Sequence[int]) -> float:
    short = min(int(img_shape[0]), int(img_shape[1]))
    return float(np.clip(round(short * 0.006), 6, 10))


def ocr_xyxy(region: TextBlock) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in region.xyxy]
    return x1, y1, x2, y2


def ocr_area(region: TextBlock) -> int:
    x1, y1, x2, y2 = ocr_xyxy(region)
    return max(0, x2 - x1) * max(0, y2 - y1)


def panel_for_region(
    region: TextBlock,
    panels: Sequence[Panel],
    img_w: int,
    img_h: int,
) -> Panel:
    full = (0, 0, img_w, img_h)
    if not panels:
        return full
    cx, cy = float(region.center[0]), float(region.center[1])
    for x1, y1, x2, y2 in panels:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return int(x1), int(y1), int(x2), int(y2)
    dists = []
    for i, (x1, y1, x2, y2) in enumerate(panels):
        dx = max(x1 - cx, 0, cx - x2)
        dy = max(y1 - cy, 0, cy - y2)
        dists.append((dx * dx + dy * dy, i))
    x1, y1, x2, y2 = panels[min(dists)[1]]
    return int(x1), int(y1), int(x2), int(y2)


def clip_box_to_panel(x: int, y: int, w: int, h: int, panel: Panel) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = x, y, x + w, y + h
    px1, py1, px2, py2 = panel
    x1 = max(x1, px1)
    y1 = max(y1, py1)
    x2 = min(x2, px2)
    y2 = min(y2, py2)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def padded_ocr_box(region: TextBlock, panel: Panel, img_w: int, img_h: int) -> SafeBox:
    x1, y1, x2, y2 = ocr_xyxy(region)
    pad_x = max(4, int((x2 - x1) * OCR_PAD_RATIO))
    pad_y = max(4, int((y2 - y1) * OCR_PAD_RATIO))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)
    x, y, w, h = clip_box_to_panel(x1, y1, x2 - x1, y2 - y1, panel)
    return SafeBox(x=x, y=y, w=w, h=h, mask=None, fallback=True, panel=panel)


def clip_mask_to_panel(mask: np.ndarray, panel: Panel) -> np.ndarray:
    clipped = np.zeros_like(mask)
    x1, y1, x2, y2 = panel
    h, w = mask.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return clipped
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


def subtract_neighbors(
    mask: np.ndarray,
    region: TextBlock,
    other_regions: Sequence[TextBlock],
) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape[:2]
    for other in other_regions:
        ox1, oy1, ox2, oy2 = ocr_xyxy(other)
        pad_x = max(2, int((ox2 - ox1) * NEIGHBOR_PAD_RATIO))
        pad_y = max(2, int((oy2 - oy1) * NEIGHBOR_PAD_RATIO))
        ox1 = max(0, ox1 - pad_x)
        oy1 = max(0, oy1 - pad_y)
        ox2 = min(w, ox2 + pad_x)
        oy2 = min(h, oy2 + pad_y)
        out[oy1:oy2, ox1:ox2] = 0
    return out


def mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    inter = np.count_nonzero((a > 0) & (b > 0))
    union = np.count_nonzero((a > 0) | (b > 0))
    return float(inter) / float(union) if union else 0.0


def shared_balloon_flags(
    masks: Sequence[Optional[np.ndarray]],
    threshold: float = SHARED_MASK_IOU,
) -> List[bool]:
    n = len(masks)
    shared = [False] * n
    for i in range(n):
        if masks[i] is None:
            continue
        for j in range(i + 1, n):
            if masks[j] is None:
                continue
            if mask_iou(masks[i], masks[j]) >= threshold:
                shared[i] = True
                shared[j] = True
    return shared


def looks_like_leak(mask: np.ndarray, panel: Panel, region_area: int) -> bool:
    area = int(np.count_nonzero(mask))
    if area <= 0:
        return True
    px1, py1, px2, py2 = panel
    panel_area = max(0, px2 - px1) * max(0, py2 - py1)
    if panel_area > 0 and area > PANEL_FILL_LEAK * panel_area:
        return True
    # Huge vs OCR is only a leak if floodfill also ran into the panel edge.
    if region_area > 0 and area > LEAK_AREA_RATIO * region_area and _touches_panel_border(mask, panel):
        return True
    return False


def _touches_panel_border(mask: np.ndarray, panel: Panel, thickness: int = 2) -> bool:
    x1, y1, x2, y2 = panel
    h, w = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return True
    t = max(1, thickness)
    strip = np.concatenate([
        mask[y1:min(y1 + t, y2), x1:x2].ravel(),
        mask[max(y2 - t, y1):y2, x1:x2].ravel(),
        mask[y1:y2, x1:min(x1 + t, x2)].ravel(),
        mask[y1:y2, max(x2 - t, x1):x2].ravel(),
    ])
    return bool(np.any(strip))


def estimate_balloon_mask(img: np.ndarray, region: TextBlock) -> Optional[np.ndarray]:
    try:
        crop, xyxy = extract_ballon_region(img, [int(v) for v in region.xywh], enlarge_ratio=ENLARGE_RATIO)
    except Exception as e:
        logger.debug("balloon estimate failed: %s", e)
        return None
    if crop is None or crop.size == 0:
        return None
    fill = float(np.count_nonzero(crop)) / float(crop.size)
    if fill < 0.05:
        return None
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    full = np.zeros(img.shape[:2], dtype=np.uint8)
    ch, cw = crop.shape[:2]
    y2 = min(y1 + ch, full.shape[0])
    x2 = min(x1 + cw, full.shape[1])
    if y2 <= y1 or x2 <= x1:
        return None
    full[y1:y2, x1:x2] = crop[: y2 - y1, : x2 - x1]
    return full


def calculate_centroid_expansion_box(
    cleaned_mask: np.ndarray,
    padding_pixels: float = 4.0,
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[float, float]]]:
    """Inscribed symmetric rectangle inside a bubble mask via distance transform + rays."""
    if cleaned_mask is None or not np.any(cleaned_mask):
        return None
    try:
        padded_mask = np.zeros(
            (cleaned_mask.shape[0] + 2, cleaned_mask.shape[1] + 2), dtype=np.uint8
        )
        padded_mask[1:-1, 1:-1] = cleaned_mask
        distance_map_padded = cv2.distanceTransform(
            padded_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        distance_map = distance_map_padded[1:-1, 1:-1]
        safe_area_mask = (distance_map >= padding_pixels).astype(np.uint8) * 255
        if not np.any(safe_area_mask):
            return None

        moments = cv2.moments(safe_area_mask)
        if moments["m00"] == 0:
            return None
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]

        _, max_val, _, max_loc = cv2.minMaxLoc(distance_map)
        mask_h, mask_w = safe_area_mask.shape
        cx_int = max(0, min(round(centroid_x), mask_w - 1))
        cy_int = max(0, min(round(centroid_y), mask_h - 1))
        dist_at_centroid = distance_map[cy_int, cx_int]
        if dist_at_centroid < max_val * 0.70:
            centroid_x, centroid_y = float(max_loc[0]), float(max_loc[1])

        cx, cy = round(centroid_x), round(centroid_y)
        if cy < 0 or cy >= mask_h or cx < 0 or cx >= mask_w or safe_area_mask[cy, cx] != 255:
            safe_pixels = np.argwhere(safe_area_mask == 255)
            if safe_pixels.size == 0:
                return None
            distances = np.sqrt(
                (safe_pixels[:, 0] - centroid_y) ** 2
                + (safe_pixels[:, 1] - centroid_x) ** 2
            )
            nearest_idx = int(np.argmin(distances))
            cy, cx = int(safe_pixels[nearest_idx][0]), int(safe_pixels[nearest_idx][1])
            centroid_x, centroid_y = float(cx), float(cy)

        left_zeros = np.where(safe_area_mask[cy, 0:cx] == 0)[0]
        dist_to_left = cx - (int(left_zeros.max()) if left_zeros.size else 0)
        right_zeros = np.where(safe_area_mask[cy, cx:] == 0)[0]
        dist_to_right = int(right_zeros.min()) if right_zeros.size else mask_w - cx
        up_zeros = np.where(safe_area_mask[0:cy, cx] == 0)[0]
        dist_to_top = cy - (int(up_zeros.max()) if up_zeros.size else 0)
        down_zeros = np.where(safe_area_mask[cy:, cx] == 0)[0]
        dist_to_bottom = int(down_zeros.min()) if down_zeros.size else mask_h - cy

        min_w = min(dist_to_left, dist_to_right)
        min_h = min(dist_to_top, dist_to_bottom)
        safe_w_base = min_w - 1 if min_w > 1 else min_w
        safe_h_base = min_h - 1 if min_h > 1 else min_h
        max_safe_width = 2 * max(0, int(safe_w_base))
        max_safe_height = 2 * max(0, int(safe_h_base))
        if max_safe_width <= 0 or max_safe_height <= 0:
            return None

        box_x = int(round(centroid_x - max_safe_width / 2.0))
        box_y = int(round(centroid_y - max_safe_height / 2.0))
        if (
            box_x < 0
            or box_y < 0
            or box_x + max_safe_width > mask_w
            or box_y + max_safe_height > mask_h
        ):
            box_x = max(0, min(box_x, mask_w - max_safe_width))
            box_y = max(0, min(box_y, mask_h - max_safe_height))
            if max_safe_width > mask_w or max_safe_height > mask_h:
                return None
        return (box_x, box_y, max_safe_width, max_safe_height), (centroid_x, centroid_y)
    except (cv2.error, ValueError, IndexError, ZeroDivisionError, OverflowError) as e:
        logger.debug("centroid expansion failed: %s", e)
        return None


def resolve_safe_box(
    original_img: np.ndarray,
    region: TextBlock,
    other_regions: Sequence[TextBlock],
    panels: Sequence[Panel],
    balloon_mask: Optional[np.ndarray] = None,
    shared: bool = False,
) -> SafeBox:
    """Pick a layout rectangle for one region. Balloon fill or padded OCR, always panel-clipped."""
    img_h, img_w = original_img.shape[:2]
    panel = panel_for_region(region, panels, img_w, img_h)
    fallback = lambda: padded_ocr_box(region, panel, img_w, img_h)

    if shared or balloon_mask is None or not np.any(balloon_mask):
        return fallback()

    mask = clip_mask_to_panel(balloon_mask, panel)
    mask = subtract_neighbors(mask, region, other_regions)
    if not np.any(mask) or looks_like_leak(mask, panel, ocr_area(region)):
        return fallback()

    expanded = calculate_centroid_expansion_box(mask, padding_px(original_img.shape))
    if expanded is None:
        return fallback()
    (x, y, w, h), _ = expanded
    x, y, w, h = clip_box_to_panel(int(x), int(y), int(w), int(h), panel)
    if w * h < MIN_SAFE_AREA_RATIO * max(ocr_area(region), 1):
        return fallback()
    return SafeBox(x=x, y=y, w=w, h=h, mask=mask, fallback=False, panel=panel)


def load_panels(img: np.ndarray, rtl: bool = True) -> List[Panel]:
    img_h, img_w = img.shape[:2]
    try:
        from ...utils.panel import get_panels_from_array
        raw = get_panels_from_array(img, rtl=rtl)
        panels = [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in raw]
        if panels:
            return panels
    except Exception as e:
        logger.warning("panel detection failed for reflow (%s), using full image", e)
    return [(0, 0, img_w, img_h)]
