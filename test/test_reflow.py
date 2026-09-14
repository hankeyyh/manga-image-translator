import numpy as np
import pytest

from manga_translator.rendering.reflow.constraint import (
    calculate_centroid_expansion_box,
    clip_box_to_panel,
    clip_mask_to_panel,
    looks_like_leak,
    padded_ocr_box,
    panel_for_region,
    shared_balloon_flags,
)
from manga_translator.rendering.reflow.layout import (
    find_breaks,
    join_tokens,
    layout_in_box,
    split_overlong_tokens,
    tokenize,
)
from manga_translator.utils import TextBlock


def _measure(size: int, text: str) -> float:
    return len(text) * size


def test_tokenize_keeps_reading_order():
    tokens = tokenize("one two three", hyphenate=False)
    assert tokens == ["one", "two", "three"]


def test_join_tokens_glues_hyphens():
    assert join_tokens(["hel-", "lo", "world"]) == "hello world"


def test_dp_breaks_are_sequential_not_center_out():
    tokens = "one two three four five six seven eight".split()
    measure = lambda s: len(s)
    lines = find_breaks(tokens, max_width=18, measure=measure, space_width=1)
    assert lines is not None
    assert lines[0].startswith("one")
    assert lines[-1].endswith("eight")
    assert " ".join(lines) == " ".join(tokens)


def test_layout_fills_larger_box_with_larger_font():
    text = "one two three four five six"
    small = layout_in_box(text, 24, 30, _measure, min_size=1, max_size=20, hyphenate=False)
    large = layout_in_box(text, 80, 80, _measure, min_size=1, max_size=20, hyphenate=False)
    assert large.font_size >= small.font_size
    assert " ".join(small.lines) == text
    assert small.lines[0].startswith("one")


def test_overlong_word_is_hyphen_split_instead_of_shrinking_to_fit():
    measure = lambda s: len(s)
    chunks = split_overlong_tokens(["Pneumonoultramicroscopicsilicovolcanoconiosis"], 10, measure)
    assert all(len(c) <= 10 for c in chunks)
    assert "".join(c.rstrip("-") for c in chunks) == "Pneumonoultramicroscopicsilicovolcanoconiosis"
    text = "hello Pneumonoultramicroscopicsilicovolcanoconiosis world"
    layout = layout_in_box(text, 12, 80, _measure, min_size=1, max_size=4, hyphenate=False)
    assert layout.font_size >= 1
    assert any("-" in line for line in layout.lines)


def test_layout_not_diamond_for_uniform_words():
    text = "aa aa aa aa aa aa aa aa aa"
    layout = layout_in_box(text, 18, 80, _measure, min_size=1, max_size=6, hyphenate=False)
    assert len(layout.lines) >= 2
    widths = [len(line) for line in layout.lines]
    if len(widths) >= 3:
        # manga2eng diamond: short / long / short. Sequential wrap keeps early lines full.
        assert widths[0] >= widths[1] * 0.7


def test_centroid_expansion_box_stays_inside_ellipse_axes():
    import cv2

    mask = np.zeros((200, 300), dtype=np.uint8)
    cv2.ellipse(mask, (150, 100), (120, 80), 0, 0, 360, 255, -1)
    result = calculate_centroid_expansion_box(mask, padding_pixels=8)
    assert result is not None
    (x, y, w, h), (cx, cy) = result
    assert abs(cx - 150) < 10
    assert abs(cy - 100) < 10
    assert w < 240 and h < 160
    assert mask[y + h // 2, x] == 255
    assert mask[y + h // 2, x + w - 1] == 255
    assert mask[y, x + w // 2] == 255
    assert mask[y + h - 1, x + w // 2] == 255


def test_padded_ocr_box_clips_to_panel():
    region = TextBlock(
        [[[180, 40], [280, 40], [280, 120], [180, 120]]],
        texts=["a"],
        translation="hello there",
    )
    panel = (0, 0, 200, 200)
    box = padded_ocr_box(region, panel, img_w=400, img_h=200)
    assert box.fallback
    assert box.x >= 0
    assert box.x + box.w <= 200
    assert box.y + box.h <= 200


def test_clip_mask_drops_pixels_outside_panel():
    mask = np.ones((50, 80), dtype=np.uint8) * 255
    clipped = clip_mask_to_panel(mask, (0, 0, 40, 50))
    assert np.any(clipped[:, :40])
    assert not np.any(clipped[:, 40:])


def test_panel_for_region_uses_containing_panel():
    region = TextBlock(
        [[[10, 10], [40, 10], [40, 40], [10, 40]]],
        texts=["a"],
        translation="hi",
    )
    panels = [(0, 0, 100, 100), (100, 0, 200, 100)]
    assert panel_for_region(region, panels, 200, 100) == (0, 0, 100, 100)


def test_shared_balloon_flags_marks_overlap():
    a = np.zeros((12, 12), dtype=np.uint8)
    a[2:10, 2:10] = 255
    b = a.copy()
    c = np.zeros((12, 12), dtype=np.uint8)
    c[0:3, 0:3] = 255
    assert shared_balloon_flags([a, b, c]) == [True, True, False]


def test_looks_like_leak_when_mask_fills_panel():
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    assert looks_like_leak(mask, (0, 0, 100, 100), region_area=20)


def test_large_closed_mask_is_not_a_leak():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    assert not looks_like_leak(mask, (0, 0, 100, 100), region_area=20)


def test_clip_box_to_panel_never_crosses():
    x, y, w, h = clip_box_to_panel(80, 10, 50, 40, (0, 0, 100, 100))
    assert x + w <= 100
    assert x >= 0


@pytest.mark.asyncio
async def test_dispatch_reflow_smoke():
    import cv2
    from manga_translator.rendering.reflow import dispatch_reflow

    img = np.full((240, 320, 3), 30, dtype=np.uint8)
    cv2.ellipse(img, (160, 120), (110, 80), 0, 0, 360, (250, 250, 250), -1)
    region = TextBlock(
        [[[120, 90], [200, 90], [200, 150], [120, 150]]],
        texts=["a", "b"],
        translation="hello there friend",
    )
    region.target_lang = "ENG"
    region.set_font_colors([0, 0, 0], [255, 255, 255])
    region.font_size = 18
    out = await dispatch_reflow(img.copy(), img, [region], hyphenate=False, rtl=True)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)
