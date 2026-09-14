from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

MeasureFn = Callable[[int, str], float]
CollidesFn = Callable[["Layout"], bool]


@dataclass
class Layout:
    font_size: int
    lines: List[str]
    block_w: float
    block_h: float
    line_widths: List[float] = field(default_factory=list)
    line_height: float = 0.0


def tokenize(text: str, hyphenate: bool = False, language: str = "en_US") -> List[str]:
    """Split text into wrap tokens. Hyphenated syllables end with '-' except the last."""
    text = " ".join(text.split())
    if not text:
        return []
    words = text.split(" ")
    if not hyphenate:
        return words
    try:
        from ..text_render import select_hyphenator
        hyphenator = select_hyphenator(language)
    except Exception:
        hyphenator = None
    if hyphenator is None:
        return words
    tokens: List[str] = []
    for word in words:
        syllables: List[str] = []
        if len(word) <= 100:
            try:
                syllables = hyphenator.syllables(word) or []
            except Exception:
                syllables = []
        if len(syllables) <= 1:
            tokens.append(word)
            continue
        for syl in syllables[:-1]:
            tokens.append(syl + "-")
        tokens.append(syllables[-1])
    return tokens


def _needs_space(prev: str, _curr: str) -> bool:
    return bool(prev) and not prev.endswith("-")


def join_tokens(tokens: Sequence[str]) -> str:
    """Join wrap tokens. A trailing '-' marks a hyphenation point and is dropped inside a line."""
    if not tokens:
        return ""
    out = tokens[0]
    for token in tokens[1:]:
        if out.endswith("-"):
            out = out[:-1] + token
        else:
            out += " " + token
    return out


def find_breaks(
    tokens: Sequence[str],
    max_width: float,
    measure: Callable[[str], float],
    space_width: float,
    hyphen_penalty: float = 1000.0,
    badness_exponent: float = 3.0,
) -> Optional[List[str]]:
    """Knuth-Plass style DP. Returns None if a token cannot fit on its own line."""
    if not tokens:
        return []
    widths = [measure(t) for t in tokens]
    if any(w > max_width for w in widths):
        return None

    n = len(tokens)
    min_cost = [float("inf")] * (n + 1)
    path = [0] * (n + 1)
    min_cost[0] = 0.0

    for i in range(1, n + 1):
        line_width = 0.0
        for j in range(i - 1, -1, -1):
            if j < i - 1 and _needs_space(tokens[j], tokens[j + 1]):
                line_width += space_width
            line_width += widths[j]
            if line_width > max_width:
                break
            slack = max_width - line_width
            badness = slack ** badness_exponent
            if tokens[i - 1].endswith("-"):
                badness += hyphen_penalty
            total = min_cost[j] + badness
            if total < min_cost[i]:
                min_cost[i] = total
                path[i] = j

    if min_cost[n] == float("inf"):
        return None

    lines: List[str] = []
    cur = n
    while cur > 0:
        prev = path[cur]
        lines.insert(0, join_tokens(tokens[prev:cur]))
        cur = prev
    return lines


def greedy_breaks(
    tokens: Sequence[str],
    max_width: float,
    measure: Callable[[str], float],
    space_width: float,
    allow_overflow: bool = False,
) -> Optional[List[str]]:
    """Left-to-right wrap. If allow_overflow, an overlong token sits on its own line."""
    if not tokens:
        return []
    lines: List[str] = []
    current: List[str] = []
    current_w = 0.0
    for token in tokens:
        tw = measure(token)
        extra = space_width if current and _needs_space(current[-1], token) else 0.0
        if current and current_w + extra + tw > max_width:
            lines.append(join_tokens(current))
            current = [token]
            current_w = tw
            continue
        if not current and tw > max_width and not allow_overflow:
            return None
        current.append(token)
        current_w += extra + tw
    if current:
        lines.append(join_tokens(current))
    return lines


def _line_height(font_size: int, line_spacing: float) -> float:
    return font_size + font_size * line_spacing


def _block_size(
    lines: Sequence[str],
    font_size: int,
    measure: MeasureFn,
    line_spacing: float,
) -> tuple[List[float], float, float, float]:
    line_h = _line_height(font_size, line_spacing)
    widths = [float(measure(font_size, line)) for line in lines]
    block_w = max(widths) if widths else 0.0
    block_h = line_h * len(lines) if lines else 0.0
    if len(lines) > 1:
        # line_height already includes spacing; n lines occupy n * line_h - trailing spacing
        block_h = font_size * len(lines) + font_size * line_spacing * (len(lines) - 1)
    return widths, block_w, block_h, line_h


def split_overlong_tokens(
    tokens: Sequence[str],
    max_width: float,
    measure: Callable[[str], float],
) -> List[str]:
    """Hyphen-split any token wider than max_width so DP can still wrap at this size."""
    fitted: List[str] = []
    for token in tokens:
        if measure(token) <= max_width:
            fitted.append(token)
            continue
        fitted.extend(_hyphen_chunks(token, max_width, measure))
    return fitted


def _hyphen_chunks(
    token: str,
    max_width: float,
    measure: Callable[[str], float],
) -> List[str]:
    if measure(token) <= max_width:
        return [token]
    chunks: List[str] = []
    rest = token
    while rest:
        took = False
        for i in range(len(rest), 0, -1):
            piece = rest[:i]
            more = i < len(rest)
            candidate = piece + ("-" if more else "")
            if measure(candidate) <= max_width or i == 1:
                if more:
                    chunks.append(piece + "-")
                    rest = rest[i:]
                else:
                    chunks.append(piece)
                    rest = ""
                took = True
                break
        if not took:
            chunks.append(rest)
            break
    return chunks or [token]


def _wrap_at_size(
    tokens: Sequence[str],
    font_size: int,
    max_width: float,
    measure: MeasureFn,
    allow_overflow: bool = False,
) -> Optional[List[str]]:
    word_measure = lambda s: measure(font_size, s)
    space_w = measure(font_size, " ")
    wrap_tokens = split_overlong_tokens(tokens, max_width, word_measure)
    lines = find_breaks(wrap_tokens, max_width, word_measure, space_w)
    if lines is None:
        lines = greedy_breaks(wrap_tokens, max_width, word_measure, space_w, allow_overflow=allow_overflow)
    return lines


def layout_in_box(
    text: str,
    box_w: int,
    box_h: int,
    measure: MeasureFn,
    min_size: int,
    max_size: int,
    line_spacing: float = 0.01,
    hyphenate: bool = False,
    language: str = "en_US",
    collides: Optional[CollidesFn] = None,
    max_squeezes: int = 3,
) -> Layout:
    """Largest font size whose sequential wrap fits in (box_w, box_h).

    If even min_size cannot fit, still return a min_size layout (may overflow the box).
    """
    text = " ".join((text or "").split())
    min_size = max(int(min_size), 1)
    max_size = max(int(max_size), min_size)
    box_w = max(int(box_w), 1)
    box_h = max(int(box_h), 1)

    if not text:
        return Layout(font_size=min_size, lines=[], block_w=0, block_h=0)

    tokens = tokenize(text, hyphenate=hyphenate, language=language)
    squeezes = max_squeezes if collides is not None else 1
    best: Optional[Layout] = None

    lo, hi = min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        fitted = _try_size(
            tokens, mid, box_w, box_h, measure, line_spacing, collides, squeezes, False
        )
        if fitted is not None:
            best = fitted
            lo = mid + 1
        else:
            hi = mid - 1

    if best is not None:
        return best

    overflow = _try_size(
        tokens, min_size, box_w, box_h, measure, line_spacing, None, 1, True
    )
    if overflow is not None:
        return overflow
    widths, block_w, block_h, line_h = _block_size([text], min_size, measure, line_spacing)
    return Layout(
        font_size=min_size,
        lines=[text],
        block_w=block_w,
        block_h=block_h,
        line_widths=widths,
        line_height=line_h,
    )


def _try_size(
    tokens: Sequence[str],
    font_size: int,
    box_w: int,
    box_h: int,
    measure: MeasureFn,
    line_spacing: float,
    collides: Optional[CollidesFn],
    max_squeezes: int,
    allow_overflow: bool,
) -> Optional[Layout]:
    width_attempt = float(box_w)
    for _ in range(max(max_squeezes, 1)):
        lines = _wrap_at_size(tokens, font_size, width_attempt, measure, allow_overflow)
        if not lines:
            return None
        widths, block_w, block_h, line_h = _block_size(lines, font_size, measure, line_spacing)
        if not allow_overflow and (block_h > box_h or block_w > box_w + 1e-6):
            # narrower wrap cannot reduce height if we already overflow height at this width
            if block_h > box_h:
                return None
            break
        layout = Layout(
            font_size=font_size,
            lines=list(lines),
            block_w=block_w,
            block_h=block_h,
            line_widths=widths,
            line_height=line_h,
        )
        if collides is not None and collides(layout):
            width_attempt *= 0.90
            continue
        return layout
    return None
