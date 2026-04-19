import importlib
import colorama
import os
import sys
from dotenv import load_dotenv

# Avoid ANSI escape sequences when stdout is non-interactive (e.g. cog openapi_schema),
# otherwise Cog may fail to parse JSON output with "\x1b" errors.
_NO_COLOR = (
    os.getenv("NO_COLOR") is not None
    or os.getenv("COG_NO_COLOR") is not None
    or not sys.stdout.isatty()
)
colorama.init(autoreset=not _NO_COLOR, strip=_NO_COLOR)
load_dotenv()

# Eager export: `from manga_translator import Config` must not use lazy __getattr__,
# or it can re-enter while `manga_translator.manga_translator` is still loading and
# cause RecursionError (e.g. Replicate worker setup + prefetch_models).
from .config import Config

# Do not `from .manga_translator import *` here: that eagerly loads the full pipeline
# (detection/ocr/inpainting/...) and breaks tools that import lightweight submodules
# or need clean stdout (e.g. `cog openapi-schema` JSON parsing).

_in_getattr = False


def __getattr__(name: str):
    """Lazy re-exports from `manga_translator.manga_translator` for backward compatibility."""
    global _in_getattr
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _in_getattr:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} (re-entrant import; avoid "
            f"`from manga_translator import ...` for names defined only in manga_translator.py "
            f"while that module is still loading)"
        )
    _in_getattr = True
    try:
        # Must use importlib: `from . import manga_translator` inside __getattr__ would look up the
        # attribute `manga_translator` on this package and recurse forever (e.g. translators/chatgpt.py:
        # `from .. import manga_translator`).
        _mt = importlib.import_module(f"{__name__}.manga_translator")
        if name == "manga_translator":
            return _mt
        try:
            return getattr(_mt, name)
        except AttributeError as e:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from e
    finally:
        _in_getattr = False


