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

# Do not `from .manga_translator import *` here: that eagerly loads the full pipeline
# (detection/ocr/inpainting/...) and breaks tools that import lightweight submodules
# or need clean stdout (e.g. `cog openapi-schema` JSON parsing).


def __getattr__(name: str):
    """Lazy re-exports from `manga_translator.manga_translator` for backward compatibility."""
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
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


