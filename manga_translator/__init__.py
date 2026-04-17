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

from .manga_translator import *
