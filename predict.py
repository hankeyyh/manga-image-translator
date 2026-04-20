import asyncio
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Keep stdout clean for `cog openapi_schema` JSON parsing.
# Any ANSI escape sequence in import-time logs can break schema decoding.
os.environ.setdefault("NO_COLOR", "1")
os.environ.setdefault("COG_NO_COLOR", "1")
os.environ.setdefault("CLICOLOR", "0")
os.environ.setdefault("PY_COLORS", "0")

from cog import BasePredictor, Input, Path as CogPath
from PIL import Image

# Import only `config` (lightweight). Do not `from manga_translator import MangaTranslator`:
# that would trigger lazy `__getattr__` and load the full pipeline during `cog openapi-schema`.
from manga_translator.config import (
    Config,
    Detector,
    Inpainter,
    Ocr,
    Translator,
    V1_DEFAULT_DETECTOR,
    V1_DEFAULT_INPAINTER,
    V1_DEFAULT_OCR,
    V1_DEFAULT_TRANSLATOR,
)


MODEL_DIR = os.getenv(
    "MANGA_TRANSLATOR_MODEL_DIR",
    str((Path(__file__).resolve().parent / "models").resolve()),
)


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    old_values = {k: os.environ.get(k) for k in updates}
    try:
        for k, v in updates.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, old in old_values.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


class Predictor(BasePredictor):
    def setup(self) -> None:
        from manga_translator.manga_translator import MangaTranslator
        from replicate.prefetch_models import main as prefetch_models_main

        if _to_bool(os.getenv("REPLICATE_PREFETCH_ON_SETUP"), default=False):
            # Cog build stage cannot import project modules; prefetch at runtime setup instead.
            asyncio.run(prefetch_models_main())

        self.translator = MangaTranslator(
            {
                "use_gpu": True,
                "model_dir": MODEL_DIR,
                "verbose": _to_bool(os.getenv("REPLICATE_VERBOSE")),
                "models_ttl": 0,
            }
        )

    def predict(
        self,
        image: CogPath = Input(description="Input manga image"),
        target_lang: str = Input(
            description="Target language code (e.g. ENG, CHS, CHT, KOR)",
            default="ENG",
        ),
        detector: str = Input(default=V1_DEFAULT_DETECTOR.value),
        ocr: str = Input(default=V1_DEFAULT_OCR.value),
        inpainter: str = Input(default=V1_DEFAULT_INPAINTER.value),
        translator: str = Input(default=V1_DEFAULT_TRANSLATOR.value),
        youdao_app_key: str | None = Input(
            description="Youdao app key (required when translator=youdao)",
            default=None,
        ),
        youdao_secret_key: str | None = Input(
            description="Youdao secret key (required when translator=youdao)",
            default=None,
        ),
    ) -> CogPath:
        config = Config()
        config.translator.target_lang = target_lang.upper()
        config.detector.detector = Detector(detector)
        config.ocr.ocr = Ocr(ocr)
        config.inpainter.inpainter = Inpainter(inpainter)
        config.translator.translator = Translator(translator)

        if config.translator.translator == Translator.youdao:
            app_key = youdao_app_key or os.getenv("YOUDAO_APP_KEY")
            secret_key = youdao_secret_key or os.getenv("YOUDAO_SECRET_KEY")
            if not app_key or not secret_key:
                raise ValueError(
                    "translator=youdao requires both youdao_app_key and youdao_secret_key "
                    "(or pre-set YOUDAO_APP_KEY/YOUDAO_SECRET_KEY env vars)."
                )

        started_at = time.perf_counter()
        with Image.open(str(image)) as input_image:
            img = input_image.convert("RGB")
        with _temporary_env(
            {
                "YOUDAO_APP_KEY": youdao_app_key or os.getenv("YOUDAO_APP_KEY"),
                "YOUDAO_SECRET_KEY": youdao_secret_key or os.getenv("YOUDAO_SECRET_KEY"),
            }
        ):
            ctx = asyncio.run(self.translator.translate(img, config))
        elapsed = time.perf_counter() - started_at

        metrics: dict[str, Any] = getattr(ctx, "metrics", {}) or {}
        stage_costs = metrics.get("timing", {})
        print(
            "replicate_predict_done",
            {
                "elapsed_sec": round(elapsed, 3),
                "target_lang": config.translator.target_lang,
                "detector": str(config.detector.detector),
                "ocr": str(config.ocr.ocr),
                "inpainter": str(config.inpainter.inpainter),
                "translator": str(config.translator.translator),
                "stages": stage_costs,
            },
        )

        out_path = Path(tempfile.mkdtemp(prefix="replicate-out-")) / "out.png"
        ctx.result.save(str(out_path))
        return CogPath(str(out_path))
