import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from cog import BasePredictor, Input, Path as CogPath
from PIL import Image

from manga_translator import Config, MangaTranslator
from manga_translator.config import (
    Detector,
    Inpainter,
    Ocr,
    Translator,
    V1_DEFAULT_DETECTOR,
    V1_DEFAULT_INPAINTER,
    V1_DEFAULT_OCR,
    V1_DEFAULT_TRANSLATOR,
)
from replicate.prefetch_models import main as prefetch_models_main


MODEL_DIR = os.getenv(
    "MANGA_TRANSLATOR_MODEL_DIR",
    str((Path(__file__).resolve().parent / "models").resolve()),
)


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Predictor(BasePredictor):
    def setup(self) -> None:
        if _to_bool(os.getenv("REPLICATE_PREFETCH_ON_SETUP"), default=True):
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
    ) -> CogPath:
        config = Config()
        config.translator.target_lang = target_lang.upper()
        config.detector.detector = Detector(detector)
        config.ocr.ocr = Ocr(ocr)
        config.inpainter.inpainter = Inpainter(inpainter)
        config.translator.translator = Translator(translator)

        started_at = time.perf_counter()
        with Image.open(str(image)) as input_image:
            img = input_image.convert("RGB")
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
