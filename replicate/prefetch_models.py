import asyncio
import os
from pathlib import Path

from manga_translator import Config
from manga_translator.detection import prepare as prepare_detection
from manga_translator.inpainting import prepare as prepare_inpainting
from manga_translator.ocr import prepare as prepare_ocr
from manga_translator.translators import prepare as prepare_translation
from manga_translator.utils.inference import ModelWrapper

from .default_weights import (
    REPLICATE_V1_DEFAULT_WEIGHT_TARGETS,
    REPLICATE_V1_WEIGHT_STRATEGY,
)


DEFAULT_MODEL_DIR = os.getenv(
    "MANGA_TRANSLATOR_MODEL_DIR",
    str((Path(__file__).resolve().parent.parent / "models").resolve()),
)


async def main() -> None:
    ModelWrapper._MODEL_DIR = DEFAULT_MODEL_DIR
    config = Config()

    print(f"Prefetching models into: {DEFAULT_MODEL_DIR}")
    print(f"Replicate V1 weight strategy: {REPLICATE_V1_WEIGHT_STRATEGY}")
    print(
        "Default model chain:",
        {
            "detector": str(config.detector.detector),
            "ocr": str(config.ocr.ocr),
            "inpainter": str(config.inpainter.inpainter),
            "translator": str(config.translator.translator),
        },
    )

    await prepare_detection(config.detector.detector)
    await prepare_ocr(config.ocr.ocr, device="cpu")
    await prepare_inpainting(config.inpainter.inpainter, device="cpu")
    await prepare_translation(config.translator.translator_gen)

    print("Expected default weight targets:")
    for target in REPLICATE_V1_DEFAULT_WEIGHT_TARGETS:
        print(f" - {target}")
    print("Prefetch completed.")


if __name__ == "__main__":
    asyncio.run(main())
