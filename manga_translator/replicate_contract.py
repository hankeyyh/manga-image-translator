from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from manga_translator.config import (
    V1_DEFAULT_DETECTOR,
    V1_DEFAULT_INPAINTER,
    V1_DEFAULT_OCR,
    V1_DEFAULT_TRANSLATOR,
)

# Replicate V1 keeps a small, synchronous surface for a single image.
DetectorChoice = Literal["default", "dbconvnext", "ctd", "craft", "paddle", "none"]
OcrChoice = Literal["32px", "48px", "48px_ctc", "mocr"]
InpainterChoice = Literal["default", "lama_large", "lama_mpe", "sd", "none", "original"]
TranslatorChoice = Literal[
    "youdao",
    "baidu",
    "deepl",
    "papago",
    "caiyun",
    "chatgpt",
    "chatgpt_2stage",
    "none",
    "original",
    "sakura",
    "deepseek",
    "groq",
    "gemini",
    "gemini_2stage",
    "custom_openai",
    "offline",
    "nllb",
    "nllb_big",
    "sugoi",
    "jparacrawl",
    "jparacrawl_big",
    "m2m100",
    "m2m100_big",
    "mbart50",
    "qwen2",
    "qwen2_big",
]


class ReplicateV1Input(BaseModel):
    image: Path
    target_lang: str = Field(default="ENG", min_length=2)
    detector: DetectorChoice = V1_DEFAULT_DETECTOR.value
    ocr: OcrChoice = V1_DEFAULT_OCR.value
    inpainter: InpainterChoice = V1_DEFAULT_INPAINTER.value
    translator: TranslatorChoice = V1_DEFAULT_TRANSLATOR.value


class ReplicateV1Output(BaseModel):
    image: Path
