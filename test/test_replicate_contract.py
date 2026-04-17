from pathlib import Path

import pytest
from pydantic import ValidationError

from manga_translator.config import (
    V1_DEFAULT_DETECTOR,
    V1_DEFAULT_INPAINTER,
    V1_DEFAULT_OCR,
    V1_DEFAULT_TRANSLATOR,
)
from manga_translator.replicate_contract import ReplicateV1Input, ReplicateV1Output


def test_replicate_v1_input_contract_defaults():
    payload = ReplicateV1Input(image=Path("demo.png"))

    assert payload.image == Path("demo.png")
    assert payload.target_lang == "ENG"
    assert payload.detector == V1_DEFAULT_DETECTOR.value
    assert payload.ocr == V1_DEFAULT_OCR.value
    assert payload.inpainter == V1_DEFAULT_INPAINTER.value
    assert payload.translator == V1_DEFAULT_TRANSLATOR.value


def test_replicate_v1_input_requires_single_image():
    with pytest.raises(ValidationError):
        ReplicateV1Input()


def test_replicate_v1_output_contract():
    result = ReplicateV1Output(image=Path("translated.png"))
    assert result.image == Path("translated.png")
