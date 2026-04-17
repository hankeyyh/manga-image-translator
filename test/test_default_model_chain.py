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


def test_v1_default_model_chain_is_frozen():
    conf = Config()

    assert V1_DEFAULT_DETECTOR == Detector.default
    assert V1_DEFAULT_OCR == Ocr.ocr48px
    assert V1_DEFAULT_INPAINTER == Inpainter.lama_large
    assert V1_DEFAULT_TRANSLATOR == Translator.sugoi

    assert conf.detector.detector == V1_DEFAULT_DETECTOR
    assert conf.ocr.ocr == V1_DEFAULT_OCR
    assert conf.inpainter.inpainter == V1_DEFAULT_INPAINTER
    assert conf.translator.translator == V1_DEFAULT_TRANSLATOR
