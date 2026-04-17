from replicate.default_weights import (
    REPLICATE_V1_DEFAULT_WEIGHT_TARGETS,
    REPLICATE_V1_WEIGHT_STRATEGY,
)


def test_replicate_v1_weight_strategy_is_build_time_prefetch():
    assert REPLICATE_V1_WEIGHT_STRATEGY == "build_time_prefetch"


def test_replicate_v1_default_weight_targets_complete():
    expected = {
        "detect-20241225.ckpt",
        "ocr_ar_48px.ckpt",
        "alphabet-all-v7.txt",
        "lama_large_512px.ckpt",
        "jparacrawl/spm.ja.nopretok.model",
        "jparacrawl/spm.en.nopretok.model",
        "jparacrawl/big-ja-en",
        "jparacrawl/big-en-ja",
        "sugoi/spm.ja.nopretok.model",
        "sugoi/spm.en.nopretok.model",
        "sugoi/big-ja-en",
    }
    assert set(REPLICATE_V1_DEFAULT_WEIGHT_TARGETS) == expected

