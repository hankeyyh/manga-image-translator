from __future__ import annotations

# Replicate V1 strategy decision:
# prefer build-time prefetch to minimize first-request latency.
REPLICATE_V1_WEIGHT_STRATEGY = "build_time_prefetch"

# Default V1 chain:
# detector=default, ocr=48px, inpainter=lama_large, translator=sugoi
# The list below includes all expected files/directories under model_dir after prefetch.
REPLICATE_V1_DEFAULT_WEIGHT_TARGETS = (
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
)

