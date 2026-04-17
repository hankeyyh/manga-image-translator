import argparse
import asyncio
import statistics
import time
from pathlib import Path

from PIL import Image

from manga_translator import Config, MangaTranslator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate V1 local validator")
    parser.add_argument("--image", required=True, help="Path to one input image")
    parser.add_argument("--target-lang", default="ENG")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--out-dir", default="result/replicate-validation")
    return parser.parse_args()


async def run_once(translator: MangaTranslator, image_path: Path, config: Config) -> float:
    with Image.open(image_path) as input_image:
        img = input_image.convert("RGB")
    start = time.perf_counter()
    ctx = await translator.translate(img, config)
    elapsed = time.perf_counter() - start

    return elapsed, ctx


async def main() -> None:
    args = parse_args()
    image_path = Path(args.image).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    translator = MangaTranslator({"use_gpu": True, "verbose": False})
    config = Config()
    config.translator.target_lang = args.target_lang.upper()

    timings = []
    last_ctx = None
    for i in range(args.runs):
        elapsed, ctx = await run_once(translator, image_path, config)
        print(f"run {i + 1}/{args.runs}: {elapsed:.3f}s")
        timings.append(elapsed)
        last_ctx = ctx

    if last_ctx is not None:
        out_path = out_dir / "final.png"
        last_ctx.result.save(out_path)
        print(f"saved result: {out_path}")

    mean_v = statistics.mean(timings)
    p95_v = sorted(timings)[max(0, int(len(timings) * 0.95) - 1)]
    print(
        "summary:",
        {
            "runs": args.runs,
            "avg_sec": round(mean_v, 3),
            "p95_sec": round(p95_v, 3),
            "detector": str(config.detector.detector),
            "ocr": str(config.ocr.ocr),
            "inpainter": str(config.inpainter.inpainter),
            "translator": str(config.translator.translator),
            "target_lang": config.translator.target_lang,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
