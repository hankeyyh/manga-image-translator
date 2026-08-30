"""
Modal batch stream 压测脚本。

目标接口：POST {url}/translate/batch/image/stream/web

Stream 帧：[1B status][4B size big-endian][nB data]
  1 过程（image_completed / image_failed / 其它）
  2 整批失败 → stream_error
  3 排队 → 记 queue_seen，不算错误
  4 即将开始 → 与 3 配对算 queue wait
  5 整批完成 → stream_ok
  断连且未见 5 → stream_error
  超过 --timeout → 取消 reader，记 timeout
  HTTP 非 2xx / 无 body → http_fail

输入:
  BASE_URL
  图片目录 / fixture
  config.json          // save.save_to=none，不落盘、不写对象存储
  concurrency          // 同时 in-flight 批次数，这是真正的压测旋钮
  batch_size           // 每发几张图，对齐生产 CONCURRENT_IMAGES=2
  duration             // 墙钟时长
  timeout_per_request  // 单次 stream 上限，必须有；现网 parse 没有超时
  ramp                 // 可选：前 T 秒从 1 升到 concurrency

共享状态:
  stop_at = now + duration
  counters = {
    submitted, http_fail, stream_ok, stream_error,
    image_failed, timeout, queue_seen
  }
  latencies = []         // submit→batch_completed
  queue_waits = []       // 首次 status=3 到 status=4
  *_latencies = []       // 相邻 status=1 阶段之间的墙钟；未完成的阶段不计入

主流程:
  预读图片到内存（避免循环里反复读盘）
  校验 len(images_per_batch) == len(config.image_identifiers)

  启动 concurrency 个 worker 协程（或线程）
  等到时 → 不再发新请求，等 in-flight 收尾（或硬停）
  打印 counters / p50 p95 / 错误分类

worker:
  while now < stop_at:
    batch = 从 fixture 取 batch_size 张（可重复同一批）
    req_id = 新 id
    把 config.image_identifiers 换成本次 req_id，避免服务端串单

    t0 = now
    result = run_one(batch, config, timeout_per_request)
    记录 latency、result.kind
    // 立刻进入下一轮，不要 sleep——空窗会让 Modal 看起来「没负载」

run_one:
  POST multipart:
    images[] = batch
    config   = json

  若 HTTP 非 2xx 或无 body → 记 http_fail，返回

  循环读帧，直到:
    status=5  → stream_ok
    status=2  → stream_error
    status=1 阶段名 → 记下一段到达之间的耗时；image_failed: 记计数并丢弃未完成阶段
    status=3  → 记 queue_seen / 排队时刻
    status=4  → 记开始处理时刻
    超时      → 取消 reader，记 timeout
    连接断开且未见 status=5 → 记 stream_error（对应现网 "stream closed before batch_completed"）

  不要把 status=3 算进错误率；那是容量信号
"""

import argparse
import json
import time
from pathlib import Path
import asyncio
import httpx
import copy

BENCH_DIR = Path(__file__).resolve().parent  # .../manga-image-translator/bench
DEFAULT_IMAGES = BENCH_DIR / "fixtures"
DEFAULT_CONFIG = BENCH_DIR / "configs/no_save.json"

IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}

# stream status=1 的阶段名 → stats 里对应的采样列表
STAGE_TO_STAT = {
    "upscaling": "upscaling_latencies",
    "detection": "detection_latencies",
    "ocr": "ocr_latencies",
    "textline_merge": "textline_merge_latencies",
    "translating": "translation_latencies",
    "inpainting": "inpaint_latencies",
    "rendering": "render_latencies",
    "downscaling": "downscaling_latencies",
}


def existing_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {value}")
    return path


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {value}")
    return path


def positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"value must be positive int: {value}")
    return n


def validate_bench_config(config: object, config_path: Path) -> dict:
    if not isinstance(config, dict):
        raise SystemExit(f"{config_path}: config 必须是 JSON object")

    # 省略 save 时服务端默认 local，会写 final.webp；压测必须显式 none
    save = config.get("save")
    if not isinstance(save, dict):
        raise SystemExit(
            f"{config_path}: 压测必须设置 save.save_to=none"
            f"（省略 save 会默认 local 并写盘）"
        )
    save_to = save.get("save_to")
    if save_to != "none":
        raise SystemExit(
            f"{config_path}: 压测禁止落盘或对象存储，save.save_to 必须是 none，当前={save_to!r}"
        )
    return config


# percent: 0.5 / 0.9 / 0.95
# nearest-rank：排序后取第 ceil(p * n) 个（1-based）
def compute_time_percent(latencies: list[float], percent: float) -> float | None:
    if not latencies:
        return None
    ordered = sorted(latencies)
    n = len(ordered)
    rank = min(n, max(1, (n * percent).__ceil__()))
    return ordered[rank - 1]


def print_time_percent(name: str, samples: list[float]) -> None:
    print(
        "{} p50={:.3f}s p90={:.3f}s p95={:.3f}s (n={})".format(
            name,
            compute_time_percent(samples, 0.5) or 0,
            compute_time_percent(samples, 0.9) or 0,
            compute_time_percent(samples, 0.95) or 0,
            len(samples),
        )
    )


def close_stage(stage_start: tuple[str, float] | None, now: float, stats: dict) -> None:
    if stage_start is None:
        return
    name, t0 = stage_start
    stats[STAGE_TO_STAT[name]].append(now - t0)


def log_req(worker: int, query: int, msg: str) -> None:
    print(f"[w{worker}#{query}] {msg}", flush=True)


# LEARN:
# type 作用 type-checking and type conversions
# 与default配合使用，只有当default是str时，type-converter才会处理default value
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Modal web 接口压测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url", type=str, required=True, help="Modal 或本地 base URL，不要带 path"
    )
    parser.add_argument(
        "--images",
        default=str(DEFAULT_IMAGES),
        type=existing_dir,
        help="图片目录；按 --batch-size 切片，不足则循环复用",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        type=existing_file,
        help="压测 config JSON 路径；必须 save.save_to=none",
    )
    parser.add_argument(
        "--concurrency",
        default=1,
        type=positive_int,
        help="同时 in-flight 的批次数（压测旋钮）",
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        type=positive_int,
        help="每发几张图；建议先对齐生产的 2",
    )
    parser.add_argument(
        "--duration", default=300, type=positive_int, help="墙钟时长，单位秒"
    )
    parser.add_argument(
        "--timeout",
        default=180,
        type=positive_int,
        help="单次 stream 上限秒数；超时记 timeout 并取消 reader",
    )
    return parser.parse_args()


async def worker(
    client: httpx.AsyncClient,
    idx: int,
    args: argparse.Namespace,
    image_batchs: list[list[tuple[str, bytes]]],
    config: dict,
    stats: dict,
):
    # LEARN: monotonic 起点未定义（可能是开机，某个epoch），因此只有两次调用结果之间的差值才是有效的
    stop_at = time.monotonic() + args.duration
    i = idx
    while time.monotonic() < stop_at:
        batch = image_batchs[i % len(image_batchs)]
        # idx 区分 worker，i 单调递增区分同一 worker 的各轮请求；
        # 不要用 i % n，转完一轮 fixture 后会撞上上一轮的 id
        config["image_identifiers"] = [
            f"{idx}-{i}-{image_idx}" for image_idx in range(len(batch))
        ]
        names = ",".join(name for name, _ in batch)
        remain = max(0.0, stop_at - time.monotonic())
        log_req(
            idx,
            i,
            f"start n={len(batch)} images={names} ids={','.join(config['image_identifiers'])} remain={remain:.0f}s",
        )
        t0 = time.monotonic()
        kind = await run_once(client, args, batch, config, stats, idx, i)
        log_req(idx, i, f"finish {kind} wall={time.monotonic() - t0:.1f}s")
        i += 1


def _decode_frame(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _since(t0: float, now: float | None = None) -> str:
    return f"+{(now if now is not None else time.monotonic()) - t0:.1f}s"


async def run_once(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    images: list[tuple[str, bytes]],
    config: dict,
    stats: dict,
    worker_idx: int,
    query_i: int,
) -> str:
    url = f'{args.url.rstrip("/")}/translate/batch/image/stream/web'
    got_body = False
    try:
        async with asyncio.timeout(args.timeout):
            submit_at = time.monotonic()
            queue_start = 0
            # 当前进行中的阶段：(stream 阶段名, 开始时刻)；超时/失败时丢弃，不算残缺样本
            stage_start: tuple[str, float] | None = None
            stats["counters"]["submitted"] += 1
            async with client.stream(
                "POST",
                url,
                data={"config": json.dumps(config)},
                files=[("images", (name, data)) for name, data in images],
            ) as response:
                if response.status_code != 200:
                    stats["counters"]["http_fail"] += 1
                    log_req(
                        worker_idx,
                        query_i,
                        f"{_since(submit_at)} http_fail status={response.status_code}",
                    )
                    return "http_fail"

                # bytes 不能修改，如果你循环拼接大量字节，不要用 b1+b2（每次生成新对象），优先用 bytearray 累积，最后转 bytes。
                buf = bytearray()

                async for chunk in response.aiter_bytes():
                    if len(chunk) > 0:
                        got_body = True
                    buf.extend(chunk)
                    while len(buf) >= 5:
                        status = buf[0]
                        size = int.from_bytes(buf[1:5], "big")
                        framesize = 5 + size
                        if len(buf) < framesize:
                            break
                        data = buf[5:framesize]
                        buf = buf[framesize:]
                        if (
                            status == 1
                        ):  # 过程：阶段名 / image_failed / image_completed / 其它
                            now = time.monotonic()
                            if data.startswith(b"image_failed:"):
                                stats["counters"]["image_failed"] += 1
                                stage_start = None
                                log_req(
                                    worker_idx,
                                    query_i,
                                    f"{_since(submit_at, now)} {_decode_frame(data)}",
                                )
                                continue
                            state = _decode_frame(data)
                            if stage_start is not None:
                                prev, t0 = stage_start
                                log_req(
                                    worker_idx,
                                    query_i,
                                    f"{_since(submit_at, now)} {state} ({prev}={now - t0:.2f}s)",
                                )
                            else:
                                log_req(
                                    worker_idx,
                                    query_i,
                                    f"{_since(submit_at, now)} {state}",
                                )
                            close_stage(stage_start, now, stats)
                            stage_start = (
                                (state, now) if state in STAGE_TO_STAT else None
                            )
                            continue
                        if status == 2:  # 整体异常报错
                            stats["counters"]["stream_error"] += 1
                            log_req(
                                worker_idx,
                                query_i,
                                f"{_since(submit_at)} stream_error {_decode_frame(data)}",
                            )
                            return "stream_error"
                        if status == 3:  # 排队中
                            pos = _decode_frame(data)
                            if queue_start == 0:
                                stats["counters"]["queue_seen"] += 1
                                queue_start = time.monotonic()
                                log_req(
                                    worker_idx,
                                    query_i,
                                    f"{_since(submit_at, queue_start)} queue pos={pos}",
                                )
                            else:
                                log_req(
                                    worker_idx,
                                    query_i,
                                    f"{_since(submit_at)} queue pos={pos}",
                                )
                            continue
                        if status == 4:  # 即将开始
                            now = time.monotonic()
                            if queue_start != 0:  # 排队时间
                                queue_wait = now - queue_start
                                stats["queue_waits"].append(queue_wait)
                                client_pending = now - submit_at
                                stats["client_pending"].append(client_pending)
                                log_req(
                                    worker_idx,
                                    query_i,
                                    f"{_since(submit_at, now)} processing queue_wait={queue_wait:.2f}s, client_pending={client_pending:.2f}s",
                                )
                            else:
                                log_req(
                                    worker_idx,
                                    query_i,
                                    f"{_since(submit_at, now)} processing",
                                )
                            continue
                        if status == 5:  # 整体翻译完成
                            now = time.monotonic()
                            if stage_start is not None:
                                prev, t0 = stage_start
                                log_req(
                                    worker_idx,
                                    query_i,
                                    f"{_since(submit_at, now)} batch_completed ({prev}={now - t0:.2f}s) latency={now - submit_at:.2f}s",
                                )
                            else:
                                log_req(
                                    worker_idx,
                                    query_i,
                                    f"{_since(submit_at, now)} batch_completed latency={now - submit_at:.2f}s",
                                )
                            close_stage(stage_start, now, stats)
                            stats["counters"]["stream_ok"] += 1
                            stats["latencies"].append(now - submit_at)
                            return "ok"
                if got_body:
                    stats["counters"]["stream_error"] += 1
                    log_req(
                        worker_idx,
                        query_i,
                        f"{_since(submit_at)} stream_error closed before batch_completed",
                    )
                    return "stream_error"
                stats["counters"]["http_fail"] += 1
                log_req(worker_idx, query_i, f"{_since(submit_at)} http_fail no body")
                return "http_fail"
    except httpx.HTTPError as e:
        if got_body:
            stats["counters"]["stream_error"] += 1
            log_req(worker_idx, query_i, f"stream_error {e}")
            return "stream_error"
        stats["counters"]["http_fail"] += 1
        log_req(worker_idx, query_i, f"http_fail {e}")
        return "http_fail"
    except TimeoutError:
        stats["counters"]["timeout"] += 1
        log_req(worker_idx, query_i, f"timeout after {args.timeout}s")
        return "timeout"


async def benchpress(args, image_batchs: list[list[tuple[str, bytes]]], config: dict):
    concurrency: int = args.concurrency
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )

    stats = {
        "counters": {
            "submitted": 0,
            "http_fail": 0,
            "stream_ok": 0,
            "stream_error": 0,
            "image_failed": 0,
            "timeout": 0,
            "queue_seen": 0,
        },
        "latencies": [],
        "client_pending": [],  # 从提交->开始翻译
        "queue_waits": [],
        "upscaling_latencies": [],
        "detection_latencies": [],
        "ocr_latencies": [],
        "textline_merge_latencies": [],
        "translation_latencies": [],
        "inpaint_latencies": [],
        "render_latencies": [],
        "downscaling_latencies": [],
    }

    # args.timeout 决定stream api整体处理时间，在run_once中设置。不需要 connect/read/write timeout
    try:
        async with httpx.AsyncClient(limits=limits, timeout=None) as client:
            workers = [
                worker(client, i, args, image_batchs, copy.deepcopy(config), stats)
                for i in range(concurrency)
            ]
            await asyncio.gather(*workers)
    except asyncio.CancelledError:
        print("Shutdown by Ctrl+C")
    finally:
        counters = stats["counters"]
        print(counters)
        submitted = counters["submitted"]
        # 请求级错误：HTTP 失败 / stream 失败 / 超时。queue_seen 是容量信号，不算错误。
        n_err = counters["http_fail"] + counters["stream_error"] + counters["timeout"]
        error_rate = (n_err / submitted * 100) if submitted else 0
        print(
            "error_rate={:.2f}% (http_fail+stream_error+timeout={}, submitted={})".format(
                error_rate, n_err, submitted
            )
        )
        print_time_percent("whole_latency", stats["latencies"])
        print_time_percent("client_pending", stats["client_pending"])
        print_time_percent("queue_wait", stats["queue_waits"])
        for stage, key in STAGE_TO_STAT.items():
            print_time_percent(stage, stats[key])


if __name__ == "__main__":
    args = parse_arguments()
    image_dir: Path = args.images
    batch_size: int = args.batch_size

    image_paths: list[Path] = []
    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIX:
            image_paths.append(p)

    if len(image_paths) == 0:
        raise SystemExit(f"no images in {image_dir}")

    # 加载图片bytes
    image_paths = sorted(image_paths)
    images = [(p.name, p.read_bytes()) for p in image_paths]

    image_batchs: list[list[tuple[str, bytes]]] = []
    image_len = len(images)
    for i in range(0, image_len, batch_size):
        # LEARN: append([xxx]) 有方框号，立刻算出全部元素，填入的才是list。否则是 generator expression
        image_batchs.append([images[(i + j) % image_len] for j in range(batch_size)])

    # 加载配置
    with args.config.open(encoding="utf-8") as f:
        config = json.load(f)

    config = validate_bench_config(config, args.config)

    asyncio.run(benchpress(args, image_batchs, config))
