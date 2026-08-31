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

报告：
GPU: A10, CPU: 4, Memory: 16GB
此配置下 8worker, 内存负载 12GB。如果 4worker, 内存负载 8GB。
下面的数据在单台容器 4worker, timeout=60s下测得:
- concurrency 16, batch-size 2。翻译流程:8s, timeout错误率:26, steramerr错误率:0, latency:47
- concurrency 8, batch-size 2。 翻译流程:8s, timeout错误率:0, steramerr错误率:0, latency:23
- concurrency 4, batch-size 2。 翻译流程:8s, timeout错误率:0, steramerr错误率:0, latency:15

-- concurrency 8, batch-size 4。翻译流程:8s, timeout错误率:0, steramerr错误率:0, latency:41
-- concurrency 4, batch-size 4。翻译流程:8s, timeout错误率:0, steramerr错误率:0, latency:26

-- concurrency 4, batch-size 8。翻译流程:8s, timeout错误率:0, steramerr错误率:0, latency:46

期望stream latency在30s左右, 单次容量尽可能大。最后决定 max_inputs=8, batch-size=4
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
LOCAL_URL = "http://127.0.0.1:8000"
REMOTE_URL = "https://hankeyyh--manga-translator-web.modal.run"

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
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--local",
        action="store_true",  # 命令行里出现它就是 True，不出现就是 False
        help=f"打本地 {LOCAL_URL}",
    )
    target.add_argument(
        "--remote",
        action="store_true",
        help=f"打 Modal {REMOTE_URL}",
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
        default=2,
        type=positive_int,
        help="每发几张图；建议先对齐生产的 2",
    )
    parser.add_argument(
        "--duration", default=300, type=positive_int, help="墙钟时长，单位秒"
    )
    parser.add_argument(
        "--timeout",
        default=60,
        type=positive_int,
        help="单次 stream 上限秒数；超时记 timeout 并取消 reader",
    )
    args = parser.parse_args()
    args.url = LOCAL_URL if args.local else REMOTE_URL
    return args


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
                # stream() 进入时 HTTP 响应头已到；body 第一帧还没读。
                # header = 上传 + 冷启动 + 服务端 Image.open，直到 return StreamingResponse。
                header_at = time.monotonic()
                header_latency = header_at - submit_at
                stats["header_latencies"].append(header_latency)
                # 服务端 ASGI 量的 request body 首块→末块；uvicorn 合成一块时会≈0，这时看 header。
                upload_s = None
                upload_hdr = response.headers.get("x-upload-seconds")
                if upload_hdr:
                    try:
                        upload_s = float(upload_hdr)
                        stats["upload_seconds"].append(upload_s)
                    except ValueError:
                        pass
                upload_note = f" upload={upload_s:.3f}s" if upload_s is not None else ""
                log_req(
                    worker_idx,
                    query_i,
                    f"{_since(submit_at, header_at)} header status={response.status_code}{upload_note}",
                )
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
                                log_req(
                                    worker_idx,
                                    query_i,
                                    f"{_since(submit_at, now)} processing queue_wait={queue_wait:.2f}s header={header_latency:.2f}s",
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
            "submitted": 0,  # 发出的批次数（含随后失败的）
            "http_fail": 0,  # HTTP 非 2xx / 无 body / 读到第一帧前断连
            "stream_ok": 0,  # 收到 status=5，整批完成
            "stream_error": 0,  # status=2，或断连且未见 5
            "image_failed": 0,  # status=1 且 image_failed:；请求可能仍继续
            "timeout": 0,  # 超过 --timeout，取消 reader
            "queue_seen": 0,  # 见过 status=3；容量信号，不算错误
        },
        "latencies": [],  # 整段墙钟：submit → status=5
        "header_latencies": [],  # submit → HTTP 响应头
        "upload_seconds": [],  # 服务端 X-Upload-Seconds：ASGI 读完 request body
        "queue_waits": [],  # 首次 status=3 → status=4
        "upscaling_latencies": [],  # status=1 阶段 upscaling 持续时长
        "detection_latencies": [],  # status=1 阶段 detection
        "ocr_latencies": [],  # status=1 阶段 ocr
        "textline_merge_latencies": [],  # status=1 阶段 textline_merge
        "translation_latencies": [],  # status=1 阶段 translating（每批一次）
        "inpaint_latencies": [],  # status=1 阶段 inpainting
        "render_latencies": [],  # status=1 阶段 rendering
        "downscaling_latencies": [],  # status=1 阶段 downscaling
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
        print("[counters]", counters)
        print("[concurrency]", concurrency)
        submitted = counters["submitted"]
        # 请求级错误：HTTP 失败 / stream 失败 / 超时。queue_seen 是容量信号，不算错误。
        n_err = counters["http_fail"] + counters["stream_error"] + counters["timeout"]
        error_rate = (n_err / submitted * 100) if submitted else 0
        print(
            "[error_rate]={:.2f}% (http_fail+stream_error+timeout={}, submitted={})".format(
                error_rate, n_err, submitted
            )
        )
        print_time_percent("[whole_latency]", stats["latencies"])
        print_time_percent("[header]", stats["header_latencies"])
        print_time_percent("[upload]", stats["upload_seconds"])
        print_time_percent("[queue_wait]", stats["queue_waits"])
        for stage, key in STAGE_TO_STAT.items():
            print_time_percent(f"[{stage}]", stats[key])


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
