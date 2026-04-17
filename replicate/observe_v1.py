import argparse
import datetime as dt
import json
import os
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_API_BASE = "https://api.replicate.com/v1"


@dataclass(frozen=True)
class ObservationSummary:
    total: int
    succeeded: int
    failed: int
    timeout_failures: int
    oom_failures: int
    pending_or_running: int
    avg_latency_sec: float | None
    p95_latency_sec: float | None

    @property
    def failure_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.failed / self.total

    @property
    def timeout_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.timeout_failures / self.total

    @property
    def oom_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.oom_failures / self.total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observe Replicate V1 private version health metrics"
    )
    parser.add_argument("--owner", required=True, help="Replicate owner/org name")
    parser.add_argument("--model", required=True, help="Replicate model name")
    parser.add_argument(
        "--version",
        default=None,
        help="Optional model version id to filter predictions",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="How many recent hours of predictions to inspect (default: 24)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Replicate API page size (default: 100)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum pages to fetch (default: 20)",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="Replicate token override, else use REPLICATE_API_TOKEN",
    )
    return parser.parse_args()


def _parse_dt(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _duration_seconds(prediction: dict[str, Any]) -> float | None:
    started_at = _parse_dt(prediction.get("started_at"))
    completed_at = _parse_dt(prediction.get("completed_at"))
    if started_at is None or completed_at is None:
        return None
    delta = (completed_at - started_at).total_seconds()
    return max(delta, 0.0)


def _contains_timeout(error_msg: str) -> bool:
    msg = error_msg.lower()
    keywords = ("timeout", "timed out", "deadline exceeded")
    return any(word in msg for word in keywords)


def _contains_oom(error_msg: str) -> bool:
    msg = error_msg.lower()
    keywords = (
        "out of memory",
        "cuda out of memory",
        "cudnn_status_not_supported",
        "resource exhausted",
        "oom",
    )
    return any(word in msg for word in keywords)


def _extract_model_name(prediction: dict[str, Any]) -> str | None:
    version_info = prediction.get("version")
    if isinstance(version_info, dict):
        return version_info.get("model")
    return None


def _extract_version_id(prediction: dict[str, Any]) -> str | None:
    version_info = prediction.get("version")
    if isinstance(version_info, dict):
        return version_info.get("id")
    if isinstance(prediction.get("version"), str):
        return prediction.get("version")
    return None


def fetch_predictions(
    *,
    owner: str,
    model: str,
    version: str | None,
    hours: int,
    page_size: int,
    max_pages: int,
    api_token: str,
    api_base: str = DEFAULT_API_BASE,
) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Token {api_token}"}
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
    url = f"{api_base}/predictions?limit={page_size}"
    predictions: list[dict[str, Any]] = []

    for _ in range(max_pages):
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        if not isinstance(results, list) or not results:
            break

        stop_fetch = False
        for prediction in results:
            if not isinstance(prediction, dict):
                continue

            created_at = _parse_dt(prediction.get("created_at"))
            if created_at is not None and created_at < cutoff:
                stop_fetch = True
                continue

            if _extract_model_name(prediction) != f"{owner}/{model}":
                continue

            if version is not None and _extract_version_id(prediction) != version:
                continue

            predictions.append(prediction)

        if stop_fetch:
            break

        next_url = payload.get("next")
        if not next_url:
            break
        url = next_url

    return predictions


def summarize_predictions(predictions: list[dict[str, Any]]) -> ObservationSummary:
    succeeded = 0
    failed = 0
    timeout_failures = 0
    oom_failures = 0
    pending_or_running = 0
    durations: list[float] = []

    for prediction in predictions:
        status = str(prediction.get("status", "")).lower()
        if status == "succeeded":
            succeeded += 1
            duration = _duration_seconds(prediction)
            if duration is not None:
                durations.append(duration)
        elif status in {"failed", "canceled"}:
            failed += 1
            error_msg = str(prediction.get("error") or "")
            if _contains_timeout(error_msg):
                timeout_failures += 1
            if _contains_oom(error_msg):
                oom_failures += 1
        elif status in {"starting", "processing"}:
            pending_or_running += 1

    if durations:
        sorted_durations = sorted(durations)
        p95_idx = max(0, int(len(sorted_durations) * 0.95) - 1)
        p95_latency = sorted_durations[p95_idx]
        avg_latency = sum(sorted_durations) / len(sorted_durations)
    else:
        avg_latency = None
        p95_latency = None

    return ObservationSummary(
        total=len(predictions),
        succeeded=succeeded,
        failed=failed,
        timeout_failures=timeout_failures,
        oom_failures=oom_failures,
        pending_or_running=pending_or_running,
        avg_latency_sec=avg_latency,
        p95_latency_sec=p95_latency,
    )


def phase2_recommendation(summary: ObservationSummary) -> str:
    if summary.total < 30:
        return "样本量不足（<30），建议继续观察再决定二期功能。"

    if summary.oom_rate > 0.01:
        return "OOM 率偏高，先优化显存与模型加载策略，再进入二期。"

    if summary.timeout_rate > 0.02:
        return "超时率偏高，先优化超时参数与并发，再进入二期。"

    if summary.failure_rate > 0.03:
        return "总体失败率偏高，优先稳定性治理，暂缓二期。"

    return "稳定性指标达标，可以开始规划二期（批处理/流式进度/更多翻译器）。"


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    token = args.api_token or os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError("Missing token. Set REPLICATE_API_TOKEN or pass --api-token.")

    predictions = fetch_predictions(
        owner=args.owner,
        model=args.model,
        version=args.version,
        hours=args.hours,
        page_size=args.page_size,
        max_pages=args.max_pages,
        api_token=token,
    )
    summary = summarize_predictions(predictions)
    recommendation = phase2_recommendation(summary)

    report = {
        "scope": {
            "owner": args.owner,
            "model": args.model,
            "version": args.version,
            "hours": args.hours,
        },
        "counts": {
            "total": summary.total,
            "succeeded": summary.succeeded,
            "failed_or_canceled": summary.failed,
            "pending_or_running": summary.pending_or_running,
            "timeout_failures": summary.timeout_failures,
            "oom_failures": summary.oom_failures,
        },
        "rates": {
            "failure_rate": round(summary.failure_rate, 4),
            "timeout_rate": round(summary.timeout_rate, 4),
            "oom_rate": round(summary.oom_rate, 4),
        },
        "latency_sec": {
            "avg": _format_metric(summary.avg_latency_sec),
            "p95": _format_metric(summary.p95_latency_sec),
        },
        "phase2_recommendation": recommendation,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
