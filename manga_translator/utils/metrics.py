"""OTLP metrics for manga-image-translator.

Call ``setup_metrics()`` once per process, then use ``record_*`` /
``create_progress_hook()``. Master and worker are separate processes: each
must set up its own MeterProvider. Instruments no-op until setup succeeds.

OTLP exporter reads ``OTEL_EXPORTER_OTLP_ENDPOINT`` / ``HEADERS`` / ``PROTOCOL``.
Resource ``deployment.environment`` comes from ``ENVIRONMENT``.
The periodic reader exports every 15s; ``shutdown()`` is registered with
``atexit`` so the last interval is flushed on process exit.
"""

from __future__ import annotations

import atexit
import os
import time
from collections.abc import Awaitable, Callable
from threading import Lock
from typing import Any

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Histogram, get_meter_provider, set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

SERVICE_NAME = "manga-image-translator"
METER_NAME = "manga.translator"

PIPELINE_OUTCOMES = frozenset(
    {
        "finished",
        "skip-no-regions",
        "skip-no-text",
        "error-translating",
        "cancelled",
    }
)

# Checkpoints / stream control messages, not timed pipeline stages.
_NON_STAGE_STATES = frozenset(
    {
        "running_pre_translation_hooks",
        "after-translating",
    }
)

_DURATION_BUCKETS_S = (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300)
_EXPORT_INTERVAL_MILLIS = 15000

ProgressHook = Callable[[str, bool], Awaitable[None]]

_lock = Lock()
_initialized = False
_atexit_registered = False
_shut_down = False
_provider: MeterProvider | None = None

_requests: Any = None
_request_duration: Histogram | None = None
_queue_depth: Any = None
_queue_wait: Histogram | None = None
_workers_busy: Any = None
_stage_duration: Histogram | None = None
_pipeline_outcome: Any = None
_api_tokens: Any = None
_api_requests: Any = None
_api_retry: Any = None
_image_failed: Any = None


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def _attrs(**kwargs: str | None) -> dict[str, str]:
    return {key: value for key, value in kwargs.items() if value}


def _is_stage_state(state: str) -> bool:
    if not state or ":" in state:
        return False
    return state not in _NON_STAGE_STATES and state not in PIPELINE_OUTCOMES


def is_enabled() -> bool:
    return _initialized and _requests is not None


def _meter_provider() -> MeterProvider | None:
    if isinstance(_provider, MeterProvider):
        return _provider
    current = get_meter_provider()
    if isinstance(current, MeterProvider):
        return current
    return None


def setup_metrics() -> bool:
    """Install OTLP MeterProvider (if needed) and create instruments.

    Idempotent. Returns True when recording is enabled.
    Skips exporter setup when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset, unless
    a SDK MeterProvider is already global (e.g. ``server/main.py``).
    """
    global _initialized, _provider, _atexit_registered
    global _requests, _request_duration, _queue_depth, _queue_wait
    global _workers_busy, _stage_duration, _pipeline_outcome, _api_tokens
    global _api_requests, _api_retry, _image_failed

    with _lock:
        if _initialized:
            return _requests is not None

        current = get_meter_provider()
        if isinstance(current, MeterProvider):
            _provider = current
        elif _env("OTEL_EXPORTER_OTLP_ENDPOINT"):
            resource = Resource.create(
                {
                    "service.name": SERVICE_NAME,
                    "deployment.environment": _env("ENVIRONMENT", "dev"),
                }
            )
            _provider = MeterProvider(
                metric_readers=[
                    PeriodicExportingMetricReader(
                        OTLPMetricExporter(),
                        export_interval_millis=_EXPORT_INTERVAL_MILLIS,
                    )
                ],
                resource=resource,
            )
            set_meter_provider(_provider)
        else:
            _initialized = True
            return False

        meter = metrics.get_meter(METER_NAME)
        _requests = meter.create_counter(
            "manga.translator.requests",
            unit="{request}",
            description="Translation API requests",
        )
        _request_duration = meter.create_histogram(
            "manga.translator.request.duration",
            unit="s",
            description="End-to-end request duration including queue wait",
            explicit_bucket_boundaries_advisory=list(_DURATION_BUCKETS_S),
        )
        _queue_depth = meter.create_gauge(
            "manga.translator.queue.depth",
            unit="{task}",
            description="Tasks waiting in the in-memory queue",
        )
        _queue_wait = meter.create_histogram(
            "manga.translator.queue.wait.duration",
            unit="s",
            description="Time from enqueue until a worker is acquired",
            explicit_bucket_boundaries_advisory=list(_DURATION_BUCKETS_S),
        )
        _workers_busy = meter.create_gauge(
            "manga.translator.workers.busy",
            unit="{worker}",
            description="Busy translator worker processes",
        )
        _stage_duration = meter.create_histogram(
            "manga.translator.stage.duration",
            unit="s",
            description="Duration of a pipeline stage",
            explicit_bucket_boundaries_advisory=list(_DURATION_BUCKETS_S),
        )
        _pipeline_outcome = meter.create_counter(
            "manga.translator.pipeline.outcome",
            unit="{run}",
            description="Pipeline terminal result",
        )
        _api_tokens = meter.create_counter(
            "manga.translator.api.tokens",
            unit="{token}",
            description="LLM tokens consumed",
        )
        _api_requests = meter.create_counter(
            "manga.translator.api.requests",
            unit="{request}",
            description="External translator API calls",
        )
        _api_retry = meter.create_counter(
            "manga.translator.api.retry",
            unit="{retry}",
            description="Translator API retries",
        )
        _image_failed = meter.create_counter(
            "manga.translator.image.failed",
            unit="{image}",
            description="Per-image pipeline failures",
        )
        if not _atexit_registered:
            atexit.register(shutdown)
            _atexit_registered = True
        _initialized = True
        return True


def flush(timeout_millis: int = 10000) -> bool:
    """Force-export buffered metrics. Used by demo/debug endpoints."""
    setup_metrics()
    provider = _meter_provider()
    if provider is None:
        return False
    return bool(provider.force_flush(timeout_millis=timeout_millis))


def shutdown(timeout_millis: int = 10000) -> bool:
    """Flush then shut down the MeterProvider. Registered with atexit."""
    global _shut_down
    with _lock:
        if _shut_down:
            return True
        provider = _meter_provider()
        _shut_down = True
    if provider is None:
        return False
    provider.shutdown(timeout_millis=timeout_millis)
    return True


def record_request(endpoint: str, outcome: str = "success") -> None:
    setup_metrics()
    if _requests is not None:
        _requests.add(1, _attrs(endpoint=endpoint, outcome=outcome))


def record_request_duration(
    seconds: float, endpoint: str, outcome: str = "success"
) -> None:
    setup_metrics()
    if _request_duration is not None and seconds >= 0:
        _request_duration.record(seconds, _attrs(endpoint=endpoint, outcome=outcome))


def set_queue_depth(depth: int) -> None:
    setup_metrics()
    if _queue_depth is not None:
        _queue_depth.set(depth)


def record_queue_wait(seconds: float, task_type: str = "single") -> None:
    setup_metrics()
    if _queue_wait is not None and seconds >= 0:
        _queue_wait.record(seconds, _attrs(task_type=task_type))


def set_workers_busy(busy: int) -> None:
    setup_metrics()
    if _workers_busy is not None:
        _workers_busy.set(busy)


def record_stage_duration(seconds: float, stage: str) -> None:
    setup_metrics()
    if _stage_duration is not None and seconds >= 0 and stage:
        _stage_duration.record(seconds, _attrs(stage=stage))


def record_pipeline_outcome(outcome: str) -> None:
    setup_metrics()
    if _pipeline_outcome is not None and outcome:
        _pipeline_outcome.add(1, _attrs(outcome=outcome))


def record_api_tokens(tokens: int, provider: str, model: str = "") -> None:
    setup_metrics()
    if _api_tokens is not None and tokens > 0:
        _api_tokens.add(tokens, _attrs(provider=provider, model=model))


def record_api_request(provider: str, status: str = "ok") -> None:
    setup_metrics()
    if _api_requests is not None:
        _api_requests.add(1, _attrs(provider=provider, status=status))


def record_api_retry(provider: str, reason: str) -> None:
    setup_metrics()
    if _api_retry is not None:
        _api_retry.add(1, _attrs(provider=provider, reason=reason))


def record_image_failed(reason: str, stage: str = "") -> None:
    setup_metrics()
    if _image_failed is not None:
        _image_failed.add(1, _attrs(reason=reason, stage=stage))


def create_progress_hook() -> ProgressHook:
    """Hook for ``MangaTranslator.add_progress_hook``.

    Records ``stage.duration`` between consecutive ``_report_progress`` calls
    and ``pipeline.outcome`` on terminal states. Also parses
    ``image_failed:{id}:{reason}``.
    """
    last_state: str | None = None
    last_t: float | None = None

    async def hook(state: str, finished: bool) -> None:
        nonlocal last_state, last_t
        now = time.perf_counter()
        if last_state is not None and last_t is not None and _is_stage_state(last_state):
            record_stage_duration(now - last_t, last_state)

        if state in PIPELINE_OUTCOMES:
            record_pipeline_outcome(state)
        elif state.startswith("image_failed:"):
            parts = state.split(":", 2)
            record_image_failed(parts[2] if len(parts) > 2 else "unknown")

        if finished or state in PIPELINE_OUTCOMES:
            last_state = None
            last_t = None
            return

        last_state = state
        last_t = now

    return hook
