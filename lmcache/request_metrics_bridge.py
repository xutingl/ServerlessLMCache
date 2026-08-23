import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_request_metrics_logger: Any | None = None
_disabled = False


def _get_request_metrics_logger() -> Any | None:
    global _disabled
    global _request_metrics_logger
    if os.getenv("LMCACHE_REQUEST_METRICS_ENABLED", "true").lower() in (
        "0",
        "false",
        "no",
    ):
        _disabled = True
        return None
    if _disabled:
        return None
    if _request_metrics_logger is not None:
        return _request_metrics_logger

    try:
        from metrics import RequestMetricsLogger

        _request_metrics_logger = RequestMetricsLogger(
            table_name=os.getenv("REQUEST_METRICS_TABLE_NAME", "RequestMetrics"),
            region_name=os.getenv("DYNAMODB_REGION", "us-east-1"),
        )
    except Exception:
        logger.debug("Request metrics logger is unavailable", exc_info=True)
        _disabled = True
        return None

    return _request_metrics_logger


def initialize_lmcache_request_metrics() -> None:
    """Pay process-local logger initialization before serving requests."""
    _get_request_metrics_logger()


def log_lmcache_count_metric(rid: str, name: str, value: Any) -> None:
    metrics_logger = _get_request_metrics_logger()
    if metrics_logger is None:
        return

    try:
        from metrics import RequestMetricType

        if hasattr(value, "item"):
            value = value.item()
        metrics_logger.log(str(rid), name, RequestMetricType.COUNT, int(value))
    except Exception:
        logger.debug("Failed to log LMCache request metric", exc_info=True)


def log_lmcache_duration_series_metric(
    rid: str,
    name: str,
    values_seconds: tuple[float, ...],
) -> None:
    metrics_logger = _get_request_metrics_logger()
    if metrics_logger is None:
        return

    try:
        from metrics import RequestMetricType

        values_ns = [int(value * 1_000_000_000) for value in values_seconds]
        metrics_logger.log(
            str(rid),
            name,
            RequestMetricType.DURATION_SERIES,
            values_ns,
        )
    except Exception:
        logger.debug("Failed to log LMCache request metric", exc_info=True)


def flush_lmcache_request_metrics() -> None:
    """Flush metrics queued in this process at an engine lifecycle boundary."""
    metrics_logger = _request_metrics_logger
    if metrics_logger is None:
        return

    try:
        if metrics_logger.pending_count:
            metrics_logger.flush()
    except Exception:
        logger.debug("Failed to flush LMCache request metrics", exc_info=True)
