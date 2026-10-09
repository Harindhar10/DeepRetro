"""Throughput metrics for one eval run, read from a vLLM server's ``/metrics``.

vLLM already computes these server-side, so nothing in the LLM call path
changes. Counters and histograms only grow, so a snapshot before and after
the run gives exactly what happened during it. Gauges (running/waiting
requests, KV-cache usage) are instantaneous, so :class:`PerfRun` also polls
them in the background while the run is going.

Everything here is best-effort: if ``/metrics`` cannot be read, the run goes on
and only the client-side throughput is reported.

Examples
--------
>>> metrics_url("http://localhost:8000/v1")
'http://localhost:8000/metrics'
"""

from __future__ import annotations

import threading
import time
from typing import Any

import requests
import structlog

logger = structlog.get_logger(__name__)

POLL_INTERVAL_S = 2.0

#: Histograms reported as p50/p90/mean, each with the vLLM names to try in
#: order (names changed across vLLM versions).
HISTOGRAMS = {
    "ttft": ("vllm:time_to_first_token_seconds",),
    "tpot": (
        "vllm:request_time_per_output_token_seconds",
        "vllm:time_per_output_token_seconds",
        "vllm:inter_token_latency_seconds",
    ),
    "queue": ("vllm:request_queue_time_seconds",),
}
RUNNING = ("vllm:num_requests_running",)
WAITING = ("vllm:num_requests_waiting",)
KV_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
GEN_TOKENS = ("vllm:generation_tokens_total",)
PROMPT_TOKENS = ("vllm:prompt_tokens_total",)
PREEMPTIONS = ("vllm:num_preemptions_total",)

Snapshot = dict[str, Any]


def metrics_url(api_base: str) -> str:
    """Turn an OpenAI-compatible base URL (``.../v1``) into its ``/metrics`` URL.

    Examples
    --------
    >>> metrics_url("http://gpu-box:8000/v1/")
    'http://gpu-box:8000/metrics'
    """
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base}/metrics"


def parse_metrics(text: str) -> Snapshot:
    """Parse Prometheus text into summed values and histogram buckets.

    Samples are summed over labels (model name, DP engine), so a server with
    several engines reports one number per metric.

    Returns
    -------
    dict
        ``{"values": {sample_name: value}, "buckets": {histogram: {le: count}}}``.

    Examples
    --------
    >>> snap = parse_metrics('vllm:num_requests_running{engine="0"} 3.0\\n'
    ...                      'vllm:num_requests_running{engine="1"} 2.0\\n')
    >>> snap["values"]["vllm:num_requests_running"]
    5.0
    """
    from prometheus_client.parser import text_string_to_metric_families

    values: dict[str, float] = {}
    buckets: dict[str, dict[float, float]] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if not sample.name.startswith("vllm:"):
                continue
            if sample.name.endswith("_bucket"):
                hist = buckets.setdefault(sample.name[: -len("_bucket")], {})
                le = float(sample.labels["le"])
                hist[le] = hist.get(le, 0.0) + sample.value
            else:
                values[sample.name] = values.get(sample.name, 0.0) + sample.value
    return {"values": values, "buckets": buckets}


def snapshot(url: str, timeout: float = 5.0) -> Snapshot | None:
    """Fetch and parse ``url``; ``None`` (with a warning) if it fails."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return parse_metrics(response.text)
    except Exception as exc:
        logger.warning("vllm_metrics.unavailable", url=url, error=str(exc))
        return None


def _first(values: dict[str, float], names: tuple[str, ...]) -> float | None:
    """Value of the first of ``names`` present in ``values``."""
    for name in names:
        if name in values:
            return values[name]
    return None


def _counter_delta(before: Snapshot, after: Snapshot, names: tuple[str, ...]) -> float | None:
    end = _first(after["values"], names)
    start = _first(before["values"], names)
    if end is None:
        return None
    return end - (start or 0.0)


def percentile(buckets: dict[float, float], q: float) -> float | None:
    """Approximate the ``q`` quantile from cumulative histogram buckets.

    Interpolates linearly inside the bucket holding the quantile, the same
    way Prometheus' ``histogram_quantile`` does.

    Examples
    --------
    >>> percentile({0.1: 0, 0.2: 10, float("inf"): 10}, 0.5)
    0.15
    """
    bounds = sorted(buckets)
    if not bounds:
        return None
    total = buckets[bounds[-1]]
    if total <= 0:
        return None
    target = q * total
    lower, below = 0.0, 0.0
    for le in bounds:
        count = buckets[le]
        if count >= target:
            if le == float("inf"):
                return lower
            if count == below:
                return le
            return lower + (le - lower) * (target - below) / (count - below)
        lower, below = le, count
    return lower


def histogram_stats(before: Snapshot, after: Snapshot, names: tuple[str, ...]) -> dict[str, float | None]:
    """p50/p90/mean of the observations made between two snapshots."""
    for name in names:
        end = after["buckets"].get(name)
        if end is None:
            continue
        start = before["buckets"].get(name, {})
        delta = {le: count - start.get(le, 0.0) for le, count in end.items()}
        n = after["values"].get(f"{name}_count", 0.0) - before["values"].get(f"{name}_count", 0.0)
        total = after["values"].get(f"{name}_sum", 0.0) - before["values"].get(f"{name}_sum", 0.0)
        return {
            "p50": percentile(delta, 0.5),
            "p90": percentile(delta, 0.9),
            "mean": total / n if n > 0 else None,
        }
    return {"p50": None, "p90": None, "mean": None}


class PerfRun:
    """Measure one timed window of requests against a vLLM server.

    Use as a context manager around the request pool, then call
    :meth:`summary`. Polls the gauges every ``interval`` seconds in a daemon
    thread; one GET of ``/metrics`` per poll, nothing on the request path.
    """

    def __init__(self, api_base: str, interval: float = POLL_INTERVAL_S) -> None:
        self.url = metrics_url(api_base)
        self.interval = interval
        self.samples: list[dict[str, float | None]] = []
        self.before: Snapshot | None = None
        self.after: Snapshot | None = None
        self.wall_s = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def __enter__(self) -> PerfRun:
        self.before = snapshot(self.url)
        if self.before is not None:
            self._thread.start()
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.wall_s = time.perf_counter() - self._started
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        if self.before is not None:
            self.after = snapshot(self.url)

    def _poll(self) -> None:
        while not self._stop.wait(self.interval):
            snap = snapshot(self.url)
            if snap is None:
                continue
            kv = _first(snap["values"], KV_USAGE)
            self.samples.append({
                "t": round(time.perf_counter() - self._started, 2),
                "running": _first(snap["values"], RUNNING),
                "waiting": _first(snap["values"], WAITING),
                "kv_pct": None if kv is None else 100 * kv,
            })

    def _gauge(self, key: str) -> tuple[float | None, float | None]:
        vals = [s[key] for s in self.samples if s[key] is not None]
        if not vals:
            return None, None
        return sum(vals) / len(vals), max(vals)

    def summary(self, n_molecules: int) -> dict[str, float | None]:
        """Flat dict of throughput and latency metrics for the window."""
        wall = self.wall_s or float("nan")
        out: dict[str, float | None] = {
            "wall_s": self.wall_s,
            "n_molecules": n_molecules,
            "mol_per_s": n_molecules / wall,
        }
        if self.before is None or self.after is None:
            return out
        before, after = self.before, self.after
        gen = _counter_delta(before, after, GEN_TOKENS)
        prompt = _counter_delta(before, after, PROMPT_TOKENS)
        out.update({
            "output_tokens": gen,
            "prompt_tokens": prompt,
            "output_tok_per_s": None if gen is None else gen / wall,
            "total_tok_per_s": None if gen is None or prompt is None else (gen + prompt) / wall,
            "preemptions": _counter_delta(before, after, PREEMPTIONS),
        })
        for short, names in HISTOGRAMS.items():
            for stat, value in histogram_stats(before, after, names).items():
                out[f"{short}_{stat}_s"] = value
        for key in ("running", "waiting", "kv_pct"):
            mean, peak = self._gauge(key)
            out[f"{key}_mean"], out[f"{key}_max"] = mean, peak
        return out


def log_to_wandb(
    project: str,
    run_name: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    """Log one run to W&B: config, gauge time-series and summary. Never raises."""
    try:
        import wandb

        run = wandb.init(project=project, name=run_name, config=config)
        for sample in samples:
            run.log({k: v for k, v in sample.items() if k != "t" and v is not None}
                    | {"elapsed_s": sample["t"]})
        run.summary.update({k: v for k, v in summary.items() if v is not None})
        run.finish()
    except Exception as exc:
        logger.warning("vllm_metrics.wandb_failed", error=str(exc))
