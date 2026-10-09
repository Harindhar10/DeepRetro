import pytest

from deepretro.utils.vllm_metrics import (
    PerfRun,
    metrics_url,
    parse_metrics,
    percentile,
)


def _metrics(gen, prompt, preempt, ttft_buckets, ttft_sum, running=0, kv=0.0):
    """vLLM-style /metrics text with two DP engines splitting the counts."""
    lines = []
    for engine in ("0", "1"):
        labels = f'engine="{engine}",model_name="m"'
        lines += [
            "# TYPE vllm:generation_tokens counter",
            f"vllm:generation_tokens_total{{{labels}}} {gen / 2}",
            "# TYPE vllm:prompt_tokens counter",
            f"vllm:prompt_tokens_total{{{labels}}} {prompt / 2}",
            "# TYPE vllm:num_preemptions counter",
            f"vllm:num_preemptions_total{{{labels}}} {preempt / 2}",
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running{{{labels}}} {running / 2}",
            "# TYPE vllm:kv_cache_usage_perc gauge",
            f"vllm:kv_cache_usage_perc{{{labels}}} {kv}",
            "# TYPE vllm:time_to_first_token_seconds histogram",
        ]
        for le, count in ttft_buckets.items():
            lines.append(f'vllm:time_to_first_token_seconds_bucket{{{labels},le="{le}"}} {count / 2}')
        total = list(ttft_buckets.values())[-1]
        lines += [
            f"vllm:time_to_first_token_seconds_count{{{labels}}} {total / 2}",
            f"vllm:time_to_first_token_seconds_sum{{{labels}}} {ttft_sum / 2}",
        ]
    return "\n".join(lines) + "\n"


BEFORE = _metrics(1000, 5000, 0, {"0.1": 10, "0.5": 10, "1.0": 10, "+Inf": 10}, 0.5)
# 100 new requests: 40 in (0, 0.1], 40 in (0.1, 0.5], 20 in (0.5, 1.0].
AFTER = _metrics(11000, 25000, 4, {"0.1": 50, "0.5": 90, "1.0": 110, "+Inf": 110}, 30.5)


def test_metrics_url():
    assert metrics_url("http://h:8000/v1") == "http://h:8000/metrics"
    assert metrics_url("http://h:8000/") == "http://h:8000/metrics"


def test_parse_sums_over_engines():
    snap = parse_metrics(AFTER)
    assert snap["values"]["vllm:generation_tokens_total"] == 11000
    assert snap["buckets"]["vllm:time_to_first_token_seconds"][float("inf")] == 110


def test_percentile_interpolates():
    buckets = {0.1: 40, 0.5: 80, 1.0: 100, float("inf"): 100}
    assert percentile(buckets, 0.5) == pytest.approx(0.2)
    assert percentile(buckets, 0.9) == pytest.approx(0.75)
    assert percentile({float("inf"): 0}, 0.5) is None


def test_summary_from_deltas():
    perf = PerfRun("http://unused/v1")
    perf.before, perf.after = parse_metrics(BEFORE), parse_metrics(AFTER)
    perf.wall_s = 10.0
    perf.samples = [
        {"t": 2.0, "running": 30.0, "waiting": 0.0, "kv_pct": 40.0},
        {"t": 4.0, "running": 50.0, "waiting": 6.0, "kv_pct": 80.0},
    ]
    s = perf.summary(n_molecules=100)
    assert s["mol_per_s"] == 10.0
    assert s["output_tok_per_s"] == 1000.0
    assert s["total_tok_per_s"] == 3000.0
    assert s["preemptions"] == 4
    assert s["ttft_mean_s"] == pytest.approx(0.3)
    assert s["ttft_p50_s"] == pytest.approx(0.1 + 0.4 * 10 / 40)
    assert s["ttft_p90_s"] == pytest.approx(0.5 + 0.5 * 10 / 20)
    assert s["running_mean"] == 40.0 and s["running_max"] == 50.0
    assert s["kv_pct_max"] == 80.0
    assert s["tpot_p50_s"] is None  # not in the fixture


def test_no_server_reports_wall_clock_only():
    with PerfRun("http://127.0.0.1:9/v1", interval=0.05) as perf:
        pass
    s = perf.summary(n_molecules=0)
    assert perf.before is None and perf.samples == []
    assert set(s) == {"wall_s", "n_molecules", "mol_per_s"}
