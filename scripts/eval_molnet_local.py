#!/usr/bin/env python
"""MoleculeNet property prediction with a local model through DeepRetro.

The local-model counterpart of ``Claude_GPT_eval/run_eval.py``: same datasets,
prompts, in-context examples, ``--limit`` sample and ROC-AUC/RMSE scoring (all
from the shared ``molnet`` package in the DFS folder), but every request goes
through DeepRetro's LLM interface to a vLLM OpenAI-compatible server.
Start the server first, e.g.::

    vllm serve zai-org/GLM-4.7-Flash --dtype bfloat16 --max-model-len 17408

then::

    HOSTED_VLLM_API_BASE=http://localhost:8000/v1 \\
    python scripts/eval_molnet_local.py \\
        --model hosted_vllm/zai-org/GLM-4.7-Flash --no-thinking --limit 50

For thinking runs, serve with ``--reasoning-parser`` (``qwen3`` for Qwen) so
the trace comes back in its own ``reasoning_content`` field; without it the
trace is split off the inline ``<think>`` block instead. ``--thinking-effort``
sets the chat template's ``reasoning_effort`` (Qwen3.8 defaults to its most
verbose level), and each level writes to its own run directory.

Generation is resumable: requests with a usable record in ``results.jsonl``
are skipped, so an interrupted run can simply be restarted. Requests that
failed at the API (e.g. prompt + ``--max-output-tokens`` over vLLM's
``--max-model-len``) are retried on the next run.

There is no strict JSON-schema decoding here, unlike the API eval: an answer
that does not parse is filled with the train prior at scoring time and shows
up as ``parse_fail_rate``/``fill_rate``. Scoring needs deepchem.

``--perf`` records vLLM throughput/latency from the server's ``/metrics`` over
the generation window into ``perf.json``. The scores and perf data are logged
to one W&B run, named after the model, in the ``Molnet_local_llm`` project. Every run logs in to W&B first with the required ``--wandb-api-key``, so a bad
key fails before any generation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT.parent):  # DeepRetro, and DFS for the shared molnet package
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import deepretro.utils.llm  # noqa: E402,F401 -- loads .env and registers the langfuse_otel callback
from deepretro.utils.llm_interface import LLMRequest, create_llm_interface, split_thinking  # noqa: E402
from deepretro.utils.vllm_metrics import PerfRun  # noqa: E402
from molnet.config import DATASETS, DEFAULT_SHOTS  # noqa: E402
from molnet.evaluate import evaluate_run, load_results  # noqa: E402
from molnet.prompts import build_messages, load_split, parse_response, system_text  # noqa: E402
import deepretro.utils.llm_interface as llm_interface  # noqa: E402

# The LLM interface takes no Langfuse metadata, so wrap the function DeepRetro builds every completion's
# metadata with and merge in this query's keys (read by litellm's langfuse_otel callback).
_query_metadata: ContextVar[dict[str, Any]] = ContextVar("query_metadata", default={})
_deepretro_langfuse_metadata = llm_interface.langfuse_metadata


def _langfuse_metadata(base: dict[str, Any] | None, *, stage: str) -> dict[str, Any]:
    return {**_deepretro_langfuse_metadata(base, stage=stage), **_query_metadata.get()}


llm_interface.langfuse_metadata = _langfuse_metadata


def build_requests(args: argparse.Namespace) -> list[tuple[str, int, int, int, str]]:
    """``(dataset, k, seed, row, custom_id)`` for every query, sampled per seed as ``run_eval.py`` does."""
    requests = []
    for seed in args.seeds:
        for dataset in args.datasets:
            test = load_split(dataset, args.split)
            rows = test.index
            if args.limit:
                rows = test.sample(n=min(args.limit, len(test)), random_state=seed).index
            requests += [(dataset, k, seed, int(row), f"{dataset}-k{k}-s{seed}-r{int(row)}")
                         for k in args.shots for row in rows]
    return requests


def generate_one(args: argparse.Namespace, dataset: str, k: int, seed: int, row: int,
                 custom_id: str) -> dict[str, Any]:
    """Run one query and return its record, in ``run_eval.make_record``'s schema plus ``seed``."""
    messages = build_messages(dataset, args.split, k, row, seed)
    smiles = load_split(dataset, args.split).loc[row, "smiles"]
    # vLLM and DeepRetro's ChatMessage take string content, not text parts.
    messages = [{"role": "system", "content": system_text(messages)}, messages[1]]
    _query_metadata.set({
        "session_id": args.run_name, "trace_name": f"{dataset}/k{k}", "generation_name": "predict",
        "tags": [args.model, dataset, f"k={k}", f"seed={seed}", "local"],
        "trace_metadata": {"dataset": dataset, "k": k, "seed": seed, "row": row}, "dataset": dataset,
    })
    # call_LLM returns only (status, text); call the interface for the separate trace and its usage.
    response = create_llm_interface(args.model).call(LLMRequest(
        molecule=smiles,
        model=args.model,
        messages=messages,
        temperature=0.0,
        enable_thinking=args.thinking,
        thinking_effort=args.thinking_effort,
        max_output_tokens=args.max_output_tokens,
    ))
    text, reasoning = response.text, response.reasoning
    if response.status_code != 200:
        predictions, rec_status, error = {}, "api_error", text
    else:
        if not reasoning:
            # Server without --reasoning-parser: the trace is still inline. Split it off, since
            # the JSON regex is greedy and would span braces in the thinking.
            reasoning, text = split_thinking(text)
        predictions, error = parse_response(dataset, text)
        if error is None:
            rec_status = "ok"
        elif reasoning and not text.strip():
            rec_status = "truncated"  # the token cap ended the trace before any answer
        else:
            rec_status = "parse_error"
    return {
        "run_id": args.run_name, "model": args.model, "provider": "local", "mode": "local",
        "thinking": args.thinking, "thinking_effort": args.thinking_effort if args.thinking else None,
        "dataset": dataset, "k": k, "seed": seed, "row": row, "custom_id": custom_id,
        "text": text, "reasoning": reasoning, "predictions": predictions, "status": rec_status,
        "error": error, "stop_reason": None, "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
        "reasoning_tokens": response.reasoning_tokens, "reasoning_chars": len(reasoning),
        "cost_usd": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def perf_window(args: argparse.Namespace) -> PerfRun | nullcontext:
    """A :class:`PerfRun` on the vLLM server when ``--perf`` is set, else a no-op."""
    if not args.perf:
        return nullcontext()
    return PerfRun(os.getenv("HOSTED_VLLM_API_BASE", "http://localhost:8000/v1"))


def report_perf(perf: PerfRun, run_dir: Path, n_requests: int) -> dict[str, Any]:
    """Write ``perf.json``, print the headline numbers and return the perf summary."""
    summary = perf.summary(n_requests)
    with open(run_dir / "perf.json", "w") as f:
        json.dump({"summary": summary, "gauge_samples": perf.samples}, f, indent=2)

    def fmt(key: str, scale: float = 1.0, unit: str = "") -> str:
        value = summary.get(key)
        return "n/a" if value is None else f"{value * scale:.1f}{unit}"

    print(f"perf: {fmt('output_tok_per_s')} out tok/s, {fmt('mol_per_s')} req/s | "
          f"TTFT p50/p90 {fmt('ttft_p50_s', 1000)}/{fmt('ttft_p90_s', 1000, 'ms')} | "
          f"TPOT p50/p90 {fmt('tpot_p50_s', 1000)}/{fmt('tpot_p90_s', 1000, 'ms')} | "
          f"queue p90 {fmt('queue_p90_s', 1000, 'ms')}")
    print(f"perf: running mean/max {fmt('running_mean')}/{fmt('running_max')}, "
          f"waiting mean/max {fmt('waiting_mean')}/{fmt('waiting_max')}, "
          f"KV max {fmt('kv_pct_max', unit='%')}, preemptions {fmt('preemptions')}")
    return summary


def log_wandb(args: argparse.Namespace, metrics: pd.DataFrame,
              perf: tuple[dict[str, Any], list[dict[str, Any]]] | None) -> None:
    """Log the scores, and the perf gauges/summary if any, to one W&B run. Never raises."""
    config = {
        "task": "molnet", "model": args.model, "thinking": args.thinking,
        "thinking_effort": args.thinking_effort if args.thinking else None,
        "max_output_tokens": args.max_output_tokens, "workers": args.workers,
        "datasets": args.datasets, "shots": args.shots, "seeds": args.seeds, "limit": args.limit,
        "tag": args.tag,
    }
    try:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)
        if perf is not None:
            summary, samples = perf
            for sample in samples:
                run.log({k: v for k, v in sample.items() if k != "t" and v is not None}
                        | {"elapsed_s": sample["t"]})
            run.summary.update({k: v for k, v in summary.items() if v is not None})
        if not metrics.empty:
            run.log({"metrics": wandb.Table(dataframe=metrics)})
            for row in metrics.itertuples():
                prefix = f"{row.dataset}/k{row.k}/s{row.seed}"
                run.summary.update({f"{prefix}/{row.metric}": row.value,
                                    f"{prefix}/parse_fail_rate": row.parse_fail_rate,
                                    f"{prefix}/fill_rate": row.fill_rate})
            across = metrics.groupby(["dataset", "k", "metric"])["value"].agg(["mean", "std"])
            for (dataset, k, metric), stats in across.iterrows():
                run.summary.update({f"{dataset}/k{k}/{metric}_mean": stats["mean"],
                                    f"{dataset}/k{k}/{metric}_std": stats["std"]})
        run.finish()
    except Exception as exc:
        print(f"W&B logging failed: {exc}")


def run_generation(args: argparse.Namespace, run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Generate every request without a usable record, appending as each finishes.

    Returns ``(perf summary, gauge samples)`` when ``--perf`` recorded a window, else ``None``.
    """
    results = load_results(run_dir)
    done = set()
    if not results.empty:
        # Keyed on columns, not custom_id: records from before --seeds have no seed and count as seed 0.
        usable = results[results["status"] != "api_error"]
        seeds = usable["seed"].fillna(0) if "seed" in usable else pd.Series(0, index=usable.index)
        done = set(zip(usable["dataset"], usable["k"].astype(int), seeds.astype(int), usable["row"].astype(int)))
    todo = [r for r in build_requests(args) if r[:4] not in done]
    print(f"{len(done)} already done, generating {len(todo)}")
    if not todo:
        return None

    started = time.time()
    counts = {"ok": 0, "parse_error": 0, "truncated": 0, "api_error": 0}
    with perf_window(args) as perf, ThreadPoolExecutor(max_workers=args.workers) as pool, \
            open(run_dir / "results.jsonl", "a") as fout:
        futures = [pool.submit(generate_one, args, *r) for r in todo]
        for n, future in enumerate(as_completed(futures), start=1):
            rec = future.result()
            counts[rec["status"]] += 1
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}")
    elapsed = time.time() - started
    print(f"{len(todo)} requests in {elapsed / 60:.2f} min ({elapsed / len(todo):.2f} s/request): "
          f"{counts['ok']} ok, {counts['parse_error']} parse errors, "
          f"{counts['truncated']} truncated in thinking, "
          f"{counts['api_error']} API errors (re-run to retry)")
    if perf is None:
        return None
    completed = counts["ok"] + counts["parse_error"] + counts["truncated"]
    return report_perf(perf, run_dir, completed), perf.samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="e.g. hosted_vllm/zai-org/GLM-4.7-Flash")
    parser.add_argument("--datasets", default=",".join(DATASETS), help="comma-separated")
    parser.add_argument("--shots", default=",".join(map(str, DEFAULT_SHOTS)),
                        help="comma-separated k values; 0 = zero-shot")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seeds", default="0",
                        help="comma-separated seeds for ICL example selection and --limit sampling; "
                             "each seed is a full pass, scored separately")
    parser.add_argument("--limit", type=int, default=None, help="random sample of N molecules per dataset")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--thinking-effort", choices=["low", "medium", "high", "max"], default="medium",
                        help="reasoning_effort sent with --thinking; high and max both map to Qwen3.8's "
                             "xhigh, its default and most verbose level")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="default: 1024, or 16384 with --thinking")
    parser.add_argument("--workers", type=int, default=64, help="concurrent requests; vLLM batches them")
    parser.add_argument("--out-root", default=str(_REPO_ROOT / "results" / "molnet"))
    parser.add_argument("--perf", action="store_true",
                        help="record vLLM throughput/latency metrics from the server's /metrics "
                             "into perf.json (server URL from HOSTED_VLLM_API_BASE)")
    parser.add_argument("--wandb-api-key", required=True,
                        help="W&B API key; the run logs in with it before generating")
    parser.add_argument("--tag", default=None,
                        help="free-form W&B label for the server setup, "
                             "e.g. 'H100x2 tp2 max-num-seqs=256'")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    import wandb

    wandb.login(key=args.wandb_api_key, relogin=True, verify=True)
    args.datasets = args.datasets.split(",")
    unknown = [d for d in args.datasets if d not in DATASETS]
    if unknown:
        sys.exit(f"Unknown datasets: {unknown}. Choose from {list(DATASETS)}")
    args.shots = [int(s) for s in args.shots.split(",")]
    args.seeds = [int(s) for s in args.seeds.split(",")]
    if args.max_output_tokens is None:
        args.max_output_tokens = 16384 if args.thinking else 1024

    model_slug = args.model.rstrip("/").split("/")[-1]
    args.wandb_project = "Molnet_local_llm"
    args.wandb_run_name = model_slug
    # Effort is in the name so each level gets its own results.jsonl; sharing one would make the
    # resume logic skip every request already answered at another level.
    args.run_name = f"{model_slug}{f'_think_{args.thinking_effort}' if args.thinking else ''}"
    run_dir = Path(args.out_root) / args.run_name
    os.makedirs(run_dir, exist_ok=True)
    print(f"writing to {run_dir}")

    perf = run_generation(args, run_dir)
    metrics = evaluate_run(run_dir, args.split, default_seed=0)  # pre---seeds records were seed 0
    if not metrics.empty:
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        print(f"\nwrote {run_dir / 'metrics.csv'}")
    log_wandb(args, metrics, perf)


if __name__ == "__main__":
    main()
