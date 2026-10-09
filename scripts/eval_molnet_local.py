#!/usr/bin/env python
"""MoleculeNet property prediction with a local model through DeepRetro.

The local-model counterpart of ``Claude_GPT_eval/run_eval.py``: same datasets,
prompts, in-context examples, ``--limit`` sample and ROC-AUC/RMSE scoring (all
from the shared ``molnet`` package in the DFS folder), but every request goes
through ``deepretro.utils.llm.call_LLM`` to a vLLM OpenAI-compatible server.
Start the server first, e.g.::

    vllm serve zai-org/GLM-4.7-Flash --dtype bfloat16 --max-model-len 17408

then::

    HOSTED_VLLM_API_BASE=http://localhost:8000/v1 \\
    python scripts/eval_molnet_local.py \\
        --model hosted_vllm/zai-org/GLM-4.7-Flash --no-thinking --limit 50

Generation is resumable: requests with a usable record in ``results.jsonl``
are skipped, so an interrupted run can simply be restarted. Requests that
failed at the API (e.g. prompt + ``--max-output-tokens`` over vLLM's
``--max-model-len``) are retried on the next run.

There is no strict JSON-schema decoding here, unlike the API eval: an answer
that does not parse is filled with the train prior at scoring time and shows
up as ``parse_fail_rate``/``fill_rate``. Scoring needs deepchem.

``--perf`` records vLLM throughput/latency from the server's ``/metrics`` over
the generation window into ``perf.json`` (and W&B with ``--wandb-project``).
Every run logs in to W&B first with the required ``--wandb-api-key``, so a bad
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT.parent):  # DeepRetro, and DFS for the shared molnet package
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from deepretro.utils.llm import call_LLM  # noqa: E402
from deepretro.utils.llm_interface import split_thinking  # noqa: E402
from deepretro.utils.vllm_metrics import PerfRun, log_to_wandb  # noqa: E402
from molnet.config import DATASETS, DEFAULT_SHOTS  # noqa: E402
from molnet.evaluate import evaluate_run, load_results  # noqa: E402
from molnet.prompts import build_messages, load_split, parse_response, system_text  # noqa: E402


def build_requests(args: argparse.Namespace) -> list[tuple[str, int, int, str]]:
    """``(dataset, k, row, custom_id)`` for every query, sampled as ``run_eval.py`` does."""
    requests = []
    for dataset in args.datasets:
        test = load_split(dataset, args.split)
        rows = test.index
        if args.limit:
            rows = test.sample(n=min(args.limit, len(test)), random_state=args.seed).index
        requests += [(dataset, k, int(row), f"{dataset}-k{k}-r{int(row)}")
                     for k in args.shots for row in rows]
    return requests


def generate_one(args: argparse.Namespace, dataset: str, k: int, row: int, custom_id: str) -> dict[str, Any]:
    """Run one query and return its record, in ``run_eval.make_record``'s schema."""
    messages = build_messages(dataset, args.split, k, row, args.seed)
    smiles = load_split(dataset, args.split).loc[row, "smiles"]
    # vLLM and DeepRetro's ChatMessage take string content, not text parts.
    messages = [{"role": "system", "content": system_text(messages)}, messages[1]]
    status, text = call_LLM(
        smiles,
        model=args.model,
        messages=messages,
        temperature=0.0,
        enable_thinking=args.thinking,
        max_output_tokens=args.max_output_tokens,
    )
    if status != 200:
        predictions, rec_status, error = {}, "api_error", text
    else:
        # Parse only the answer: the JSON regex is greedy and would span braces in the thinking.
        _, answer = split_thinking(text)
        predictions, error = parse_response(dataset, answer)
        rec_status = "ok" if error is None else "parse_error"
    return {
        "run_id": args.run_name, "model": args.model, "provider": "local", "mode": "local",
        "thinking": args.thinking, "dataset": dataset, "k": k, "row": row, "custom_id": custom_id,
        "text": text, "predictions": predictions, "status": rec_status, "error": error,
        "stop_reason": None, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
        "cache_write_tokens": 0, "reasoning_tokens": 0, "cost_usd": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def perf_window(args: argparse.Namespace) -> PerfRun | nullcontext:
    """A :class:`PerfRun` on the vLLM server when ``--perf`` is set, else a no-op."""
    if not args.perf:
        return nullcontext()
    return PerfRun(os.getenv("HOSTED_VLLM_API_BASE", "http://localhost:8000/v1"))


def report_perf(args: argparse.Namespace, perf: PerfRun, run_dir: Path, n_requests: int) -> None:
    """Write ``perf.json``, print the headline numbers and log the run to W&B."""
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
    if args.wandb_project:
        config = {
            "task": "molnet", "model": args.model, "thinking": args.thinking,
            "max_output_tokens": args.max_output_tokens, "workers": args.workers,
            "datasets": args.datasets, "shots": args.shots, "limit": args.limit, "tag": args.tag,
        }
        log_to_wandb(args.wandb_project, args.run_name, config, summary, perf.samples)


def run_generation(args: argparse.Namespace, run_dir: Path) -> None:
    """Generate every request without a usable record, appending as each finishes."""
    results = load_results(run_dir)
    done = set() if results.empty else set(results.loc[results["status"] != "api_error", "custom_id"])
    todo = [r for r in build_requests(args) if r[3] not in done]
    print(f"{len(done)} already done, generating {len(todo)}")
    if not todo:
        return

    started = time.time()
    counts = {"ok": 0, "parse_error": 0, "api_error": 0}
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
          f"{counts['api_error']} API errors (re-run to retry)")
    if perf is not None:
        report_perf(args, perf, run_dir, counts["ok"] + counts["parse_error"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="e.g. hosted_vllm/zai-org/GLM-4.7-Flash")
    parser.add_argument("--datasets", default=",".join(DATASETS), help="comma-separated")
    parser.add_argument("--shots", default=",".join(map(str, DEFAULT_SHOTS)),
                        help="comma-separated k values; 0 = zero-shot")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0, help="seed for ICL example selection and --limit sampling")
    parser.add_argument("--limit", type=int, default=None, help="random sample of N molecules per dataset")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="default: 1024, or 16384 with --thinking")
    parser.add_argument("--workers", type=int, default=64, help="concurrent requests; vLLM batches them")
    parser.add_argument("--out-root", default=str(_REPO_ROOT / "results" / "molnet"))
    parser.add_argument("--perf", action="store_true",
                        help="record vLLM throughput/latency metrics from the server's /metrics "
                             "into perf.json (server URL from HOSTED_VLLM_API_BASE)")
    parser.add_argument("--wandb-api-key", required=True,
                        help="W&B API key; the run logs in with it before generating")
    parser.add_argument("--wandb-project", default=None,
                        help="with --perf: also log the run to this W&B project")
    parser.add_argument("--tag", default=None,
                        help="with --perf: free-form label for the server setup, "
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
    if args.max_output_tokens is None:
        args.max_output_tokens = 16384 if args.thinking else 1024

    model_slug = args.model.rstrip("/").split("/")[-1]
    args.run_name = f"{model_slug}{'_think' if args.thinking else ''}"
    run_dir = Path(args.out_root) / args.run_name
    os.makedirs(run_dir, exist_ok=True)
    print(f"writing to {run_dir}")

    run_generation(args, run_dir)
    metrics = evaluate_run(run_dir, args.split)
    if metrics.empty:
        return
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    print(f"\nwrote {run_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
