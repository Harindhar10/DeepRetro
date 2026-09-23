#!/usr/bin/env python
"""Single-step retrosynthesis evaluation of a local model through DeepRetro.

The DeepRetro-native counterpart of ``code_v2_vllm.ipynb``: same dataset, same
prompts and the same metrics, but every request goes through
``deepretro.utils.llm`` to a vLLM OpenAI-compatible server. Start the server
first (in the vLLM venv), e.g.::

    VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve zai-org/GLM-4.7-Flash \\
        --dtype bfloat16 --max-model-len 17408 --enable-prefix-caching

then::

    HOSTED_VLLM_API_BASE=http://localhost:8000/v1 \\
    python scripts/eval_single_step_local.py \\
        --model hosted_vllm/zai-org/GLM-4.7-Flash --no-thinking --limit 10

Modes:

``raw``
    One ``call_LLM`` per molecule at temperature 0. The raw text is stored and
    parsed at scoring time -- this is what the notebook measures.
``pipeline``
    One ``llm_pipeline`` per molecule: temperature retries, ``validity_check``
    and the optional stability/hallucination filters. Scores what DeepRetro
    would actually hand to the multi-step search.
``autosolve``
    AiZynthFinder first, LLM as fallback -- what ``AutoSolver.single_step``
    does. AZ's answer is the first reaction of each route it finds. Both AZ
    and the LLM run on every molecule (separate checkpoints), and three views
    are scored: ``hybrid`` (AZ if it solved, else LLM), ``az`` and ``llm``.
    Needs aizynthfinder and ``AZ_MODELS_PATH``/``AZ_MODEL_CONFIG_PATH``.

Generation is resumable: finished molecules are read back from the JSONL
checkpoint and skipped, so an interrupted run can simply be restarted.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from deepretro.utils.llm import call_LLM, llm_pipeline  # noqa: E402
from deepretro.utils.one_step_eval import (  # noqa: E402
    hybrid_views,
    load_checkpoint,
    run_az_record,
    score_record,
    summarize,
    write_summary,
)

from deepretro.utils.utils_molecule import canonicalize  # noqa: E402

AUTOSOLVE_VIEWS = ("hybrid", "az", "llm")


def generate_one(args: argparse.Namespace, smiles: str) -> dict[str, Any]:
    """Run one molecule and return the fields to store in its record."""
    if args.mode == "raw":
        status, text = call_LLM(
            smiles,
            model=args.model,
            temperature=0.0,
            enable_thinking=args.thinking,
            max_output_tokens=args.max_output_tokens,
        )
        return {"call_status": status, "raw": text}
    pathways, _, _ = llm_pipeline(
        smiles,
        model=args.model,
        enable_thinking=args.thinking,
        max_output_tokens=args.max_output_tokens,
        stability_check=args.stability_check,
        hallucination_check=args.hallucination_check,
    )
    return {"proposals": pathways, "status": 200 if pathways else 504}


def run_generation(args: argparse.Namespace, df: pd.DataFrame, jsonl: str) -> None:
    """Generate for every molecule not yet in ``jsonl``, appending as each finishes."""
    done = load_checkpoint(jsonl)
    todo = [(int(i), row) for i, row in df.iterrows() if int(i) not in done]
    if args.limit is not None:
        todo = todo[: max(0, args.limit - len(done))]
    print(f"{len(done)} already done, generating {len(todo)}")
    if not todo:
        return

    started = time.time()
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(jsonl, "a") as fout:
        futures = {pool.submit(generate_one, args, row["input"]): (i, row) for i, row in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            i, row = futures[future]
            result = future.result()
            if result.get("call_status", 200) != 200:
                failed += 1
                continue  # not checkpointed, so a re-run retries it
            fout.write(json.dumps({
                "mol_no": i,
                "model_id": args.model,
                "thinking": args.thinking,
                "mode": args.mode,
                "input": row["input"],
                "output": row["output"],
                **result,
            }) + "\n")
            fout.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}")
    elapsed = time.time() - started
    print(
        f"{len(todo)} molecules in {elapsed / 60:.2f} min "
        f"({elapsed / len(todo):.2f} s/mol), {failed} API failures (re-run to retry)"
    )


def _pending(df: pd.DataFrame, done: dict[int, Any], limit: int | None) -> list[tuple[int, Any]]:
    """Rows of ``df`` (first ``limit``) that are not yet in ``done``."""
    rows = [(int(i), row) for i, row in df.iterrows()]
    if limit is not None:
        rows = rows[:limit]
    return [(i, row) for i, row in rows if i not in done]


def resolve_az_config(az_model: str) -> str:
    """Return the AZ config that will be used, failing early if there is none."""
    try:
        import aizynthfinder  # noqa: F401
    except ImportError:
        sys.exit("aizynthfinder is not installed in this environment (pip install aizynthfinder)")
    from deepretro.utils import az

    wanted = os.path.join(az.AZ_MODELS_PATH, az_model, "config.yml")
    if os.path.exists(wanted):
        return wanted
    if os.path.exists(az.AZ_MODEL_CONFIG_PATH):
        print(f"WARNING: {wanted} not found; AZ will use the fallback "
              f"AZ_MODEL_CONFIG_PATH={az.AZ_MODEL_CONFIG_PATH}")
        return az.AZ_MODEL_CONFIG_PATH
    sys.exit(f"No AZ config: neither {wanted} nor AZ_MODEL_CONFIG_PATH="
             f"{az.AZ_MODEL_CONFIG_PATH} exists. Set AZ_MODELS_PATH (relative to the repo root).")


def run_llm_step(args: argparse.Namespace, df: pd.DataFrame, jsonl: str) -> None:
    """LLM proposals for every molecule via ``AutoSolver.run_llm``, checkpointed."""
    from deepretro.algorithms.autosolve import AutoSolver

    todo = _pending(df, load_checkpoint(jsonl), args.limit)
    print(f"[llm] generating {len(todo)}")
    if not todo:
        return
    solver = AutoSolver(
        llm=args.model,
        az_model=args.az_model,
        stability_check=args.stability_check,
        hallucination_mode="heuristic" if args.hallucination_check else "none",
        enable_thinking=args.thinking,
        max_output_tokens=args.max_output_tokens,
    )
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(jsonl, "a") as fout:
        futures = {pool.submit(solver.run_llm, canonicalize(row["input"])): i for i, row in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            pathways, _explanations, confidence = future.result()
            fout.write(json.dumps({
                "mol_no": futures[future], "proposals": pathways, "confidence": confidence,
            }) + "\n")
            fout.flush()
            if n % 25 == 0:
                print(f"  [llm] {n}/{len(todo)}")
    print(f"[llm] {len(todo)} molecules in {(time.time() - started) / 60:.2f} min")


def run_az_step(args: argparse.Namespace, df: pd.DataFrame, jsonl: str) -> None:
    """AiZynthFinder search for every molecule in worker processes, checkpointed.

    Processes, not threads: each AZ finder is cached per process and keeps
    per-search state, and the search is CPU-bound.
    """
    todo = _pending(df, load_checkpoint(jsonl), args.limit)
    print(f"[az] searching {len(todo)} with {args.az_workers} workers")
    if not todo:
        return
    started = time.time()
    # spawn, not fork: by now litellm/langfuse have background threads, and
    # forking a threaded process can deadlock the child.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.az_workers, mp_context=context) as pool, \
            open(jsonl, "a") as fout:
        futures = [pool.submit(run_az_record, i, canonicalize(row["input"]), args.az_model) for i, row in todo]
        for n, future in enumerate(as_completed(futures), start=1):
            rec = future.result()
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if n % 10 == 0:
                print(f"  [az] {n}/{len(todo)} ({(time.time() - started) / 60:.1f} min)")
    print(f"[az] {len(todo)} molecules in {(time.time() - started) / 60:.2f} min")


def score_autosolve(args: argparse.Namespace, df: pd.DataFrame, out_dir: str,
                    meta: dict[str, Any]) -> None:
    """Score the hybrid/az/llm views and write per-view results and one summary."""
    az_recs = load_checkpoint(os.path.join(out_dir, "az.jsonl"))
    llm_recs = load_checkpoint(os.path.join(out_dir, "llm.jsonl"))
    both = sorted(set(az_recs) & set(llm_recs))
    missing = len(set(az_recs) ^ set(llm_recs))
    if missing:
        print(f"{missing} molecules are in only one checkpoint and are left out; re-run to fill them")
    if not both:
        print("nothing to score")
        return

    rows: dict[str, list[dict[str, Any]]] = {v: [] for v in AUTOSOLVE_VIEWS}
    sources = {}
    for i in both:
        row = df.loc[i]
        base = {"mol_no": i, "input": row["input"], "output": row["output"], "model_id": args.model}
        views = hybrid_views(az_recs[i], llm_recs[i], base)
        sources[i] = views["hybrid"]["source"]
        for v in AUTOSOLVE_VIEWS:
            rows[v].append(score_record(views[v]))

    scored = {v: pd.DataFrame(rows[v]).set_index("mol_no") for v in AUTOSOLVE_VIEWS}
    scored["hybrid"]["source"] = pd.Series(sources)
    az_meta = pd.DataFrame([az_recs[i] for i in both]).set_index("mol_no")
    scored["az"] = scored["az"].join(az_meta[["solved", "n_routes", "error", "seconds"]])
    for v in AUTOSOLVE_VIEWS:
        scored[v].to_csv(os.path.join(out_dir, f"results_{v}.csv"))

    n = len(both)
    view_summaries, reports = {}, []
    for v in AUTOSOLVE_VIEWS:
        view_summaries[v], text = summarize(scored[v], {**meta, "view": v})
        reports.append(f"##### view: {v} #####\n{text}")

    # Head-to-head where AZ answered: would the LLM have done better there?
    az_answered = [i for i in both if sources[i] == "az"]
    subset = {}
    if az_answered:
        for v in ("az", "llm"):
            sub = scored[v].loc[az_answered]
            subset[v] = {
                "top1_all_correct": 100 * sub.all_correct.mean(),
                "topk_all_correct": 100 * sub.all_correct_topk.mean(),
                "top1_maxfrag": 100 * sub.maxfrag_correct.mean(),
            }
    coverage = {
        "az_solved_pct": 100 * sum(bool(az_recs[i]["solved"]) for i in both) / n,
        "az_answered_pct": 100 * len(az_answered) / n,
        "az_errors": sum(bool(az_recs[i].get("error")) for i in both),
        "az_mean_seconds": float(az_meta["seconds"].mean()),
    }
    summary = {**meta, "n_scored": n, "coverage": coverage,
               "az_answered_subset": subset, "views": view_summaries}

    lines = [
        f"model: {args.model}   az_model: {args.az_model}   scored {n} molecules",
        f"AZ solved {coverage['az_solved_pct']:.1f}%, answered (has a first step) "
        f"{coverage['az_answered_pct']:.1f}%, {coverage['az_errors']} AZ errors, "
        f"{coverage['az_mean_seconds']:.1f} s/molecule",
    ]
    if subset:
        lines.append(f"On the {len(az_answered)} molecules AZ answered -- "
                     f"top-1 all_correct: AZ {subset['az']['top1_all_correct']:.1f}% "
                     f"vs LLM {subset['llm']['top1_all_correct']:.1f}%")
    report = "\n".join(lines) + "\n\n" + "\n\n".join(reports)
    print(report)
    write_summary(summary, report, out_dir,
                  os.path.join(args.out_root, "summary_all_deepretro.csv"), views=view_summaries)
    print(f"\nwrote {out_dir}/summary.json, summary.txt and results_{{hybrid,az,llm}}.csv")


def run_autosolve(args: argparse.Namespace, df: pd.DataFrame, out_dir: str,
                  meta: dict[str, Any]) -> None:
    config = resolve_az_config(args.az_model)
    print(f"AZ config: {config}")
    run_llm_step(args, df, os.path.join(out_dir, "llm.jsonl"))
    run_az_step(args, df, os.path.join(out_dir, "az.jsonl"))
    score_autosolve(args, df, out_dir, {**meta, "az_model": args.az_model, "az_config": config})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="e.g. hosted_vllm/zai-org/GLM-4.7-Flash")
    parser.add_argument("--data", default=str(_REPO_ROOT / "data" / "uspto_50k_test_250.csv"),
                        help="CSV with 'input' (product) and 'output' (reactants) columns")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="default: 1024, or 16384 with --thinking")
    parser.add_argument("--mode", choices=["raw", "pipeline", "autosolve"], default="raw")
    parser.add_argument("--az-model", default="USPTO",
                        help="autosolve mode: AZ model folder under AZ_MODELS_PATH")
    parser.add_argument("--az-workers", type=int, default=4,
                        help="autosolve mode: parallel AZ processes (each loads its own model)")
    parser.add_argument("--workers", type=int, default=64,
                        help="concurrent requests; vLLM batches them")
    parser.add_argument("--limit", type=int, default=None, help="evaluate the first N molecules")
    parser.add_argument("--stability-check", action="store_true", help="pipeline/autosolve modes")
    parser.add_argument("--hallucination-check", action="store_true", help="pipeline/autosolve modes")
    parser.add_argument("--out-root", default=str(_REPO_ROOT / "results"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.max_output_tokens is None:
        args.max_output_tokens = 16384 if args.thinking else 1024

    model_slug = args.model.rstrip("/").split("/")[-1]
    data_slug = Path(args.data).stem
    run_name = f"{data_slug}_{model_slug}{'_think' if args.thinking else ''}_deepretro_{args.mode}"
    if args.mode == "autosolve":
        run_name += f"_{args.az_model}"
    out_dir = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing to {out_dir}")

    df = pd.read_csv(args.data)
    meta = {
        "run": run_name,
        "model_id": args.model,
        "mode": args.mode,
        "thinking": args.thinking,
        "max_new_tokens": args.max_output_tokens,
        "data_csv": args.data,
        "n_total": int(len(df)),
    }
    if args.mode == "autosolve":
        run_autosolve(args, df, out_dir, meta)
        return

    jsonl = os.path.join(out_dir, "generations.jsonl")
    run_generation(args, df, jsonl)

    records = load_checkpoint(jsonl)
    if not records:
        print("nothing to score")
        return
    scored = pd.DataFrame([score_record(r) for r in records.values()]).set_index("mol_no").sort_index()
    scored.to_csv(os.path.join(out_dir, "results.csv"))
    summary, report = summarize(scored, meta)
    print(report)
    write_summary(summary, report, out_dir, os.path.join(args.out_root, "summary_all_deepretro.csv"))
    print(f"\nwrote {out_dir}/summary.json, summary.txt and results.csv")


if __name__ == "__main__":
    main()
