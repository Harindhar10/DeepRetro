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

Generation is resumable: finished molecules are read back from the JSONL
checkpoint and skipped, so an interrupted run can simply be restarted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from deepretro.utils.llm import call_LLM, llm_pipeline  # noqa: E402
from deepretro.utils.one_step_eval import (  # noqa: E402
    load_checkpoint,
    score_record,
    summarize,
    write_summary,
)


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="e.g. hosted_vllm/zai-org/GLM-4.7-Flash")
    parser.add_argument("--data", default=str(_REPO_ROOT / "data" / "uspto_50k_test_250.csv"),
                        help="CSV with 'input' (product) and 'output' (reactants) columns")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="default: 1024, or 16384 with --thinking")
    parser.add_argument("--mode", choices=["raw", "pipeline"], default="raw")
    parser.add_argument("--workers", type=int, default=64,
                        help="concurrent requests; vLLM batches them")
    parser.add_argument("--limit", type=int, default=None, help="evaluate the first N molecules")
    parser.add_argument("--stability-check", action="store_true", help="pipeline mode only")
    parser.add_argument("--hallucination-check", action="store_true", help="pipeline mode only")
    parser.add_argument("--out-root", default=str(_REPO_ROOT / "results"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.max_output_tokens is None:
        args.max_output_tokens = 16384 if args.thinking else 1024

    model_slug = args.model.rstrip("/").split("/")[-1]
    data_slug = Path(args.data).stem
    run_name = f"{data_slug}_{model_slug}{'_think' if args.thinking else ''}_deepretro_{args.mode}"
    out_dir = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, "generations.jsonl")
    print(f"writing to {out_dir}")

    df = pd.read_csv(args.data)
    run_generation(args, df, jsonl)

    records = load_checkpoint(jsonl)
    if not records:
        print("nothing to score")
        return
    scored = pd.DataFrame([score_record(r) for r in records.values()]).set_index("mol_no").sort_index()
    scored.to_csv(os.path.join(out_dir, "results.csv"))
    summary, report = summarize(scored, {
        "run": run_name,
        "model_id": args.model,
        "mode": args.mode,
        "thinking": args.thinking,
        "max_new_tokens": args.max_output_tokens,
        "data_csv": args.data,
        "n_total": int(len(df)),
    })
    print(report)
    write_summary(summary, report, out_dir, os.path.join(args.out_root, "summary_all_deepretro.csv"))
    print(f"\nwrote {out_dir}/summary.json, summary.txt and results.csv")


if __name__ == "__main__":
    main()
