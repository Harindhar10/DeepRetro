"""Single-step retrosynthesis evaluation against ground-truth reactants.

Scores model proposals the way ``notebooks/prod_challenge.ipynb`` and
``code_v2_vllm.ipynb`` do: canonicalize predicted precursors and ground-truth
reactants with RDKit, then require equal cardinality plus set membership.
``all_correct`` means every predicted precursor is in the ground truth,
``any_correct`` means at least one is. ``maxfrag_correct`` is the MaxFrag
accuracy of Tetko et al., *Nat. Commun.* 11, 5575 (2020): only the largest
predicted precursor (most heavy atoms) must match the largest reactant.

Each metric is reported at top-1 (first proposal) and top-k (best proposal).
Parse failures count as incorrect; they are never dropped from the denominator.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger

from deepretro.utils.llm import validate_split_json
from deepretro.utils.llm_helpers import Pathway
from deepretro.utils.llm_interface import create_llm_interface

RDLogger.DisableLog("rdApp.*")


def load_checkpoint(path: str) -> dict[int, dict[str, Any]]:
    """Load generation records from a JSONL file, keyed by ``mol_no``.

    Parameters
    ----------
    path : str
        JSONL checkpoint path. A missing file yields an empty dict.

    Returns
    -------
    dict[int, dict[str, Any]]
        Records keyed by molecule number.
    """
    done: dict[int, dict[str, Any]] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["mol_no"]] = rec
    return done


def canon(smiles: str) -> str | None:
    """Return the RDKit canonical SMILES, or ``None`` if it does not parse.

    Examples
    --------
    >>> canon("OCC")
    'CCO'
    >>> canon("not smiles") is None
    True
    """
    try:
        return Chem.CanonSmiles(smiles)
    except Exception:
        return None


def max_frag(canon_list: list[str]) -> str | None:
    """Return the largest fragment of a list of canonical SMILES (MaxFrag).

    "Largest" is the molecule with the most heavy atoms; ties go to the longer
    canonical SMILES so the choice is deterministic.

    Examples
    --------
    >>> max_frag(["CCO", "CCCCO"])
    'CCCCO'
    >>> max_frag([]) is None
    True
    """
    mols = [(Chem.MolFromSmiles(s), s) for s in canon_list]
    mols = [(m, s) for m, s in mols if m is not None]
    if not mols:
        return None
    return max(mols, key=lambda ms: (ms[0].GetNumHeavyAtoms(), len(ms[1])))[1]


def score_proposal(
    pred_smiles_list: list[str], gt_canon: list[str]
) -> tuple[int, int, int, list[str], list[str], bool]:
    """Score one proposal against canonical ground-truth reactants.

    Returns
    -------
    tuple
        ``(any_correct, all_correct, maxfrag_correct, not_common, missed,
        valid)``. ``valid`` is ``False`` when a predicted SMILES does not parse.

    Examples
    --------
    >>> score_proposal(["OCC", "CC(=O)O"], ["CCO", "CC(=O)O"])[:3]
    (1, 1, 1)
    >>> score_proposal(["CCO"], ["CCO", "CC(=O)O"])[:3]
    (0, 0, 0)
    """
    pred = [canon(s) for s in pred_smiles_list]
    if any(p is None for p in pred):
        invalid = [s for s, p in zip(pred_smiles_list, pred) if p is None]
        return 0, 0, 0, invalid, gt_canon, False
    canon_pred = [p for p in pred if p is not None]
    same_len = len(gt_canon) == len(canon_pred)
    any_c = int(any(p in gt_canon for p in canon_pred) and same_len)
    all_c = int(all(p in gt_canon for p in canon_pred) and same_len)
    pred_frag = max_frag(canon_pred)
    maxfrag_c = int(pred_frag is not None and pred_frag == max_frag(gt_canon))
    not_common = [p for p in canon_pred if p not in gt_canon]
    missed = [g for g in gt_canon if g not in canon_pred]
    return any_c, all_c, maxfrag_c, not_common, missed, True


def parse_proposals(raw: str, model: str) -> tuple[int, list[Pathway]]:
    """Parse a raw model response into precursor proposals via DeepRetro's parser.

    Parameters
    ----------
    raw : str
        Raw model response text.
    model : str
        Model identifier used to select the provider parser.

    Returns
    -------
    tuple[int, list[Pathway]]
        Status code (``200`` on success) and the proposals.
    """
    status, _, json_content = create_llm_interface(model).parse_response(raw)
    if status != 200:
        return status, []
    status, pathways, _, _ = validate_split_json(json_content)
    if status != 200:
        return status, []
    proposals = [
        [s for s in pathway if isinstance(s, str) and s.strip()]
        for pathway in pathways
        if isinstance(pathway, list)
    ]
    proposals = [p for p in proposals if p]
    return (200, proposals) if proposals else (504, [])


def score_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Score one generation record at top-1 and top-k.

    ``rec`` holds either ``raw`` (a model response, parsed here with the
    parser for ``rec["model_id"]``) or ``proposals`` (already parsed, as the
    pipeline mode stores them).
    """
    gt_canon = [canon(s) for s in str(rec["output"]).split(".")]
    gt = [g for g in gt_canon if g is not None]

    if "proposals" in rec:
        proposals = rec["proposals"]
        status = 200 if proposals else rec.get("status", 504)
    else:
        status, proposals = parse_proposals(rec["raw"], rec["model_id"])

    row: dict[str, Any] = {
        "mol_no": rec["mol_no"],
        "input": rec["input"],
        "output": rec["output"],
        "parse_status": status,
        "n_proposals": len(proposals),
        "proposals": proposals,
        "any_correct": 0,
        "all_correct": 0,
        "maxfrag_correct": 0,
        "any_correct_topk": 0,
        "all_correct_topk": 0,
        "maxfrag_correct_topk": 0,
        "first_hit_rank": None,
        "not_common": [],
        "missed": gt,
        "has_invalid_smiles": 0,
    }
    if status != 200 or not proposals:
        return row

    any_invalid = False
    for rank, prop in enumerate(proposals, start=1):
        any_c, all_c, maxfrag_c, not_common, missed, valid = score_proposal(prop, gt)
        if not valid:
            any_invalid = True
        if rank == 1:
            row.update(
                any_correct=any_c,
                all_correct=all_c,
                maxfrag_correct=maxfrag_c,
                not_common=not_common,
                missed=missed,
            )
        row["any_correct_topk"] = max(row["any_correct_topk"], any_c)
        row["all_correct_topk"] = max(row["all_correct_topk"], all_c)
        row["maxfrag_correct_topk"] = max(row["maxfrag_correct_topk"], maxfrag_c)
        if all_c and row["first_hit_rank"] is None:
            row["first_hit_rank"] = rank
    row["has_invalid_smiles"] = int(any_invalid)
    return row


def _counts_and_pcts(all_c: pd.Series, any_c: pd.Series, maxfrag_c: pd.Series, n: int) -> dict[str, Any]:
    counts = {
        "n_all_correct": int(all_c.sum()),
        "n_any_correct": int(any_c.sum()),
        "n_maxfrag": int(maxfrag_c.sum()),
    }
    pcts = {k[2:]: 100 * v / n for k, v in counts.items()}
    return {**pcts, **counts}


def summarize(scored: pd.DataFrame, meta: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Build the headline summary dict and printable report for scored rows.

    Parameters
    ----------
    scored : pandas.DataFrame
        Output of :func:`score_record` for every molecule.
    meta : dict[str, Any]
        Run metadata placed at the front of the summary (model, data, ...).

    Returns
    -------
    tuple[dict[str, Any], str]
        Summary dict and the report text rendered from it.
    """
    n = len(scored)
    summary = {
        **meta,
        "n_scored": int(n),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "top1": _counts_and_pcts(scored.all_correct, scored.any_correct, scored.maxfrag_correct, n),
        "topk": _counts_and_pcts(
            scored.all_correct_topk, scored.any_correct_topk, scored.maxfrag_correct_topk, n
        ),
        "parse_failures_pct": 100 * (scored.parse_status != 200).sum() / n,
        "invalid_smiles_pct": 100 * scored.has_invalid_smiles.sum() / n,
        "mean_proposals": float(scored.n_proposals.mean()),
    }
    top1, topk = summary["top1"], summary["topk"]
    report = "\n".join([
        f"model: {meta.get('model_id')}  mode: {meta.get('mode')}",
        f"scored {n} molecules",
        "",
        "=== top-1 (first proposal) ===",
        f"All correct % {top1['all_correct']:.2f}  ({top1['n_all_correct']}/{n})",
        f"Any correct % {top1['any_correct']:.2f}  ({top1['n_any_correct']}/{n})",
        f"MaxFrag     % {top1['maxfrag']:.2f}  ({top1['n_maxfrag']}/{n})",
        "",
        "=== top-k (best proposal) ===",
        f"All correct % {topk['all_correct']:.2f}  ({topk['n_all_correct']}/{n})",
        f"Any correct % {topk['any_correct']:.2f}  ({topk['n_any_correct']}/{n})",
        f"MaxFrag     % {topk['maxfrag']:.2f}  ({topk['n_maxfrag']}/{n})",
        "",
        "=== response quality ===",
        f"parse failures  : {summary['parse_failures_pct']:.2f}%",
        f"invalid SMILES  : {summary['invalid_smiles_pct']:.2f}%",
        f"mean proposals  : {summary['mean_proposals']:.2f}",
        "",
        f"Denominator is all {n} scored rows -- parse failures count as incorrect.",
    ])
    return summary, report


def write_summary(
    summary: dict[str, Any], report: str, out_dir: str, all_summaries_csv: str
) -> None:
    """Write ``summary.json``/``summary.txt`` and append a row to the cross-run log.

    The cross-run CSV gets one flat row per scoring run; if the column set
    changed since it was created, the file is rewritten so headers stay aligned.
    """
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    flat = {k: v for k, v in summary.items() if not isinstance(v, dict)}
    flat.update({f"top1_{k}": v for k, v in summary["top1"].items()})
    flat.update({f"topk_{k}": v for k, v in summary["topk"].items()})
    row = pd.DataFrame([flat])
    if os.path.exists(all_summaries_csv):
        prev = pd.read_csv(all_summaries_csv)
        if list(prev.columns) == list(row.columns):
            row.to_csv(all_summaries_csv, mode="a", index=False, header=False)
        else:
            pd.concat([prev, row], ignore_index=True).to_csv(all_summaries_csv, index=False)
    else:
        row.to_csv(all_summaries_csv, index=False)
