"""Tests for the local (vLLM ``hosted_vllm/``) LLM backend."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from deepretro.utils import llm_interface
from deepretro.utils.llm import llm_pipeline, validate_split_json
from deepretro.utils.llm_helpers import (
    build_completion_params,
    infer_provider,
    resolve_model_selection,
)
from deepretro.utils.llm_interface import LocalLLM, create_llm_interface
from deepretro.utils.one_step_eval import score_record, summarize
from deepretro.utils.variables import SYS_PROMPT_OPENAI

LOCAL_MODEL = "hosted_vllm/zai-org/GLM-4.7-Flash"
MESSAGES = [{"role": "user", "content": "Reply OK"}]


@pytest.mark.parametrize(
    "model",
    [
        LOCAL_MODEL,
        "hosted_vllm/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "hosted_vllm/openai/gpt-oss-20b",
    ],
)
def test_local_models_are_never_routed_to_hosted_providers(model: str) -> None:
    selection = resolve_model_selection(model)

    assert infer_provider(model) == "local"
    assert selection.completion_model == model
    assert selection.family == "openai"
    assert not selection.supports_reasoning_effort
    assert isinstance(create_llm_interface(model), LocalLLM)


def test_local_completion_params_target_the_vllm_server(monkeypatch: Any) -> None:
    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "http://gpu-box:8001/v1")
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)

    params = build_completion_params(LOCAL_MODEL, MESSAGES, 1024, 0.0, enable_thinking=False)

    assert params["api_base"] == "http://gpu-box:8001/v1"
    assert params["api_key"] == "EMPTY"
    assert params["max_tokens"] == 1024
    assert params["temperature"] == 0.0
    assert params["seed"] == 42
    assert "reasoning_effort" not in params
    assert params["messages"] is MESSAGES  # no Anthropic cache_control markers
    assert params["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "stop": ["</json>"],
        "include_stop_str_in_output": True,
    }


def test_local_thinking_uses_recommended_sampling() -> None:
    params = build_completion_params(LOCAL_MODEL, MESSAGES, 16384, 0.3, enable_thinking=True)

    assert params["temperature"] == 0.6
    assert params["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert (params["extra_body"]["top_p"], params["extra_body"]["top_k"]) == (0.95, 20)


def test_local_parser_ignores_json_drafted_inside_thinking() -> None:
    parser = LocalLLM(LOCAL_MODEL)
    draft = '<json>{"data": [["WRONG"]]}</json>'
    final = '{"data": [["CCO"]], "explanation": ["x"], "confidence_scores": [0.9]}'

    status, thinking, payload = parser.parse_response(
        f"<think>let me draft {draft}</think>\n<json>{final}</json>"
    )
    assert (status, payload) == (200, final)
    assert thinking == [f"let me draft {draft}"]

    # Template put <think> in the prompt: only the closing tag is in the output.
    assert parser.parse_response(f"draft {draft}</think><json>{final}</json>")[2] == final
    # Server-side --reasoning-parser: no thinking tags at all.
    assert parser.parse_response(f"<json>{final}</json>") == (200, [], final)
    # Thinking cut off by the token cap: the draft must not be scored.
    assert parser.parse_response(f"<think>draft {draft}")[0] == 502


def test_validate_split_json_accepts_strict_json_and_flat_lists() -> None:
    assert validate_split_json(
        '{"data": [["CCO"]], "explanation": [null], "confidence_scores": [1], "ok": true}'
    ) == (200, [["CCO"]], [None], [1.0])
    assert validate_split_json(
        '{"data": ["CCO", "CC(=O)O"], "explanation": ["e"], "confidence_scores": [0.5]}'
    ) == (200, [["CCO", "CC(=O)O"]], ["e"], [0.5])


def test_llm_pipeline_sends_local_requests_without_hosted_fallback(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    responses = iter([
        "</think>no json here",
        '<json>{"data": [["CCO", "CC(=O)O"]], "explanation": ["esterification"], '
        '"confidence_scores": [0.8]}</json>',
    ])

    def fake_completion(**params: Any) -> Any:
        calls.append(params)
        message = SimpleNamespace(content=next(responses))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    monkeypatch.setattr(llm_interface, "completion", fake_completion)

    pathways, explanations, confidence = llm_pipeline(
        "CCOC(C)=O", model=LOCAL_MODEL, enable_thinking=False, max_output_tokens=1024
    )

    assert pathways == [["CCO", "CC(=O)O"]]
    assert (explanations, confidence) == (["esterification"], [0.8])
    assert [c["model"] for c in calls] == [LOCAL_MODEL, LOCAL_MODEL]
    assert [c["temperature"] for c in calls] == [0.0, 0.1]
    assert calls[0]["messages"][0]["content"] == SYS_PROMPT_OPENAI


def test_score_record_matches_notebook_metrics() -> None:
    base = {"input": "CCOC(C)=O", "output": "CCO.CC(=O)O", "model_id": LOCAL_MODEL}
    raw_hit_second = (
        '<json>{"data": [["CCO"], ["OCC", "OC(C)=O"]], '
        '"explanation": ["a", "b"], "confidence_scores": [0.5, 0.4]}</json>'
    )
    rows = [
        score_record({**base, "mol_no": 0, "raw": raw_hit_second}),
        score_record({**base, "mol_no": 1, "raw": "garbage"}),
        score_record({**base, "mol_no": 2, "proposals": [["CC(=O)O", "CCO"]]}),
    ]

    assert (rows[0]["all_correct"], rows[0]["all_correct_topk"], rows[0]["first_hit_rank"]) == (0, 1, 2)
    assert rows[0]["maxfrag_correct"] == 0
    assert (rows[1]["parse_status"], rows[1]["all_correct_topk"]) == (502, 0)
    assert (rows[2]["all_correct"], rows[2]["maxfrag_correct"]) == (1, 1)

    summary, report = summarize(pd.DataFrame(rows), {"model_id": LOCAL_MODEL, "mode": "raw"})
    assert summary["top1"]["n_all_correct"] == 1
    assert summary["topk"]["n_all_correct"] == 2
    assert "parse failures" in report


def _mol(smiles: str, children: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"type": "mol", "smiles": smiles, "children": children or []}


def _route(product: str, reactants: list[str], deeper: dict[str, Any] | None = None) -> dict[str, Any]:
    kids = [_mol(r) for r in reactants]
    if deeper is not None:
        kids[0] = deeper
    return _mol(product, [{"type": "reaction", "children": kids}])


def test_az_first_step_proposals_takes_only_the_first_reaction() -> None:
    from deepretro.utils.one_step_eval import az_first_step_proposals

    # Two-level route: only the top reaction's reactants are the single step.
    deep = _route("CCOC(C)=O", ["OCC", "CC(=O)Cl"],
                  deeper=_route("OCC", ["C=C", "O"]))
    assert az_first_step_proposals([deep]) == [["OCC", "CC(=O)Cl"]]
    # Same first step spelled differently in a second route: kept once.
    same = _route("CCOC(C)=O", ["CC(=O)Cl", "CCO"])
    other = _route("CCOC(C)=O", ["CCO", "CC(=O)O"])
    assert az_first_step_proposals([deep, same, other]) == [
        ["OCC", "CC(=O)Cl"], ["CCO", "CC(=O)O"],
    ]
    # Basic / in-stock target: no reaction, no step.
    assert az_first_step_proposals([{"type": "mol", "smiles": "O", "in_stock": True}]) == []


@pytest.mark.parametrize(
    ("az_rec", "source"),
    [
        ({"solved": True, "proposals": [["CCO", "CC(=O)O"]]}, "az"),
        ({"solved": False, "proposals": []}, "llm"),
        ({"solved": False, "proposals": [], "error": "RuntimeError: boom"}, "llm"),
        ({"solved": True, "proposals": []}, "llm"),  # solved but no first step
    ],
)
def test_hybrid_view_uses_az_only_when_it_gives_a_first_step(
    az_rec: dict[str, Any], source: str
) -> None:
    from deepretro.utils.one_step_eval import hybrid_views

    base = {"mol_no": 0, "input": "CCOC(C)=O", "output": "CCO.CC(=O)O", "model_id": LOCAL_MODEL}
    llm_rec = {"proposals": [["CCO"]]}
    views = hybrid_views(az_rec, llm_rec, base)

    assert views["hybrid"]["source"] == source
    expected = az_rec["proposals"] if source == "az" else llm_rec["proposals"]
    assert views["hybrid"]["proposals"] == expected
    assert views["llm"]["proposals"] == [["CCO"]]
    assert score_record(views["hybrid"])["all_correct"] == int(source == "az")


def test_write_summary_logs_one_row_per_view(tmp_path: Any) -> None:
    from deepretro.utils.one_step_eval import write_summary

    base = {"mol_no": 0, "input": "CCOC(C)=O", "output": "CCO.CC(=O)O", "model_id": LOCAL_MODEL}
    scored = pd.DataFrame([score_record({**base, "proposals": [["CCO", "CC(=O)O"]]})])
    views = {v: summarize(scored, {"view": v})[0] for v in ("hybrid", "az", "llm")}
    log = tmp_path / "all.csv"

    write_summary({"views": views}, "report", str(tmp_path), str(log), views=views)

    logged = pd.read_csv(log)
    assert list(logged["view"]) == ["hybrid", "az", "llm"]
    assert list(logged["top1_all_correct"]) == [100.0, 100.0, 100.0]
