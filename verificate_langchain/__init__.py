"""verificate-langchain — a veto gate for AI-written code in LangChain / LangGraph.

Verificate is the merge gate for AI-written work: 17 deterministic reality gates
(mock/placeholder veto, invented-API checks, false-completion detection) plus a
frontier-model review, with veto power. An agent that ships confident-but-wrong
code loses its user's trust; this is the safety net that keeps it shipping.

No signup: every machine gets 100 free validations. After that, pass a token (30-day
trial at https://verificate.ai/auth/signup).

Quick start:

    from verificate_langchain import load_verificate_tools, validate

    # one-shot check
    verdict = await validate("def refund(x):\\n    pass  # TODO real refund")
    print(verdict.approved, verdict.findings)

    # or load the tools straight into a LangGraph agent
    tools = await load_verificate_tools()          # + your other tools
    agent = create_react_agent(model, tools)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__version__ = "0.1.1"

VERIFICATE_MCP_URL = "https://mcp.verificate.ai/mcp"


@dataclass
class Verdict:
    """A validation result. `approved` is authoritative — a deterministic gate veto
    cannot be argued past; the flagged content itself must change."""
    approved: bool
    score: float | None = None
    findings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:      # `if verdict:` reads as "is it approved"
        return self.approved


def _client(token: str | None):
    from langchain_mcp_adapters.client import MultiServerMCPClient
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return MultiServerMCPClient(
        {"verificate": {"url": VERIFICATE_MCP_URL,
                        "transport": "streamable_http", "headers": headers}}
    )


async def load_verificate_tools(token: str | None = None) -> list:
    """Return Verificate's MCP tools as LangChain tools (validate_ai_output,
    validate_plan, analyze_code, generate_code) — add them to any agent."""
    return await _client(token).get_tools()


def _texts(result: Any) -> list[str]:
    """Every text payload in a tool result, whatever shape the adapter hands back: a str (adapters
    <0.2), a list of content-block DICTS (0.3.x: [{'type': 'text', 'text': ...}]), a list of objects
    with .text, or a (content, artifact) tuple. 0.1.0 only handled str and objects, so on current
    adapters EVERY verdict — including approvals — came back as "could not parse verdict"."""
    if isinstance(result, str):
        return [result]
    if isinstance(result, tuple) and result:
        return _texts(result[0])
    if isinstance(result, dict):
        return [result["text"]] if isinstance(result.get("text"), str) else []
    if isinstance(result, (list,)):
        out: list[str] = []
        for block in result:
            out.extend(_texts(block) if isinstance(block, (dict, str, list, tuple))
                       else ([block.text] if isinstance(getattr(block, "text", None), str) else []))
        return out
    text = getattr(result, "text", None) or getattr(result, "content", None)
    return _texts(text) if text is not None and text is not result else []


def _parse(result: Any) -> Verdict:
    p = None
    for text in _texts(result):                      # the verdict is the first block that is a JSON object
        try:
            candidate = json.loads(text)
        except Exception:
            continue
        if isinstance(candidate, dict):
            p = candidate
            break
    if p is None:
        return Verdict(approved=False, findings=["could not parse verdict"], raw={"text": str(result)[:2000]})
    if p.get("validation_type") == "quota" or p.get("review_unavailable"):
        # Not a judgement of the code: the gate refused access (quota / key) or could not complete
        # the review. Still not approved — but say why, so callers don't "fix" code that is fine.
        why = "review did not complete — retry" if p.get("review_unavailable") else \
            str((p.get("suggestions") or p.get("issues") or ["access refused"])[0])
        return Verdict(approved=False, score=None, findings=["gate unavailable: " + why],
                       suggestions=list(p.get("suggestions", [])), raw=p)
    return Verdict(
        approved=bool(p.get("valid")),
        score=p.get("score"),
        findings=list(p.get("issues", [])),
        suggestions=list(p.get("suggestions", [])),
        raw=p,
    )


async def validate(code: str, *, validation_type: str = "code_generation",
                   token: str | None = None) -> Verdict:
    """Gate one piece of output. Returns a Verdict (truthy iff approved).

    validation_type: 'code_generation' (default) for source; 'documentation',
    'report', 'plan', ... for prose/designs."""
    tools = {t.name: t for t in await load_verificate_tools(token)}
    result = await tools["validate_ai_output"].ainvoke(
        {"ai_output": code, "validation_type": validation_type})
    return _parse(result)


class VerificateGuardrail:
    """A callable veto gate for LangGraph. Use as a node, or as an edge condition.

        gate = VerificateGuardrail(token=os.getenv("VERIFICATE_TOKEN"))
        graph.add_node("gate", gate.node)          # sets state['verdict']/['findings']
        graph.add_conditional_edges("gate", gate.route, {"fix": "write", "done": END})
    """

    def __init__(self, token: str | None = None, max_attempts: int = 3,
                 code_key: str = "code"):
        self.token = token
        self.max_attempts = max_attempts
        self.code_key = code_key

    async def node(self, state: dict) -> dict:
        v = await validate(state[self.code_key], token=self.token)
        return {**state,
                "verdict": "approved" if v.approved else "rejected",
                "findings": v.findings,
                "attempts": state.get("attempts", 0) + 1}

    def route(self, state: dict) -> str:
        if state.get("verdict") == "approved":
            return "done"
        return "fix" if state.get("attempts", 0) < self.max_attempts else "give_up"


__all__ = ["Verdict", "load_verificate_tools", "validate", "VerificateGuardrail",
           "VERIFICATE_MCP_URL", "__version__"]
