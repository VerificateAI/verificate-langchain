"""verificate-langchain — a veto gate for AI-written code in LangChain / LangGraph.

Verificate is the merge gate for AI-written work: 17 deterministic reality gates
(mock/placeholder veto, invented-API checks, false-completion detection) plus a
frontier-model review, with veto power. An agent that ships confident-but-wrong
code loses its user's trust; this is the safety net that keeps it shipping.

No signup: every machine gets 25 free validations. After that, pass a token (30-day
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

__version__ = "0.1.0"

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


def _parse(result: Any) -> Verdict:
    text = result if isinstance(result, str) else getattr(result[0], "text", str(result))
    try:
        p = json.loads(text)
    except Exception:
        return Verdict(approved=False, findings=["could not parse verdict"], raw={"text": text})
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
