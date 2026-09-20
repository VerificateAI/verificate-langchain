# verificate-langchain

[![Verificate Gate](https://img.shields.io/badge/gated%20by-Verificate%20Gate-2ea44f?logo=shield&logoColor=white)](https://github.com/VerificateAI/verificate-gate-action)


**A veto gate for AI-written code in LangChain / LangGraph.** Your agent writes code; Verificate runs it through 17 deterministic reality gates (mock/placeholder veto, invented-API checks, false-completion detection) plus a frontier-model review, and can **veto**. An agent that ships confident-but-wrong code loses its user's trust — this is the safety net that keeps it shipping.

```bash
pip install verificate-langchain
```

**100 free validations per machine — no signup, no token.** After that, pass a token (30-day trial, no card: https://verificate.ai/auth/signup).

## Measured

A frontier model asked *"is this OK to merge?"* missed reward-gaming (a test that only does `assert True`) and a hallucinated API in **0 of 6 runs each**. Verificate's gate catches both **6 / 6 — deterministically**, with 0 false positives on clean code. Battle-tested on 2,581 audited validations, including guarding the write-path of a 21M-entity source-cited knowledge base. Full benchmark: https://github.com/Verificate-Dev/verificate-mcp-quickstart/blob/master/COMPARISON.md

## One-shot check

```python
from verificate_langchain import validate

v = await validate("def refund(order_id, amount):\n    pass  # TODO real refund")
print(v.approved)     # False — the mock refund path is vetoed
print(v.findings)     # ['[code_reality_gate] Placeholder detected ...', ...]
```

`Verdict` is truthy iff approved, so `if await validate(code): ship(code)` reads naturally.

## As a LangGraph guardrail node

```python
from verificate_langchain import VerificateGuardrail
from langgraph.graph import StateGraph, START, END

gate = VerificateGuardrail(max_attempts=3)           # optional token=...

g = StateGraph(dict)
g.add_node("write", write_code)                       # your codegen node, sets state["code"]
g.add_node("gate", gate.node)                         # sets state["verdict"] / ["findings"]
g.add_edge(START, "write")
g.add_edge("write", "gate")
g.add_conditional_edges("gate", gate.route, {"fix": "write", "done": END, "give_up": END})
app = g.compile()
```

The output must *pass through* the gate node — the veto is structural, not an optional tool call the agent can skip under pressure.

## Load the raw tools

```python
from verificate_langchain import load_verificate_tools

tools = await load_verificate_tools()   # validate_ai_output, validate_plan, analyze_code, generate_code
agent = create_react_agent(model, tools + your_other_tools)
```

## What it catches (real, verbatim)

> *"N+1 synchronous API calls … 100 sequential HTTP roundtrips … will trigger Stripe rate limiting"* · *"`stripe.Inventory` is not a valid Stripe SDK resource"* · *"Stripe API requires integer cents"*

Read-only: code is analyzed, never executed, never trained on. https://verificate.ai/privacy

Uses the official [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters) against the hosted [Verificate MCP server](https://mcp.verificate.ai/mcp). All clients + one-click installs: https://github.com/Verificate-Dev/verificate-mcp-quickstart
