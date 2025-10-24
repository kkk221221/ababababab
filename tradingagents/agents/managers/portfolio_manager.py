"""LLM-powered portfolio manager agent that allocates capital across ideas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence

from tradingagents.agents.utils.portfolio_feedback import format_portfolio_feedback
from tradingagents.portfolio.state import PortfolioSnapshot
from tradingagents.dataflows.macro import MacroSnapshot


PORTFOLIO_MANAGER_RESPONSE_SCHEMA: Dict[str, Any] = {
    "allocation_plan": {
        "total_budget": "float – USD allocated for the upcoming cycle",
        "cash_buffer_pct": "float – percentage of equity to keep as cash",
        "targets": [
            {
                "symbol": "ticker",
                "target_weight": "float between -1 and 1 where negative denotes short",
                "max_allocation": "float – USD ceiling for the position",
                "risk_budget": "float – percentage of overall risk budget",
                "confidence": "float 0-1 expressing conviction level",
                "notes": "string rationale"
            }
        ],
    },
    "portfolio_actions": {
        "rebalance": "string summary of required trades to achieve targets",
        "hedging": "string suggestions for hedges or overlays if needed",
        "watchlist": ["tickers requiring monitoring"],
    },
    "risk_overrides": {
        "constraints": "string summary of imposed limits",
        "follow_up_requests": "string instructions for other teams",
    },
}


PORTFOLIO_MANAGER_PROMPT = """You are the senior Portfolio Manager overseeing the multi-asset book.
You receive the current portfolio snapshot, macro context, and a list of trade proposals
coming from single-ticker agent pods. Produce a structured JSON response following the schema below.

Guidelines:
- Respect concentration limits and risk budgets implied by the current holdings.
- Highlight any conflicts between macro trends and proposed trades.
- Suggest cash buffers or hedges if the portfolio becomes unbalanced.
- Only return JSON – no Markdown fences or additional commentary.

Schema:
{schema}

Portfolio State:
{portfolio_state}

Macro Overview:
{macro_overview}

Trade Opportunities (proposals awaiting approval):
{trade_opportunities}

Portfolio Feedback from recent executions:
{portfolio_feedback}
"""


def _coerce_portfolio_snapshot(payload: Any) -> Optional[PortfolioSnapshot]:
    if isinstance(payload, PortfolioSnapshot):
        return payload
    if isinstance(payload, Mapping):
        try:
            return PortfolioSnapshot.from_dict(payload)
        except Exception:
            return None
    return None


def _coerce_macro_snapshot(payload: Any) -> Optional[MacroSnapshot]:
    if isinstance(payload, MacroSnapshot):
        return payload
    if isinstance(payload, Mapping) and "as_of" in payload:
        try:
            as_of_raw = payload.get("as_of")
            as_of = datetime.fromisoformat(str(as_of_raw)) if as_of_raw else datetime.utcnow()
            return MacroSnapshot(
                as_of=as_of,
                rates=list(payload.get("rates", [])),
                inflation=list(payload.get("inflation", [])),
                sectors=list(payload.get("sectors", [])),
                sentiment=dict(payload.get("sentiment", {})),
                errors=list(payload.get("errors", [])),
            )
        except Exception:
            return None
    return None


def _format_portfolio(snapshot: PortfolioSnapshot) -> str:
    lines = [f"Cash: {snapshot.cash:.2f}", f"Realized PnL: {snapshot.realized_pnl:.2f}"]
    lines.append("Positions:")
    if not snapshot.positions:
        lines.append("- None")
    for position in snapshot.positions.values():
        lines.append(
            "- {symbol}: qty={qty:.2f}, cost={cost:.2f}, price={price}, mv={mv:.2f}, unrealized={pnl:.2f}".format(
                symbol=position.symbol,
                qty=position.quantity,
                cost=position.average_cost,
                price=f"{position.market_price:.2f}" if position.market_price is not None else "N/A",
                mv=position.market_value,
                pnl=position.unrealized_pnl,
            )
        )
    lines.append(f"Total equity: {snapshot.total_equity:.2f}")
    return "\n".join(lines)


def _format_trade_opportunities(trade_opportunities: Sequence[Mapping[str, Any]]) -> str:
    if not trade_opportunities:
        return "(no pending trade ideas)"
    return json.dumps(list(trade_opportunities), indent=2)


def _extract_json_payload(response_text: str) -> Dict[str, Any]:
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = response_text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {"raw": response_text}
    return {"raw": response_text}


@dataclass
class PortfolioManagerOutput:
    """Typed container for portfolio manager responses."""

    raw_text: str
    parsed: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"raw_text": self.raw_text, "parsed": self.parsed}


def create_portfolio_manager(llm):
    """Create a portfolio manager callable compatible with the LangGraph nodes."""

    def portfolio_manager_node(state: Mapping[str, Any]) -> Dict[str, Any]:
        portfolio_snapshot = _coerce_portfolio_snapshot(state.get("portfolio_snapshot"))
        macro_snapshot = _coerce_macro_snapshot(state.get("macro_snapshot"))
        raw_opportunities = state.get("trade_opportunities", [])
        feedback_note = format_portfolio_feedback(
            state.get("portfolio_feedback"), "portfolio_manager"
        )
        if isinstance(raw_opportunities, Mapping):
            trade_opportunities: Sequence[Mapping[str, Any]] = [raw_opportunities]
        elif isinstance(raw_opportunities, Sequence):
            trade_opportunities = raw_opportunities  # type: ignore[assignment]
        else:
            trade_opportunities = []

        prompt = PORTFOLIO_MANAGER_PROMPT.format(
            schema=json.dumps(PORTFOLIO_MANAGER_RESPONSE_SCHEMA, indent=2),
            portfolio_state=_format_portfolio(portfolio_snapshot) if portfolio_snapshot else "Unavailable",
            macro_overview=macro_snapshot.to_prompt_block() if macro_snapshot else "Unavailable",
            trade_opportunities=_format_trade_opportunities(trade_opportunities),
            portfolio_feedback=feedback_note or "None provided",
        )

        response = llm.invoke(prompt)
        response_text = getattr(response, "content", "")
        parsed = _extract_json_payload(response_text)

        output = PortfolioManagerOutput(raw_text=response_text, parsed=parsed)

        return {
            "portfolio_manager_plan": output.to_dict(),
        }

    return portfolio_manager_node


__all__ = ["create_portfolio_manager", "PortfolioManagerOutput", "PORTFOLIO_MANAGER_RESPONSE_SCHEMA"]

