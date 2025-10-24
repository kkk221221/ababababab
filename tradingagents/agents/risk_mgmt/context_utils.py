"""Helper utilities for injecting portfolio risk context into prompts."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional


def _round_float(value: Any) -> Any:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return round(value, 6)
    return value


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(key): _sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(item) for item in obj]
    return _round_float(obj)


def serialize_portfolio_context(portfolio_context: Optional[Mapping[str, Any]]) -> str:
    """Convert the portfolio context into a JSON string safe for LLM prompts."""

    if not portfolio_context:
        return "{}"
    sanitized = _sanitize(portfolio_context)
    try:
        return json.dumps(sanitized, indent=2, sort_keys=True)
    except TypeError:
        return json.dumps({}, indent=2)


def summarize_portfolio_risk(portfolio_context: Optional[Mapping[str, Any]]) -> str:
    """Human-readable overview highlighting the most important risk metrics."""

    if not portfolio_context:
        return "No portfolio risk context provided."

    risk_metrics = portfolio_context.get("risk_metrics") or {}
    exposure = portfolio_context.get("exposure") or {}
    budgets = portfolio_context.get("budgets") or {}

    summary_parts = []

    as_of = portfolio_context.get("as_of")
    total_equity = portfolio_context.get("total_equity")
    cash = portfolio_context.get("cash")
    if as_of or total_equity or cash:
        summary_parts.append(
            f"Snapshot as of {as_of or 'n/a'} with equity ~{total_equity} and cash {cash}."
        )

    beta = risk_metrics.get("beta")
    benchmark = risk_metrics.get("benchmark_symbol", "benchmark")
    if beta is not None:
        summary_parts.append(f"Beta vs {benchmark}: {beta}.")

    var_pct = risk_metrics.get("value_at_risk_pct")
    if var_pct is not None:
        confidence = risk_metrics.get("var_confidence", 0.95)
        summary_parts.append(
            f"One-day VaR at {int(confidence * 100)}% confidence: {round(var_pct * 100, 2)}%."
        )

    sharpe = risk_metrics.get("sharpe_ratio")
    if sharpe is not None:
        summary_parts.append(f"Annualised Sharpe ratio: {round(sharpe, 2)}.")

    gross = exposure.get("gross_exposure_pct")
    if gross is not None:
        summary_parts.append(f"Gross exposure as % of equity: {round(gross * 100, 2)}%.")

    risk_budget = budgets.get("risk_budget")
    if risk_budget is not None:
        summary_parts.append(f"Risk budget allocated for this trade: {risk_budget}.")

    sector_exposure = risk_metrics.get("sector_exposure") or {}
    if sector_exposure:
        formatted = ", ".join(
            f"{sector}: {round(pct * 100, 2)}%" for sector, pct in sector_exposure.items()
        )
        summary_parts.append(f"Sector exposure weights: {formatted}.")

    if not summary_parts:
        return "No quantitative risk metrics available."

    return " \n".join(summary_parts)


__all__ = ["serialize_portfolio_context", "summarize_portfolio_risk"]
