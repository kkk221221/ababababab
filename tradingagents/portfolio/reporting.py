"""Utilities for persisting portfolio performance history and reports."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Mapping, Optional

from tradingagents.portfolio.state import PortfolioSnapshot
from tradingagents.portfolio.storage import PortfolioStorage


def _parse_timestamp(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _compute_drawdown(nav_history: list[dict]) -> float:
    max_drawdown = 0.0
    peak = None
    for entry in nav_history:
        nav = float(entry.get("nav", 0.0) or 0.0)
        if peak is None or nav > peak:
            peak = nav
        if peak and peak > 0:
            drawdown = (nav - peak) / peak
            if drawdown < max_drawdown:
                max_drawdown = drawdown
    return max_drawdown


def _compute_cagr(nav_history: list[dict]) -> Optional[float]:
    if len(nav_history) < 2:
        return None
    start_nav = float(nav_history[0].get("nav", 0.0) or 0.0)
    end_nav = float(nav_history[-1].get("nav", 0.0) or 0.0)
    if start_nav <= 0 or end_nav <= 0:
        return None
    start_time = _parse_timestamp(nav_history[0].get("as_of", ""))
    end_time = _parse_timestamp(nav_history[-1].get("as_of", ""))
    if not start_time or not end_time or end_time <= start_time:
        return None
    days = (end_time - start_time).days or 1
    years = days / 365.25
    if years <= 0:
        return None
    return (end_nav / start_nav) ** (1 / years) - 1


def update_performance_report(
    storage: PortfolioStorage,
    snapshot: PortfolioSnapshot,
) -> Mapping[str, object]:
    """Append the latest NAV entry and refresh the performance summary."""

    nav_history = storage.load_nav_history()
    previous_nav = float(nav_history[-1]["nav"]) if nav_history else None

    nav = float(snapshot.total_equity or snapshot.cash or 0.0)
    entry = {
        "as_of": snapshot.as_of.isoformat(),
        "nav": nav,
        "cash": snapshot.cash,
        "market_value": snapshot.total_market_value,
    }
    if previous_nav and previous_nav > 0:
        entry["return"] = (nav - previous_nav) / previous_nav
    else:
        entry["return"] = None

    nav_history.append(entry)
    storage.save_nav_history(nav_history)

    max_drawdown = _compute_drawdown(nav_history)
    cagr = _compute_cagr(nav_history)
    total_return = None
    if len(nav_history) >= 2:
        start_nav = float(nav_history[0].get("nav", 0.0) or 0.0)
        if start_nav > 0:
            total_return = (nav / start_nav) - 1

    report: Dict[str, object] = {
        "last_updated": snapshot.as_of.isoformat(),
        "nav_history": nav_history,
        "metrics": {
            "max_drawdown": max_drawdown,
            "cagr": cagr,
            "total_return": total_return,
        },
    }

    storage.save_performance_report(report)
    return report


__all__ = ["update_performance_report"]
