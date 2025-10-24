"""Feedback generation utilities for portfolio level loops."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from tradingagents.portfolio.risk import PortfolioRiskMetrics
from tradingagents.portfolio.state import PortfolioSnapshot, TransactionRecord


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _position_exposures(snapshot: PortfolioSnapshot) -> Dict[str, float]:
    total_equity = snapshot.total_equity or snapshot.cash or 0.0
    exposures: Dict[str, float] = {}
    if total_equity <= 0:
        return exposures
    for position in snapshot.iter_positions():
        exposures[position.symbol] = float(position.market_value or 0.0) / total_equity
    return exposures


def _gross_exposure(snapshot: PortfolioSnapshot) -> float:
    total_equity = snapshot.total_equity or snapshot.cash or 0.0
    if total_equity <= 0:
        return 0.0
    gross_value = sum(abs(float(pos.market_value or 0.0)) for pos in snapshot.iter_positions())
    return gross_value / total_equity if total_equity else 0.0


def _build_rebalance_actions(
    snapshot: PortfolioSnapshot,
    *,
    target_cash_buffer: float,
    tolerance: float,
    max_single_pct: Optional[float],
    max_gross_pct: Optional[float],
) -> Dict[str, Any]:
    total_equity = snapshot.total_equity or snapshot.cash or 0.0
    actions: Dict[str, Any] = {"cash": None, "positions": []}
    if total_equity <= 0:
        return actions

    cash_pct = (snapshot.cash / total_equity) if total_equity else 0.0
    if cash_pct < max(target_cash_buffer - tolerance, 0.0):
        shortfall = max(target_cash_buffer - cash_pct, 0.0)
        actions["cash"] = {
            "target_pct": target_cash_buffer,
            "current_pct": cash_pct,
            "suggested_raise": shortfall * total_equity,
        }

    exposures = _position_exposures(snapshot)
    for symbol, exposure_pct in exposures.items():
        if max_single_pct and exposure_pct > (max_single_pct + tolerance):
            actions["positions"].append(
                {
                    "symbol": symbol,
                    "action": "trim",
                    "current_pct": exposure_pct,
                    "limit_pct": max_single_pct,
                    "target_pct": max_single_pct - tolerance,
                }
            )

    gross_pct = _gross_exposure(snapshot)
    if max_gross_pct and gross_pct > (max_gross_pct + tolerance):
        actions.setdefault("alerts", []).append(
            {
                "type": "gross_exposure",
                "current_pct": gross_pct,
                "limit_pct": max_gross_pct,
            }
        )

    return actions


def _build_alerts(
    snapshot: PortfolioSnapshot,
    actions: Mapping[str, Any],
    *,
    risk_metrics: Optional[PortfolioRiskMetrics],
    portfolio_cfg: Mapping[str, Any],
) -> Iterable[str]:
    alerts = []
    cash_alert = actions.get("cash")
    if isinstance(cash_alert, Mapping):
        alerts.append(
            "Cash buffer below target: {current:.2f}% vs {target:.2f}%".format(
                current=(cash_alert.get("current_pct", 0.0) or 0.0) * 100,
                target=(cash_alert.get("target_pct", 0.0) or 0.0) * 100,
            )
        )

    for pos_action in actions.get("positions", []):
        if isinstance(pos_action, Mapping):
            alerts.append(
                "Position {symbol} exceeds limit ({current} > {limit})".format(
                    symbol=pos_action.get("symbol", ""),
                    current=_format_pct(float(pos_action.get("current_pct", 0.0))),
                    limit=_format_pct(float(pos_action.get("limit_pct", 0.0))),
                )
            )

    gross_actions = actions.get("alerts", [])
    if isinstance(gross_actions, list):
        for gross in gross_actions:
            if isinstance(gross, Mapping) and gross.get("type") == "gross_exposure":
                alerts.append(
                    "Gross exposure at {current} exceeds limit {limit}".format(
                        current=_format_pct(float(gross.get("current_pct", 0.0))),
                        limit=_format_pct(float(gross.get("limit_pct", 0.0))),
                    )
                )

    if risk_metrics is not None:
        max_var_pct = float(portfolio_cfg.get("max_var_pct", 0.0) or 0.0)
        if (
            max_var_pct
            and risk_metrics.value_at_risk_pct is not None
            and risk_metrics.value_at_risk_pct > max_var_pct
        ):
            alerts.append(
                f"Value-at-risk {risk_metrics.value_at_risk_pct:.2%} breaches limit of {max_var_pct:.2%}."
            )
        max_beta = float(portfolio_cfg.get("max_portfolio_beta", 0.0) or 0.0)
        if max_beta and risk_metrics.beta is not None and risk_metrics.beta > max_beta:
            alerts.append(
                f"Portfolio beta {risk_metrics.beta:.2f} above ceiling {max_beta:.2f}."
            )
        max_sector_pct = float(portfolio_cfg.get("max_sector_exposure_pct", 0.0) or 0.0)
        if max_sector_pct and risk_metrics.sector_exposure:
            for sector, pct in risk_metrics.sector_exposure.items():
                if pct > max_sector_pct:
                    alerts.append(
                        f"Sector {sector} exposure {_format_pct(pct)} above limit {_format_pct(max_sector_pct)}"
                    )

    return alerts


def generate_portfolio_feedback(
    snapshot: PortfolioSnapshot,
    *,
    config: Mapping[str, Any],
    transaction: TransactionRecord,
    risk_metrics: Optional[PortfolioRiskMetrics] = None,
) -> Dict[str, Any]:
    """Create structured feedback for downstream agents after an execution."""

    portfolio_cfg = config.get("portfolio", {})
    target_cash_buffer = float(portfolio_cfg.get("target_cash_buffer_pct", 0.0) or 0.0)
    tolerance = float(portfolio_cfg.get("rebalance_tolerance_pct", 0.0) or 0.0)
    max_single_pct = portfolio_cfg.get("max_single_position_pct")
    max_gross_pct = portfolio_cfg.get("max_gross_exposure_pct")

    rebalance_actions = _build_rebalance_actions(
        snapshot,
        target_cash_buffer=target_cash_buffer,
        tolerance=tolerance,
        max_single_pct=float(max_single_pct) if max_single_pct not in (None, "") else None,
        max_gross_pct=float(max_gross_pct) if max_gross_pct not in (None, "") else None,
    )

    alerts = list(
        _build_alerts(
            snapshot,
            rebalance_actions,
            risk_metrics=risk_metrics,
            portfolio_cfg=portfolio_cfg,
        )
    )

    total_equity = snapshot.total_equity or snapshot.cash or 0.0
    cash_pct = (snapshot.cash / total_equity) if total_equity else 0.0
    exposures = _position_exposures(snapshot)

    timestamp = snapshot.as_of.isoformat()
    trade_desc = (
        f"{transaction.side} {transaction.quantity:.2f} {transaction.symbol} at ${transaction.price:.2f}"
    )

    summary = (
        "Portfolio updated on {date} after {trade}. Cash {cash_pct}, equity ${equity:,.2f}, holdings: {holdings}."
    ).format(
        date=timestamp,
        trade=trade_desc,
        cash_pct=_format_pct(cash_pct),
        equity=total_equity,
        holdings=", ".join(
            f"{symbol} {_format_pct(pct)}" for symbol, pct in exposures.items()
        )
        or "no open positions",
    )

    analyst_brief = []
    if alerts:
        analyst_brief.append("Key alerts: " + "; ".join(alerts))
    if rebalance_actions.get("positions"):
        trims = ", ".join(
            f"{item['symbol']} ({_format_pct(float(item['current_pct']))})"
            for item in rebalance_actions.get("positions", [])
            if isinstance(item, Mapping)
        )
        if trims:
            analyst_brief.append(
                f"Seek alternative ideas to reduce concentration in: {trims}."
            )

    research_brief = []
    if rebalance_actions.get("positions"):
        for item in rebalance_actions.get("positions", []):
            if isinstance(item, Mapping):
                research_brief.append(
                    f"Investigate hedges or replacements for {item.get('symbol')} given exposure {_format_pct(float(item.get('current_pct', 0.0)))}."
                )
    if not research_brief and alerts:
        research_brief.append("Monitor macro/risk themes tied to current alerts.")

    trader_brief = [
        "Rebalance guidance: {actions}".format(
            actions="; ".join(alerts) if alerts else "portfolio within limits"
        )
    ]
    cash_plan = rebalance_actions.get("cash")
    if isinstance(cash_plan, Mapping):
        trader_brief.append(
            "Raise approximately ${amount:,.0f} to restore cash buffer to {target}.".format(
                amount=float(cash_plan.get("suggested_raise", 0.0) or 0.0),
                target=_format_pct(float(cash_plan.get("target_pct", 0.0) or 0.0)),
            )
        )

    manager_brief = [summary]
    if alerts:
        manager_brief.extend(alerts)

    risk_brief = []
    if risk_metrics is not None:
        risk_brief.append(
            (
                "Risk snapshot – VaR: {var}, Beta: {beta}, Sharpe: {sharpe}."
            ).format(
                var=f"{risk_metrics.value_at_risk_pct:.2%}" if risk_metrics.value_at_risk_pct is not None else "N/A",
                beta=f"{risk_metrics.beta:.2f}" if risk_metrics.beta is not None else "N/A",
                sharpe=f"{risk_metrics.sharpe_ratio:.2f}" if risk_metrics.sharpe_ratio is not None else "N/A",
            )
        )
    if alerts:
        risk_brief.extend(alerts)

    lessons = []
    if alerts:
        situation = (
            f"Portfolio exposures on {timestamp} after {trade_desc}: cash {_format_pct(cash_pct)}, holdings {exposures}."
        )
        for alert in alerts:
            lessons.append({"situation": situation, "recommendation": alert})

    feedback = {
        "timestamp": timestamp,
        "summary": summary,
        "alerts": alerts,
        "rebalance_actions": rebalance_actions,
        "audience": {
            "analysts": analyst_brief,
            "research": research_brief,
            "trader": trader_brief,
            "portfolio_manager": manager_brief,
            "risk": risk_brief,
        },
        "lessons": lessons,
    }

    return feedback


__all__ = ["generate_portfolio_feedback"]
