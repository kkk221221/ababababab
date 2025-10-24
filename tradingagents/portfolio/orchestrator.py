"""High level orchestration utilities for multi-asset portfolio workflows."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from tradingagents.dataflows.macro import MacroSnapshot, gather_macro_snapshot
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.portfolio.state import (
    PortfolioSnapshot,
    Position,
    TransactionRecord,
    empty_portfolio,
)
from tradingagents.portfolio.feedback import generate_portfolio_feedback
from tradingagents.portfolio.reporting import update_performance_report
from tradingagents.portfolio.risk import compute_portfolio_risk_metrics
from tradingagents.portfolio.storage import PortfolioStorage


def _clone_portfolio_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)
    portfolio_cfg = dict(cfg.get("portfolio", {}))
    cfg["portfolio"] = portfolio_cfg
    return cfg


def _infer_side(decision_text: str) -> str:
    upper_text = decision_text.upper()
    if "SELL" in upper_text and "BUY" not in upper_text:
        return "SELL"
    if "BUY" in upper_text and "SELL" not in upper_text:
        return "BUY"
    if "SHORT" in upper_text and "BUY" not in upper_text:
        return "SELL"
    if "LONG" in upper_text and "SELL" not in upper_text:
        return "BUY"
    if "HOLD" in upper_text:
        return "HOLD"
    return "REVIEW"


def _extract_confidence(decision_text: str) -> Optional[float]:
    match = re.search(r"(confidence|conviction)[^0-9]*([0-9]+(?:\.[0-9]+)?)%?", decision_text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(2)) / (100.0 if "%" in match.group(0) else 1.0)
        except ValueError:
            return None
    return None


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _estimate_correlations(
    focus_symbol: str,
    snapshot: PortfolioSnapshot,
    *,
    trade_date: str,
    lookback_days: int,
) -> Dict[str, float]:
    """Estimate correlations between the focus symbol and existing holdings."""

    other_symbols = [
        position.symbol
        for position in snapshot.iter_positions()
        if position.symbol != focus_symbol
    ]
    if not other_symbols:
        return {}

    try:
        import pandas as pd
        import yfinance as yf
    except ImportError:
        return {}

    try:
        end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        end_dt = snapshot.as_of
    start_dt = end_dt - timedelta(days=max(lookback_days, 1) * 2)

    tickers = sorted({focus_symbol, *other_symbols})
    price_series: List[pd.Series] = []
    for ticker in tickers:
        history = yf.Ticker(ticker).history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        if history.empty or "Close" not in history:
            continue
        series = history["Close"].rename(ticker).dropna()
        if not series.empty:
            price_series.append(series)

    if not price_series:
        return {}

    combined = pd.concat(price_series, axis=1).dropna(how="any")
    if combined.empty or focus_symbol not in combined.columns:
        return {}

    returns = combined.pct_change().dropna(how="any")
    if returns.empty:
        return {}

    corr_series = returns.corr().get(focus_symbol)
    if corr_series is None:
        return {}

    correlations: Dict[str, float] = {}
    for symbol, value in corr_series.items():
        if symbol == focus_symbol:
            continue
        if isinstance(value, float) and not math.isnan(value):
            correlations[symbol] = float(round(value, 4))
    return correlations


@dataclass
class TradeOpportunity:
    """Structured representation of a single-ticker agent recommendation."""

    symbol: str
    trade_date: str
    side: str
    summary: str
    trader_plan: str
    risk_summary: str
    confidence: Optional[float]
    source_state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "symbol": self.symbol,
            "trade_date": self.trade_date,
            "side": self.side,
            "summary": self.summary,
            "trader_plan": self.trader_plan,
            "risk_summary": self.risk_summary,
            "confidence": self.confidence,
            "source_state": self.source_state,
        }
        return payload


class PortfolioOrchestrator:
    """Coordinates macro intake, per-ticker analysis, and trade aggregation."""

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        storage: Optional[PortfolioStorage] = None,
        graph_factory: Optional[Callable[[], TradingAgentsGraph]] = None,
    ) -> None:
        base_config = _clone_portfolio_config(config or {})
        self.config: Dict[str, Any] = base_config
        if storage is None:
            portfolio_cfg = self.config.get("portfolio", {})
            self.storage = PortfolioStorage(
                self.config.get("results_dir", "./results"),
                snapshot_filename=portfolio_cfg.get("snapshot_filename"),
                transactions_filename=portfolio_cfg.get("transactions_filename"),
                feedback_filename=portfolio_cfg.get("feedback_filename"),
                lessons_filename=portfolio_cfg.get("lessons_filename"),
                nav_history_filename=portfolio_cfg.get("nav_history_filename"),
                performance_filename=portfolio_cfg.get("performance_filename"),
            )
        else:
            self.storage = storage
        if graph_factory is None:
            self._graph_factory = lambda: TradingAgentsGraph(config=self.config)
        else:
            self._graph_factory = graph_factory

    def load_portfolio(self, *, default_cash: Optional[float] = None) -> PortfolioSnapshot:
        snapshot = self.storage.load_snapshot()
        if snapshot is None:
            starting_cash = (
                default_cash
                if default_cash is not None
                else float(self.config.get("portfolio", {}).get("starting_cash", 0.0))
            )
            snapshot = empty_portfolio(starting_cash)
        return snapshot

    def _build_portfolio_context(
        self,
        snapshot: PortfolioSnapshot,
        symbol: str,
        trade_date: str,
        *,
        risk_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        portfolio_cfg = self.config.get("portfolio", {})
        total_equity = snapshot.total_equity or snapshot.cash
        max_single_pct = portfolio_cfg.get("max_single_position_pct")
        risk_per_trade_pct = portfolio_cfg.get("risk_per_trade_pct", 0.01)

        positions_payload: Dict[str, Dict[str, Any]] = {}
        gross_exposure = 0.0
        focus_position: Optional[Position] = None
        for pos in snapshot.iter_positions():
            exposure_pct = _safe_div(pos.market_value, total_equity)
            positions_payload[pos.symbol] = {
                "quantity": pos.quantity,
                "average_cost": pos.average_cost,
                "market_value": pos.market_value,
                "exposure_pct": exposure_pct,
            }
            gross_exposure += abs(pos.market_value)
            if pos.symbol == symbol:
                focus_position = pos

        focus_value = focus_position.market_value if focus_position else 0.0
        focus_pct = _safe_div(focus_value, total_equity)
        gross_exposure_pct = _safe_div(gross_exposure, total_equity)

        max_single_value = (
            total_equity * float(max_single_pct)
            if max_single_pct not in (None, 0)
            else None
        )
        remaining_allocation = None
        if max_single_value is not None:
            remaining_allocation = max(0.0, max_single_value - focus_value)

        available_cash = snapshot.cash
        allocatable_cash = available_cash
        if remaining_allocation is not None:
            allocatable_cash = min(available_cash, remaining_allocation)
        risk_budget = min(
            allocatable_cash,
            total_equity * float(risk_per_trade_pct) if total_equity > 0 else allocatable_cash,
        )

        correlation_lookback = int(portfolio_cfg.get("correlation_lookback_days", 60))
        correlations = _estimate_correlations(
            symbol,
            snapshot,
            trade_date=trade_date,
            lookback_days=correlation_lookback,
        )

        risk_metrics_payload: Dict[str, Any] = risk_metrics or {}
        if not risk_metrics_payload:
            try:
                risk_metrics_obj = compute_portfolio_risk_metrics(
                    snapshot,
                    trade_date=trade_date,
                    benchmark_symbol=str(
                        portfolio_cfg.get("risk_benchmark_symbol", "SPY")
                    ),
                    lookback_days=int(portfolio_cfg.get("risk_lookback_days", 180)),
                    confidence=float(portfolio_cfg.get("var_confidence", 0.95)),
                    risk_free_rate=float(portfolio_cfg.get("risk_free_rate", 0.02)),
                )
                risk_metrics_payload = risk_metrics_obj.to_dict()
            except Exception:
                risk_metrics_payload = {}

        return {
            "as_of": snapshot.as_of.isoformat(),
            "cash": snapshot.cash,
            "total_equity": total_equity,
            "positions": positions_payload,
            "constraints": {
                "max_single_position_pct": max_single_pct,
                "max_gross_exposure_pct": portfolio_cfg.get("max_gross_exposure_pct"),
                "risk_per_trade_pct": risk_per_trade_pct,
                "atr_position_multiple": portfolio_cfg.get("atr_position_multiple"),
                "min_trade_notional": portfolio_cfg.get("min_trade_notional"),
                "max_var_pct": portfolio_cfg.get("max_var_pct"),
                "max_portfolio_beta": portfolio_cfg.get("max_portfolio_beta"),
                "min_portfolio_sharpe": portfolio_cfg.get("min_portfolio_sharpe"),
                "max_sector_exposure_pct": portfolio_cfg.get("max_sector_exposure_pct"),
            },
            "budgets": {
                "available_cash": available_cash,
                "max_single_position_value": max_single_value,
                "remaining_allocation": remaining_allocation,
                "risk_budget": risk_budget,
                "existing_position_value": focus_value,
            },
            "exposure": {
                "focus_position_value": focus_value,
                "focus_position_pct": focus_pct,
                "gross_exposure_pct": gross_exposure_pct,
            },
            "correlations": correlations,
            "focus_symbol": symbol,
            "risk_metrics": risk_metrics_payload,
            "feedback": self.storage.load_latest_feedback() or {},
            "performance_report": self.storage.load_performance_report() or {},
        }

    def _aggregate_opportunity(
        self,
        symbol: str,
        trade_date: str,
        final_state: Mapping[str, Any],
    ) -> TradeOpportunity:
        final_decision = _coerce_text(final_state.get("final_trade_decision"))
        trader_plan = _coerce_text(final_state.get("trader_investment_plan"))
        risk_state = final_state.get("risk_debate_state")
        risk_summary = ""
        if isinstance(risk_state, Mapping):
            risk_summary = _coerce_text(risk_state.get("judge_decision"))
        if not risk_summary:
            risk_summary = final_decision
        confidence = _extract_confidence(" ".join([final_decision, trader_plan, risk_summary]))
        return TradeOpportunity(
            symbol=symbol,
            trade_date=trade_date,
            side=_infer_side(final_decision or trader_plan or risk_summary),
            summary=final_decision,
            trader_plan=trader_plan,
            risk_summary=risk_summary,
            confidence=confidence,
            source_state={
                "market_report": final_state.get("market_report"),
                "sentiment_report": final_state.get("sentiment_report"),
                "news_report": final_state.get("news_report"),
                "fundamentals_report": final_state.get("fundamentals_report"),
                "investment_plan": final_state.get("investment_plan"),
                "trader_structured_plan": final_state.get("trader_structured_plan"),
            },
        )

    def apply_execution(self, execution: Mapping[str, Any]) -> PortfolioSnapshot:
        """Persist an approved trade execution, update reports, and log feedback."""

        snapshot = self.load_portfolio()
        portfolio_cfg = self.config.get("portfolio", {})
        commission_per_share = float(portfolio_cfg.get("commission_per_share", 0.0))
        default_slippage = float(portfolio_cfg.get("default_slippage_bps", 0.0))

        symbol = str(execution.get("symbol", "")).upper()
        side = str(execution.get("side", "")).upper()
        quantity = float(execution.get("quantity", 0.0))
        price = float(execution.get("price", 0.0))
        if not symbol or side not in {"BUY", "SELL"} or quantity <= 0 or price <= 0:
            return snapshot

        timestamp_raw = execution.get("timestamp")
        if isinstance(timestamp_raw, datetime):
            timestamp = timestamp_raw
        else:
            try:
                timestamp = datetime.fromisoformat(str(timestamp_raw))
            except (TypeError, ValueError):
                timestamp = datetime.utcnow()

        transaction = TransactionRecord(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            fees=float(execution.get("fees", 0.0)),
            slippage_bps=float(execution.get("slippage_bps", default_slippage)),
        )

        snapshot.apply_transaction(
            transaction,
            commission_per_share=commission_per_share,
        )
        self.storage.save_snapshot(snapshot)
        self.storage.record_transaction(transaction)

        risk_metrics = None
        try:
            risk_metrics = compute_portfolio_risk_metrics(
                snapshot,
                trade_date=transaction.timestamp.strftime("%Y-%m-%d"),
                benchmark_symbol=str(
                    portfolio_cfg.get("risk_benchmark_symbol", "SPY")
                ),
                lookback_days=int(portfolio_cfg.get("risk_lookback_days", 180)),
                confidence=float(portfolio_cfg.get("var_confidence", 0.95)),
                risk_free_rate=float(portfolio_cfg.get("risk_free_rate", 0.02)),
            )
        except Exception:
            risk_metrics = None

        feedback_entry = generate_portfolio_feedback(
            snapshot,
            config=self.config,
            transaction=transaction,
            risk_metrics=risk_metrics,
        )
        history_limit = int(portfolio_cfg.get("feedback_history_limit", 0))
        self.storage.append_feedback(
            feedback_entry,
            history_limit=history_limit if history_limit > 0 else None,
        )
        lessons = feedback_entry.get("lessons", [])
        if lessons:
            self.storage.append_lessons(lessons)
        update_performance_report(self.storage, snapshot)
        return snapshot

    def run_universe(
        self,
        tickers: Sequence[str],
        trade_date: str,
        *,
        macro_snapshot: Optional[MacroSnapshot] = None,
        portfolio_snapshot: Optional[PortfolioSnapshot] = None,
    ) -> Dict[str, Any]:
        if portfolio_snapshot is None:
            portfolio_snapshot = self.load_portfolio()
        if macro_snapshot is None:
            macro_snapshot = gather_macro_snapshot(trade_date, config=self.config)

        portfolio_cfg = self.config.get("portfolio", {})
        risk_metrics_payload: Dict[str, Any] = {}
        try:
            risk_metrics_obj = compute_portfolio_risk_metrics(
                portfolio_snapshot,
                trade_date=trade_date,
                benchmark_symbol=str(
                    portfolio_cfg.get("risk_benchmark_symbol", "SPY")
                ),
                lookback_days=int(portfolio_cfg.get("risk_lookback_days", 180)),
                confidence=float(portfolio_cfg.get("var_confidence", 0.95)),
                risk_free_rate=float(portfolio_cfg.get("risk_free_rate", 0.02)),
            )
            risk_metrics_payload = risk_metrics_obj.to_dict()
        except Exception:
            risk_metrics_payload = {}

        opportunities: List[TradeOpportunity] = []
        graph_results: List[Dict[str, Any]] = []
        latest_feedback = self.storage.load_latest_feedback()

        for symbol in tickers:
            graph = self._graph_factory()
            portfolio_context = self._build_portfolio_context(
                portfolio_snapshot,
                symbol,
                trade_date,
                risk_metrics=risk_metrics_payload,
            )
            final_state, processed_signal = graph.propagate(
                symbol,
                trade_date,
                portfolio_context=portfolio_context,
                portfolio_feedback=latest_feedback,
            )
            opportunities.append(self._aggregate_opportunity(symbol, trade_date, final_state))
            graph_results.append(
                {
                    "symbol": symbol,
                    "final_state": final_state,
                    "processed_signal": processed_signal,
                }
            )

        return {
            "portfolio_snapshot": portfolio_snapshot,
            "macro_snapshot": macro_snapshot,
            "trade_opportunities": [item.to_dict() for item in opportunities],
            "graph_results": graph_results,
            "portfolio_manager_input": {
                "portfolio_snapshot": portfolio_snapshot.to_dict(),
                "macro_snapshot": macro_snapshot.to_dict(),
                "risk_metrics": risk_metrics_payload,
                "trade_opportunities": [item.to_dict() for item in opportunities],
                "portfolio_feedback": latest_feedback or {},
            },
        }


__all__ = ["PortfolioOrchestrator", "TradeOpportunity"]
