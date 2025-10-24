"""High level orchestration utilities for multi-asset portfolio workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from tradingagents.dataflows.macro import MacroSnapshot, gather_macro_snapshot
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.portfolio.state import PortfolioSnapshot, empty_portfolio
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

    def _build_portfolio_context(self, snapshot: PortfolioSnapshot, symbol: str) -> Dict[str, Any]:
        positions_payload = {
            pos.symbol: {
                "quantity": pos.quantity,
                "average_cost": pos.average_cost,
                "market_value": pos.market_value,
            }
            for pos in snapshot.iter_positions()
        }
        portfolio_cfg = self.config.get("portfolio", {})
        return {
            "as_of": snapshot.as_of.isoformat(),
            "cash": snapshot.cash,
            "total_equity": snapshot.total_equity,
            "positions": positions_payload,
            "constraints": {
                "max_single_position_pct": portfolio_cfg.get("max_single_position_pct"),
                "max_gross_exposure_pct": portfolio_cfg.get("max_gross_exposure_pct"),
            },
            "focus_symbol": symbol,
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
            },
        )

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

        opportunities: List[TradeOpportunity] = []
        graph_results: List[Dict[str, Any]] = []

        for symbol in tickers:
            graph = self._graph_factory()
            portfolio_context = self._build_portfolio_context(portfolio_snapshot, symbol)
            final_state, processed_signal = graph.propagate(
                symbol,
                trade_date,
                portfolio_context=portfolio_context,
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
                "trade_opportunities": [item.to_dict() for item in opportunities],
            },
        }


__all__ = ["PortfolioOrchestrator", "TradeOpportunity"]
