"""Portfolio management primitives for TradingAgents."""

from importlib import import_module
from typing import Any

from .state import PortfolioSnapshot, Position, TransactionRecord, empty_portfolio
from .feedback import generate_portfolio_feedback
from .reporting import update_performance_report
from .storage import PortfolioStorage
from .risk import PortfolioRiskMetrics, compute_portfolio_risk_metrics

__all__ = [
    "PortfolioSnapshot",
    "Position",
    "TransactionRecord",
    "PortfolioStorage",
    "empty_portfolio",
    "PortfolioOrchestrator",
    "TradeOpportunity",
    "PortfolioRiskMetrics",
    "compute_portfolio_risk_metrics",
    "generate_portfolio_feedback",
    "update_performance_report",
]


def __getattr__(name: str) -> Any:
    if name in {"PortfolioOrchestrator", "TradeOpportunity"}:
        module = import_module(".orchestrator", __name__)
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
