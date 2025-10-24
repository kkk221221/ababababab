# TradingAgents/graph/propagation.py

from typing import Dict, Any, Mapping, Optional
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        portfolio_context: Optional[Mapping[str, Any]] = None,
        portfolio_feedback: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph."""
        state = {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date": str(trade_date),
            "investment_debate_state": InvestDebateState(
                {"history": "", "current_response": "", "count": 0}
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "history": "",
                    "current_risky_response": "",
                    "current_safe_response": "",
                    "current_neutral_response": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            "portfolio_context": None,
            "portfolio_feedback": None,
            "trader_structured_plan": None,
        }
        if portfolio_context is not None:
            state["portfolio_context"] = dict(portfolio_context)
        if portfolio_feedback is not None:
            state["portfolio_feedback"] = dict(portfolio_feedback)
        return state

    def get_graph_args(self) -> Dict[str, Any]:
        """Get arguments for the graph invocation."""
        return {
            "stream_mode": "values",
            "config": {"recursion_limit": self.max_recur_limit},
        }
