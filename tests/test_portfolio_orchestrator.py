import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch


class _RequestsResponseStub:
    text = ""

    def raise_for_status(self) -> None:
        raise RuntimeError("requests.get should not be invoked during tests")


if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace(
        get=lambda *args, **kwargs: _RequestsResponseStub()
    )


class _PandasStub:
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - defensive stub
        raise RuntimeError("pandas should not be invoked during tests")


if "pandas" not in sys.modules:
    sys.modules["pandas"] = _PandasStub()


def _relativedelta_stub(*args, **kwargs):  # pragma: no cover - defensive stub
    raise RuntimeError("dateutil.relativedelta should not be invoked during tests")


if "dateutil" not in sys.modules:
    sys.modules["dateutil"] = types.ModuleType("dateutil")
relativedelta_module = types.ModuleType("dateutil.relativedelta")
relativedelta_module.relativedelta = _relativedelta_stub
sys.modules["dateutil.relativedelta"] = relativedelta_module
sys.modules["dateutil"].relativedelta = relativedelta_module


if "tqdm" not in sys.modules:
    tqdm_module = types.ModuleType("tqdm")

    def _tqdm_stub(iterable, *args, **kwargs):  # pragma: no cover - defensive stub
        return iterable

    tqdm_module.tqdm = _tqdm_stub
    sys.modules["tqdm"] = tqdm_module


if "praw" not in sys.modules:
    praw_module = types.ModuleType("praw")
    sys.modules["praw"] = praw_module


class DummyMacroSnapshot:
    def __init__(self, as_of: datetime) -> None:
        self.as_of = as_of

    def to_dict(self) -> Dict[str, Any]:
        return {"as_of": self.as_of.isoformat()}

    def to_prompt_block(self) -> str:
        return f"Macro snapshot as of {self.as_of.isoformat()}"


def _gather_macro_snapshot_stub(
    trade_date: str,
    *,
    config: Dict[str, Any] | None = None,
    lookback_days: int = 30,
) -> DummyMacroSnapshot:
    try:
        as_of = datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        as_of = datetime.utcnow()
    return DummyMacroSnapshot(as_of)


macro_module = types.ModuleType("tradingagents.dataflows.macro")
macro_module.MacroSnapshot = DummyMacroSnapshot
macro_module.gather_macro_snapshot = _gather_macro_snapshot_stub
sys.modules["tradingagents.dataflows.macro"] = macro_module

graph_module = types.ModuleType("tradingagents.graph.trading_graph")


class _GraphStub:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - defensive stub
        pass

    def propagate(self, *args, **kwargs):  # pragma: no cover - defensive stub
        raise RuntimeError("TradingAgentsGraph stub should not be used in tests")


graph_module.TradingAgentsGraph = _GraphStub
sys.modules["tradingagents.graph.trading_graph"] = graph_module

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio.orchestrator import PortfolioOrchestrator
from tradingagents.portfolio.state import empty_portfolio
from tradingagents.portfolio.storage import PortfolioStorage


class DummyRiskMetrics:
    def __init__(self) -> None:
        self.as_of = datetime(2024, 1, 5)
        self.lookback_days = 30
        self.benchmark_symbol = "SPY"
        self.beta = 0.5
        self.value_at_risk_pct = 0.02
        self.var_confidence = 0.95
        self.sharpe_ratio = 1.15
        self.risk_free_rate = 0.02
        self.sector_exposure = {"Technology": 0.4}
        self.symbol_sectors = {"AAPL": "Technology"}
        self.invested_market_value = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "lookback_days": self.lookback_days,
            "benchmark_symbol": self.benchmark_symbol,
            "beta": self.beta,
            "value_at_risk_pct": self.value_at_risk_pct,
            "var_confidence": self.var_confidence,
            "sharpe_ratio": self.sharpe_ratio,
            "risk_free_rate": self.risk_free_rate,
            "sector_exposure": self.sector_exposure,
            "symbol_sectors": self.symbol_sectors,
            "invested_market_value": self.invested_market_value,
        }


class RecordingGraph:
    def __init__(self, recorder: List[Dict[str, Any]]) -> None:
        self.recorder = recorder

    def propagate(
        self,
        symbol: str,
        trade_date: str,
        *,
        portfolio_context: Dict[str, Any] | None = None,
        portfolio_feedback: Dict[str, Any] | None = None,
    ):
        self.recorder.append(
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "portfolio_context": portfolio_context,
                "portfolio_feedback": portfolio_feedback,
            }
        )
        final_state = {
            "final_trade_decision": f"BUY {symbol} with conviction 80%",
            "trader_investment_plan": f"Scale into {symbol} with staged limit orders.",
            "risk_debate_state": {"judge_decision": f"Risk team approves {symbol}"},
            "trader_structured_plan": {"symbol": symbol, "side": "BUY"},
        }
        return final_state, {"signal": "BUY"}


class PortfolioOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.results_path = Path(self.temp_dir.name)
        self.config = dict(DEFAULT_CONFIG)
        self.config["results_dir"] = str(self.results_path)
        self.config["portfolio"] = dict(DEFAULT_CONFIG.get("portfolio", {}))
        self.config["portfolio"]["starting_cash"] = 100_000.0
        self.config["portfolio"]["tickers_per_batch"] = 2

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @patch("tradingagents.portfolio.orchestrator.compute_portfolio_risk_metrics")
    def test_run_universe_collects_opportunities(self, mock_metrics) -> None:
        mock_metrics.return_value = DummyRiskMetrics()
        recorder: List[Dict[str, Any]] = []

        storage = PortfolioStorage(self.results_path)
        orchestrator = PortfolioOrchestrator(
            config=self.config,
            storage=storage,
            graph_factory=lambda: RecordingGraph(recorder),
        )

        snapshot = empty_portfolio(
            starting_cash=self.config["portfolio"]["starting_cash"],
            as_of=datetime(2024, 1, 3),
        )

        macro_snapshot = DummyMacroSnapshot(as_of=datetime(2024, 1, 5))

        run_result = orchestrator.run_universe(
            ["AAPL", "MSFT"],
            "2024-01-05",
            macro_snapshot=macro_snapshot,
            portfolio_snapshot=snapshot,
        )

        opportunities = run_result["trade_opportunities"]
        self.assertEqual(len(opportunities), 2)
        self.assertEqual({opp["symbol"] for opp in opportunities}, {"AAPL", "MSFT"})
        self.assertTrue(all(item["portfolio_context"] for item in recorder))
        self.assertTrue(mock_metrics.called)
        risk_payload = run_result["portfolio_manager_input"]["risk_metrics"]
        self.assertAlmostEqual(risk_payload["beta"], mock_metrics.return_value.beta)

    @patch("tradingagents.portfolio.orchestrator.compute_portfolio_risk_metrics")
    def test_apply_execution_updates_snapshot_and_storage(self, mock_metrics) -> None:
        mock_metrics.return_value = DummyRiskMetrics()

        storage = PortfolioStorage(self.results_path)
        orchestrator = PortfolioOrchestrator(config=self.config, storage=storage)
        storage.save_snapshot(
            empty_portfolio(
                starting_cash=self.config["portfolio"]["starting_cash"],
                as_of=datetime(2024, 1, 4),
            )
        )

        snapshot = orchestrator.apply_execution(
            {
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10,
                "price": 50.0,
                "timestamp": "2024-01-05T15:30:00",
            }
        )

        self.assertAlmostEqual(
            snapshot.cash,
            self.config["portfolio"]["starting_cash"] - 500.0,
            places=2,
        )
        self.assertIn("AAPL", snapshot.positions)

        transactions = storage.load_transaction_records()
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0].symbol, "AAPL")

        latest_feedback = storage.load_latest_feedback()
        self.assertIsInstance(latest_feedback, dict)
        performance_report = storage.load_performance_report()
        self.assertIsInstance(performance_report, dict)


if __name__ == "__main__":
    unittest.main()
