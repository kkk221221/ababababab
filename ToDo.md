# ToDo: Portfolio Manager Expansion Plan

## Background Analysis
- The current trading workflow (`tradingagents/graph/trading_graph.py`) orchestrates a single-ticker pipeline that ends with a risk manager verdict and does not retain portfolio context between runs.
- Trader decisions (`tradingagents/agents/trader/trader.py`) output narrative BUY/SELL/HOLD proposals without quantity sizing or exposure checks.
- Risk debate agents (`tradingagents/agents/risk_mgmt/*.py`) focus on qualitative discussion and lack quantitative portfolio metrics.
- State definitions (`tradingagents/agents/utils/agent_states.py`) track reports and debate transcripts but have no fields for holdings, cash, or cross-asset risk.
- Available dataflow utilities (`tradingagents/dataflows/y_finance.py`, `tradingagents/dataflows/alpha_vantage_*.py`) already supply price history, fundamentals, and macro data that can support portfolio calculations.

## Step-by-Step Implementation Plan

### Phase 1 – Core Portfolio State & Persistence
- [x] Create `tradingagents/portfolio/state.py` defining dataclasses or TypedDicts for `PortfolioSnapshot`, `Position`, and `TransactionRecord` that capture cash, per-ticker quantities, cost basis, valuation, and realized/unrealized P&L.
- [x] Extend persistence by adding `tradingagents/portfolio/storage.py` to load/save portfolio snapshots and transaction ledgers under `DEFAULT_CONFIG['results_dir']`, mirroring the logging pattern already used in `_log_state` inside `TradingAgentsGraph`.
- [x] Update `tradingagents/default_config.py` with portfolio-related knobs (starting capital, slippage assumptions, target risk limits) so orchestrators and agents can reference consistent settings.

### Phase 2 – Portfolio Manager Agent & Macro Intake
- [x] Introduce `tradingagents/agents/managers/portfolio_manager.py` that accepts the new portfolio state, macro data summaries, and trade proposals, producing allocation targets and per-ticker risk budgets.
- [x] Build supporting dataflow helpers (e.g., `tradingagents/dataflows/macro.py`) that wrap existing Alpha Vantage and news utilities to fetch rates, CPI, sector indices, and sentiment aggregates for the Portfolio Manager’s macro prompt.
- [x] Define prompt templates and structured outputs (JSON schema) so the Portfolio Manager returns normalized objectives (e.g., target weights, confidence scores) consumable by downstream components.

### Phase 3 – Multi-Asset Orchestration Layer
- [x] Design a `PortfolioOrchestrator` class under `tradingagents/portfolio/orchestrator.py` that coordinates macro assessment,
     selects a ticker universe, and invokes per-ticker `TradingAgentsGraph` runs in parallel or sequence using allocated budgets.
- [x] Refactor `TradingAgentsGraph.propagate` to optionally accept a portfolio context object (current holdings, budget, concentration limits) so single-ticker analysis can tailor outputs without breaking existing CLI usage.
- [x] Implement an aggregation routine that collects each ticker’s `final_trade_decision` and converts it into structured trade opportunities forwarded to the Portfolio Manager for approval.

### Phase 4 – Trader Enhancements for Position Sizing
- [x] Modify `create_trader` (and associated state updates) to request portfolio context from the orchestrator, compute position sizes using volatility-adjusted sizing (e.g., ATR from `get_stock_stats_indicators_window`), and output JSON containing side, quantity, entry constraints, and rationale.
- [x] Update `AgentState` (or extend with a `PortfolioContext` field) so traders and risk agents access assigned cash budgets, current exposure, and instrument correlations while generating recommendations.
- [x] Persist executed trades via the new storage module, ensuring the Trader records fills and updates holdings after Portfolio Manager approval.

### Phase 5 – Risk Management Across the Portfolio
- [ ] Implement quantitative calculators in `tradingagents/portfolio/risk.py` that derive portfolio beta, sector exposure, VaR, and Sharpe ratio using price history sourced from `y_finance` utilities.
- [ ] Feed those metrics into the risk debate prompts (`aggresive_debator.py`, `conservative_debator.py`, `neutral_debator.py`) so each viewpoint references concrete numbers (e.g., "Current portfolio beta is 1.3") alongside qualitative reasoning.
- [ ] Expand the risk judge (`risk_manager.py`) to cross-check proposed trades against concentration limits, VaR thresholds, and stress scenarios, potentially requesting hedging suggestions when limits are breached.

### Phase 6 – Portfolio Manager Decision Loop & Feedback
- [ ] Implement a closed-loop update where executed trades trigger portfolio snapshot refreshes, rebalancing checks, and feedback messages distributed to analysts, researchers, and the Portfolio Manager to inform future decisions.
- [ ] Add hooks so `FinancialSituationMemory` can store notable portfolio-level lessons (e.g., leverage errors, concentration issues) for retrieval during future sessions across all agents.
- [ ] Create reports summarizing portfolio performance over time (daily NAV, drawdowns) saved alongside existing evaluation logs for transparency and backtesting.

### Phase 7 – Interface, Tests, and Documentation
- [ ] Extend the CLI (`cli/main.py`) to expose portfolio-wide configuration (initial capital, tickers per batch, allocation preferences) and display aggregated portfolio dashboards after each run.
- [ ] Write integration tests or simulations (`tests/` or extend `test.py`) that exercise the orchestrator with mocked data to verify position sizing, portfolio updates, and risk limit enforcement without hitting external APIs.
- [ ] Update `README.md` with a dedicated section describing the new portfolio workflow, configuration settings, and sample outputs so users understand the expanded capabilities.

## Validation Checklist
- [ ] Verify backward compatibility by allowing legacy single-ticker runs to bypass the Portfolio Manager pipeline when portfolio features are disabled.
- [ ] Ensure all new LLM prompts enforce structured JSON outputs to simplify downstream parsing and reduce hallucination risk.
- [ ] Confirm that persistence and logging respect the existing directory layout and avoid leaking sensitive API keys or credentials.
