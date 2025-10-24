import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # Portfolio configuration
    "portfolio": {
        "starting_cash": float(os.getenv("TRADINGAGENTS_STARTING_CASH", 1_000_000.0)),
        "tickers_per_batch": int(os.getenv("TRADINGAGENTS_TICKERS_PER_BATCH", 5)),
        "default_slippage_bps": 5.0,
        "commission_per_share": 0.0,
        "max_single_position_pct": 0.1,
        "max_gross_exposure_pct": 2.0,
        "risk_per_trade_pct": 0.01,
        "atr_position_multiple": 1.5,
        "min_trade_notional": 1_000.0,
        "correlation_lookback_days": 60,
        "target_cash_buffer_pct": 0.05,
        "rebalance_tolerance_pct": 0.02,
        "feedback_history_limit": 50,
        "risk_benchmark_symbol": "SPY",
        "risk_lookback_days": 180,
        "var_confidence": 0.95,
        "risk_free_rate": 0.02,
        "max_var_pct": 0.05,
        "max_portfolio_beta": 1.5,
        "min_portfolio_sharpe": -0.5,
        "max_sector_exposure_pct": 0.35,
        "snapshot_filename": "portfolio_snapshot.json",
        "transactions_filename": "transactions.json",
        "feedback_filename": "portfolio_feedback.json",
        "lessons_filename": "portfolio_lessons.json",
        "nav_history_filename": "nav_history.json",
        "performance_filename": "portfolio_performance.json",
    },
    # Macro data ingestion defaults for the portfolio manager
    "macro": {
        # Treasury and funding rate tickers sourced via the configured market data vendor
        "rate_symbols": ["^TNX", "^IRX"],
        # Sector ETF tickers used to infer relative performance trends
        "sector_symbols": ["XLY", "XLF", "XLK", "XLI", "XLE", "XLV"],
        # Alpha Vantage macroeconomic series identifiers (if API access is configured)
        "inflation_series": ["CPI"],
        # Query terms leveraged by global news sentiment fetchers
        "sentiment_query": "global market risk appetite",
        # Lookback horizon (in days) for macro news aggregation
        "sentiment_lookback_days": 3,
    },
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "qwen3-max",
    "quick_think_llm": "qwen3-max",
    "backend_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        # Prefer mainland China data providers by default
        "core_stock_apis": "akshare",        # Options: yfinance, alpha_vantage, akshare, baostock, local
        "technical_indicators": "akshare",   # Options: yfinance, alpha_vantage, local, akshare
        "fundamental_data": "baostock",      # Options: openai, alpha_vantage, akshare, baostock, local
        "news_data": "google",               # Options: openai, alpha_vantage, google, local
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
        # Example: "get_news": "openai",               # Override category default
    },
}
