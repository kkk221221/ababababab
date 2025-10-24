import functools
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from tradingagents.agents.utils.portfolio_feedback import format_portfolio_feedback
from tradingagents.dataflows.y_finance import get_stock_stats_indicators_window


def _extract_latest_numeric(indicator_text: str) -> Optional[float]:
    for line in indicator_text.splitlines():
        if ":" not in line:
            continue
        _, value_part = line.split(":", 1)
        value_str = value_part.strip().split(" ")[0].replace(",", "")
        try:
            return float(value_str)
        except ValueError:
            continue
    return None


def _fetch_atr(symbol: str, trade_date: str, lookback_days: int = 14) -> Optional[float]:
    try:
        raw = get_stock_stats_indicators_window(symbol, "atr", trade_date, lookback_days)
    except Exception:
        return None
    return _extract_latest_numeric(raw)


def _fetch_latest_close(symbol: str, trade_date: str) -> Optional[float]:
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=10)

    history = yf.Ticker(symbol).history(
        start=start_dt.strftime("%Y-%m-%d"),
        end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if history.empty or "Close" not in history:
        return None
    closes = history["Close"].dropna()
    if closes.empty:
        return None
    return float(closes.iloc[-1])


def _build_sizing_guidance(
    symbol: str,
    trade_date: str,
    portfolio_context: Optional[Dict[str, Any]],
    latest_price: Optional[float],
    atr: Optional[float],
) -> Dict[str, Any]:
    portfolio_context = portfolio_context or {}
    budgets = portfolio_context.get("budgets", {})
    constraints = portfolio_context.get("constraints", {})
    exposure = portfolio_context.get("exposure", {})

    available_cash = float(budgets.get("available_cash", 0.0) or 0.0)
    remaining_allocation = budgets.get("remaining_allocation")
    if remaining_allocation is not None:
        remaining_allocation = max(0.0, float(remaining_allocation))
    max_buy_notional = available_cash
    if remaining_allocation is not None:
        max_buy_notional = min(max_buy_notional, remaining_allocation)
    min_trade_notional = float(constraints.get("min_trade_notional", 0.0) or 0.0)
    if max_buy_notional < min_trade_notional:
        max_buy_notional = 0.0

    existing_position = (portfolio_context.get("positions", {}) or {}).get(symbol)
    existing_quantity = 0.0
    if isinstance(existing_position, dict):
        existing_quantity = float(existing_position.get("quantity", 0.0) or 0.0)

    risk_budget = float(budgets.get("risk_budget", 0.0) or 0.0)
    atr_multiple = float(constraints.get("atr_position_multiple", 1.5) or 1.5)
    stop_distance = atr * atr_multiple if atr is not None else None

    suggested_buy_quantity = 0
    if latest_price and latest_price > 0 and max_buy_notional > 0:
        suggested_buy_quantity = int(max_buy_notional // latest_price)

    suggested_stop_price = None
    suggested_take_profit = None
    if latest_price and stop_distance:
        suggested_stop_price = round(latest_price - stop_distance, 2)
        suggested_take_profit = round(latest_price + (stop_distance * 2), 2)

    baseline_notional = (
        suggested_buy_quantity * latest_price if latest_price and suggested_buy_quantity else 0.0
    )

    return {
        "symbol": symbol,
        "trade_date": trade_date,
        "last_close": latest_price,
        "atr": atr,
        "risk_budget": risk_budget,
        "available_cash": available_cash,
        "max_buy_notional": max_buy_notional,
        "baseline_notional": baseline_notional,
        "max_buy_quantity": suggested_buy_quantity,
        "max_sell_quantity": existing_quantity,
        "stop_distance": stop_distance,
        "suggested_stop_price": suggested_stop_price,
        "suggested_take_profit": suggested_take_profit,
        "exposure": exposure,
        "constraints": constraints,
        "correlations": portfolio_context.get("correlations", {}),
    }


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        trade_date = state["trade_date"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        portfolio_context = state.get("portfolio_context")
        trader_feedback = format_portfolio_feedback(
            state.get("portfolio_feedback"), "trader"
        )

        curr_situation = (
            f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for rec in past_memories:
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        atr = _fetch_atr(company_name, trade_date)
        latest_price = _fetch_latest_close(company_name, trade_date)
        sizing_guidance = _build_sizing_guidance(
            company_name,
            trade_date,
            portfolio_context,
            latest_price,
            atr,
        )

        prompt_payload = {
            "company": company_name,
            "trade_date": trade_date,
            "investment_plan": investment_plan,
            "market_research": market_research_report,
            "sentiment": sentiment_report,
            "news": news_report,
            "fundamentals": fundamentals_report,
            "portfolio_context": portfolio_context or {},
            "sizing_guidance": sizing_guidance,
            "trader_memory": past_memory_str.strip(),
            "portfolio_feedback": trader_feedback,
        }

        system_prompt = (
            "You are a professional execution trader. Analyse the provided research, "
            "portfolio context, and sizing guidance to recommend a concrete trade. "
            "Respond with a single JSON object matching this schema: {"
            "\"symbol\": str, \"trade_date\": str, \"decision\": {\"side\": one of ['BUY','SELL','HOLD'], "
            "\"confidence\": float between 0 and 1}, \"sizing\": {\"target_quantity\": float, "
            "\"notional\": float, \"max_buy_quantity\": float, \"max_sell_quantity\": float, "
            "\"risk_budget\": float, \"volatility_adjusted_stop\": float or null}, "
            "\"entry\": {\"limit_price\": float or null, \"stop_loss\": float or null, "
            "\"take_profit\": float or null}, \"rationale\": {\"summary\": str, "
            "\"portfolio_alignment\": str, \"risk_notes\": str}}. Ensure every numeric field uses decimals, "
            "and omit additional commentary outside the JSON object."
        )
        if trader_feedback:
            system_prompt += (
                " Emphasize alignment with the following portfolio feedback: "
                + trader_feedback
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(prompt_payload, indent=2)},
        ]

        result = llm.invoke(messages)

        raw_content = result.content.strip() if isinstance(result.content, str) else ""
        structured_plan: Optional[Dict[str, Any]] = None
        if raw_content:
            try:
                structured_plan = json.loads(raw_content)
            except json.JSONDecodeError:
                structured_plan = None

        trader_plan_text = (
            json.dumps(structured_plan, indent=2) if structured_plan is not None else raw_content
        )

        return {
            "messages": [result],
            "trader_investment_plan": trader_plan_text,
            "trader_structured_plan": structured_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
