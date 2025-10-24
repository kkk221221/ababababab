import time
import json

from tradingagents.agents.risk_mgmt.context_utils import (
    serialize_portfolio_context,
    summarize_portfolio_risk,
)
from tradingagents.agents.utils.portfolio_feedback import format_portfolio_feedback


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]
        portfolio_context = state.get("portfolio_context")
        structured_plan = state.get("trader_structured_plan")

        risk_summary = summarize_portfolio_risk(portfolio_context)
        risk_json = serialize_portfolio_context(portfolio_context)
        feedback_note = format_portfolio_feedback(
            state.get("portfolio_feedback"), "risk"
        )

        constraints = (portfolio_context or {}).get("constraints", {})
        exposure = (portfolio_context or {}).get("exposure", {})
        positions = (portfolio_context or {}).get("positions", {})
        risk_metrics = (portfolio_context or {}).get("risk_metrics", {})

        total_equity = float((portfolio_context or {}).get("total_equity", 0.0) or 0.0)
        existing_position = {}
        if isinstance(positions, dict):
            existing_position = positions.get(company_name, {}) or {}
        existing_value = float(existing_position.get("market_value", 0.0) or 0.0)

        violations = []
        warnings = []
        trade_side = ""
        trade_notional = 0.0

        if isinstance(structured_plan, dict):
            decision = structured_plan.get("decision", {}) or {}
            sizing = structured_plan.get("sizing", {}) or {}
            trade_side = str(decision.get("side", "")).upper()
            trade_notional = float(sizing.get("notional", 0.0) or 0.0)
            target_qty = float(sizing.get("target_quantity", 0.0) or 0.0)

            max_single_pct = constraints.get("max_single_position_pct")
            if (
                trade_side == "BUY"
                and trade_notional > 0
                and max_single_pct not in (None, "")
                and total_equity > 0
            ):
                limit_value = total_equity * float(max_single_pct)
                projected = existing_value + trade_notional
                if projected > limit_value:
                    violations.append(
                        f"Single-position cap exceeded: projected {projected:.2f} vs limit {limit_value:.2f}."
                    )

            gross_limit = constraints.get("max_gross_exposure_pct")
            gross_pct = float(exposure.get("gross_exposure_pct", 0.0) or 0.0)
            if (
                gross_limit not in (None, "")
                and total_equity > 0
                and trade_notional > 0
            ):
                delta = trade_notional / total_equity
                projected_gross = gross_pct + (delta if trade_side == "BUY" else -delta)
                gross_limit_value = float(gross_limit)
                if projected_gross > gross_limit_value:
                    violations.append(
                        f"Gross exposure would reach {projected_gross:.2f} (> limit {gross_limit_value})."
                    )

            max_sector_pct = constraints.get("max_sector_exposure_pct")
            sector_exposure = risk_metrics.get("sector_exposure", {}) or {}
            symbol_sectors = risk_metrics.get("symbol_sectors", {}) or {}
            invested_value = float(risk_metrics.get("invested_market_value", 0.0) or 0.0)
            trade_sector = symbol_sectors.get(company_name)
            if (
                trade_sector
                and max_sector_pct not in (None, "")
                and invested_value >= 0
                and trade_notional > 0
            ):
                current_sector_value = sector_exposure.get(trade_sector, 0.0) * invested_value
                if trade_side == "BUY":
                    projected_sector_value = current_sector_value + trade_notional
                    projected_invested = invested_value + trade_notional
                else:
                    projected_sector_value = max(0.0, current_sector_value - trade_notional)
                    projected_invested = max(0.0, invested_value - trade_notional)
                if projected_invested > 0:
                    projected_sector_pct = projected_sector_value / projected_invested
                    max_sector_value = float(max_sector_pct)
                    if projected_sector_pct > max_sector_value:
                        violations.append(
                            f"Sector exposure for {trade_sector} would reach {projected_sector_pct:.2%} (> limit {max_sector_value:.2%})."
                        )

            if trade_side == "SELL" and target_qty <= 0 and existing_value > 0:
                warnings.append("Sell decision lacks target quantity despite existing exposure.")

        max_var_pct = constraints.get("max_var_pct")
        portfolio_var = risk_metrics.get("value_at_risk_pct")
        if (
            max_var_pct not in (None, "")
            and portfolio_var is not None
            and portfolio_var > float(max_var_pct)
        ):
            warnings.append(
                f"Portfolio VaR {portfolio_var:.2%} already exceeds limit {float(max_var_pct):.2%}."
            )

        max_beta = constraints.get("max_portfolio_beta")
        beta = risk_metrics.get("beta")
        if max_beta not in (None, "") and beta is not None and beta > float(max_beta):
            warnings.append(
                f"Portfolio beta {beta:.2f} is above cap {float(max_beta):.2f}."
            )

        min_sharpe = constraints.get("min_portfolio_sharpe")
        sharpe = risk_metrics.get("sharpe_ratio")
        if (
            min_sharpe not in (None, "")
            and sharpe is not None
            and sharpe < float(min_sharpe)
        ):
            warnings.append(
                f"Sharpe ratio {sharpe:.2f} is below minimum {float(min_sharpe):.2f}."
            )

        violation_text = "\n".join(f"- {item}" for item in violations) or "None identified."
        warning_text = "\n".join(f"- {item}" for item in warnings) or "None identified."
        structured_plan_json = (
            json.dumps(structured_plan, indent=2) if isinstance(structured_plan, dict) else str(structured_plan or "None")
        )

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Risk Management Judge and Debate Facilitator, your goal is to evaluate the debate between three risk analysts—Risky, Neutral, and Safe/Conservative—and determine the best course of action for the trader. Your decision must result in a clear recommendation: Buy, Sell, or Hold. Choose Hold only if strongly justified by specific arguments, not as a fallback when all sides seem valid. Strive for clarity and decisiveness.

Guidelines for Decision-Making:
1. **Summarize Key Arguments**: Extract the strongest points from each analyst, focusing on relevance to the context.
2. **Provide Rationale**: Support your recommendation with direct quotes and counterarguments from the debate.
3. **Refine the Trader's Plan**: Start with the trader's original plan, **{trader_plan}**, and the structured trader output: {structured_plan_json}. Adjust it based on the analysts' insights.
4. **Cross-Check Quantitative Limits**: Use the portfolio risk snapshot and identified risk flags to ensure the final recommendation respects exposure, VaR, beta, and Sharpe constraints. Violations must be addressed directly—recommend hedges, position sizing adjustments, or rejection of the trade if needed.
5. **Learn from Past Mistakes**: Use lessons from **{past_memory_str}** to address prior misjudgments and improve the decision you are making now to make sure you don't make a wrong BUY/SELL/HOLD call that loses money.

Deliverables:
- A clear and actionable recommendation: Buy, Sell, or Hold.
- Detailed reasoning anchored in the debate, quantitative checks, and past reflections.
- If you approve the trade, explicitly explain how it satisfies each relevant risk limit and whether any mitigations are required.
- If violations remain unresolved, recommend alternative actions (hedging, scaling down, or rejecting the trade).

---

**Analysts Debate History:**
{history}

---

Portfolio risk snapshot:
{risk_summary}

Structured portfolio context:
{risk_json}

Portfolio feedback summary:
{feedback_note if feedback_note else "None"}

Identified risk limit breaches:
{violation_text}

Outstanding warnings:
{warning_text}

---

Focus on actionable insights and continuous improvement. Build on past lessons, critically evaluate all perspectives, and ensure each decision advances better outcomes."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
