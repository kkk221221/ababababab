"""Macro data aggregation helpers feeding the portfolio manager."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .alpha_vantage_common import (
    AlphaVantageRateLimitError,
    _make_api_request,
)
from .interface import route_to_vendor
from tradingagents.default_config import DEFAULT_CONFIG


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_csv_series(csv_blob: str) -> List[Dict[str, Any]]:
    """Convert a CSV string returned by data vendors into dictionaries."""

    if not csv_blob:
        return []

    lines = [line for line in csv_blob.splitlines() if line and not line.startswith("#")]
    if len(lines) < 2:
        return []
    reader = csv.DictReader(StringIO("\n".join(lines)))
    rows: List[Dict[str, Any]] = []
    for row in reader:
        rows.append(row)
    return rows


def _series_change_pct(rows: Sequence[Mapping[str, Any]], value_key: str = "Close") -> Optional[float]:
    if len(rows) < 2:
        return None
    latest = _coerce_float(rows[-1].get(value_key))
    previous = _coerce_float(rows[-2].get(value_key))
    if latest is None or previous in (None, 0.0):
        return None
    return ((latest - previous) / previous) * 100.0


def _extract_latest_alpha_series(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the latest data point from an Alpha Vantage macro response."""

    candidates: Iterable[Any] = ()
    if isinstance(payload, Mapping):
        if isinstance(payload.get("data"), list):
            candidates = payload["data"]  # type: ignore[assignment]
        elif isinstance(payload.get("Data"), list):
            candidates = payload["Data"]  # type: ignore[assignment]
        else:
            # Sometimes the payload nests series under other keys
            for value in payload.values():
                if isinstance(value, list):
                    candidates = value
                    break
                if isinstance(value, Mapping) and isinstance(value.get("data"), list):
                    candidates = value["data"]  # type: ignore[assignment]
                    break
    latest: Optional[Dict[str, Any]] = None
    for item in candidates:
        if isinstance(item, Mapping):
            if latest is None:
                latest = dict(item)
            else:
                # Series are usually chronological; keep the one with the most recent date
                current_time = item.get("date") or item.get("timestamp") or item.get("time")
                latest_time = latest.get("date") or latest.get("timestamp") or latest.get("time")
                if isinstance(current_time, str) and isinstance(latest_time, str):
                    if current_time > latest_time:
                        latest = dict(item)
    return latest


def _fetch_alpha_macro_point(function_name: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    params_dict = dict(params or {})
    params_dict.setdefault("datatype", "json")
    try:
        response_text = _make_api_request(function_name, params_dict)
    except (ValueError, AlphaVantageRateLimitError) as exc:
        return {"error": str(exc)}
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        return {"error": f"Failed to parse {function_name} payload: {exc}"}
    latest = _extract_latest_alpha_series(payload)
    if not latest:
        return {"error": f"No data available for {function_name}"}
    value = _coerce_float(latest.get("value") or latest.get("CPI"))
    timestamp = latest.get("date") or latest.get("timestamp") or latest.get("time")
    return {
        "function": function_name,
        "timestamp": timestamp,
        "value": value,
        "raw": latest,
    }


@dataclass
class MacroSnapshot:
    """Normalized macro-economic snapshot used in prompts."""

    as_of: datetime
    rates: List[Dict[str, Any]] = field(default_factory=list)
    inflation: List[Dict[str, Any]] = field(default_factory=list)
    sectors: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        lines = [f"Macro snapshot as of {self.as_of.isoformat()} (UTC)"]
        if self.rates:
            lines.append("\nInterest Rates:")
            for item in self.rates:
                change_pct = item.get("change_pct")
                change_str = f" ({change_pct:.2f}% change)" if isinstance(change_pct, float) else ""
                lines.append(f"- {item['label']}: {item.get('latest')}%{change_str}")
        if self.inflation:
            lines.append("\nInflation Indicators:")
            for item in self.inflation:
                value = item.get("value")
                if value is None:
                    continue
                lines.append(f"- {item['label']}: {value}")
        if self.sectors:
            lines.append("\nSector Performance (last close and day-over-day change):")
            for sector in self.sectors:
                change = sector.get("change_pct")
                change_str = f" ({change:+.2f}% vs prior close)" if isinstance(change, float) else ""
                lines.append(f"- {sector['label']}: {sector.get('latest')} {change_str}")
        if self.sentiment:
            lines.append("\nNews & Sentiment Highlights:")
            headlines = self.sentiment.get("headlines") or []
            for headline in headlines[:5]:
                lines.append(f"- {headline}")
            summary = self.sentiment.get("summary")
            if summary:
                lines.append(f"\nSummary: {summary}")
        if self.errors:
            lines.append("\nData Gaps / Errors:")
            for error in self.errors:
                lines.append(f"- {error}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "rates": self.rates,
            "inflation": self.inflation,
            "sectors": self.sectors,
            "sentiment": self.sentiment,
            "errors": self.errors,
        }


def gather_macro_snapshot(
    trade_date: str,
    *,
    lookback_days: int = 30,
    config: Optional[Mapping[str, Any]] = None,
) -> MacroSnapshot:
    """Collect macro data relevant for the portfolio manager prompts."""

    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    macro_config = cfg.get("macro", {})

    as_of_dt = datetime.strptime(trade_date, "%Y-%m-%d") if trade_date else datetime.utcnow()
    start_dt = as_of_dt - timedelta(days=lookback_days)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = as_of_dt.strftime("%Y-%m-%d")

    snapshot = MacroSnapshot(as_of=as_of_dt)

    # Interest rates via configured vendor
    for symbol in macro_config.get("rate_symbols", []):
        try:
            csv_blob = route_to_vendor("get_stock_data", symbol, start_date, end_date)
            rows = _parse_csv_series(csv_blob if isinstance(csv_blob, str) else "")
            latest_value = _coerce_float(rows[-1].get("Close")) if rows else None
            snapshot.rates.append(
                {
                    "symbol": symbol,
                    "label": symbol,
                    "latest": latest_value,
                    "change_pct": _series_change_pct(rows),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            snapshot.errors.append(f"Rate fetch failed for {symbol}: {exc}")

    # Sector ETFs
    for symbol in macro_config.get("sector_symbols", []):
        try:
            csv_blob = route_to_vendor("get_stock_data", symbol, start_date, end_date)
            rows = _parse_csv_series(csv_blob if isinstance(csv_blob, str) else "")
            latest_close = _coerce_float(rows[-1].get("Close")) if rows else None
            snapshot.sectors.append(
                {
                    "symbol": symbol,
                    "label": symbol,
                    "latest": latest_close,
                    "change_pct": _series_change_pct(rows),
                }
            )
        except Exception as exc:  # pragma: no cover
            snapshot.errors.append(f"Sector fetch failed for {symbol}: {exc}")

    # Inflation metrics via Alpha Vantage if available
    for series in macro_config.get("inflation_series", []):
        data_point = _fetch_alpha_macro_point(series)
        if "error" in data_point:
            snapshot.errors.append(f"{series}: {data_point['error']}")
        else:
            snapshot.inflation.append(
                {
                    "label": series,
                    "timestamp": data_point.get("timestamp"),
                    "value": data_point.get("value"),
                }
            )

    # News sentiment via configured vendor(s)
    sentiment_query = macro_config.get("sentiment_query", "global markets")
    sentiment_lookback = int(macro_config.get("sentiment_lookback_days", 3))
    headlines: List[str] = []
    sentiment_payloads: List[Any] = []
    try:
        global_blob = route_to_vendor("get_global_news", end_date, sentiment_lookback, 5)
        sentiment_payloads.append(global_blob)
        if isinstance(global_blob, str):
            for line in global_blob.splitlines():
                if line.startswith("### "):
                    headlines.append(line.replace("### ", "").strip())
    except Exception as exc:  # pragma: no cover
        snapshot.errors.append(f"Global news fetch failed: {exc}")

    try:
        targeted_blob = route_to_vendor("get_news", sentiment_query, start_date, end_date)
        sentiment_payloads.append(targeted_blob)
        if isinstance(targeted_blob, str):
            for line in targeted_blob.splitlines():
                if line.startswith("### "):
                    headlines.append(line.replace("### ", "").strip())
    except Exception as exc:  # pragma: no cover
        snapshot.errors.append(f"Targeted news fetch failed for '{sentiment_query}': {exc}")

    snapshot.sentiment = {
        "query": sentiment_query,
        "lookback_days": sentiment_lookback,
        "headlines": headlines,
        "raw": sentiment_payloads,
    }

    return snapshot


__all__ = ["MacroSnapshot", "gather_macro_snapshot"]

