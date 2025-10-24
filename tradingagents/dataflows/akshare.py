"""AKShare vendor integration for A-share market data."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

import akshare as ak


EXCHANGE_PREFIX_MAP = {
    "SH": "sh",
    "SS": "sh",
    "SSE": "sh",
    "SHSE": "sh",
    "SHSZ": "sh",
    "XSHG": "sh",
    "CNSESH": "sh",
    "SZ": "sz",
    "SHE": "sz",
    "SZSE": "sz",
    "XSHE": "sz",
    "CNSESZ": "sz",
    "SZA": "sz",
    "BSE": "bj",
    "BJ": "bj",
    "BJS": "bj",
    "BJSE": "bj",
    "CNSEBJ": "bj",
}


def _split_symbol(symbol: str) -> tuple[str, str]:
    ticker = symbol.strip().upper()

    if "." in ticker:
        code, exchange = ticker.split(".", 1)
        return code, exchange

    if ticker.lower().startswith(("sh", "sz", "bj")) and len(ticker) >= 8:
        return ticker[2:], ticker[:2]

    return ticker, ""


def _normalize_symbol(symbol: str, *, with_prefix: bool = True) -> str:
    code, exchange = _split_symbol(symbol)
    core = code[:6]

    if not with_prefix:
        return core.lower()

    if exchange:
        mapped = EXCHANGE_PREFIX_MAP.get(exchange.upper())
        prefix = mapped or exchange.lower()
    elif core.startswith("6"):
        prefix = "sh"
    elif core.startswith(("0", "3")):
        prefix = "sz"
    elif core.startswith(("4", "8")):
        prefix = "bj"
    else:
        prefix = ""

    return f"{prefix}{core}".lower() if prefix else core.lower()


def _format_dataframe(data: Optional[object], title: str, symbol: str) -> str:
    if data is None:
        return f"No {title.lower()} available for symbol '{symbol}'"

    empty = getattr(data, "empty", None)
    if empty is True:
        return f"No {title.lower()} available for symbol '{symbol}'"

    try:
        csv_string = data.to_csv(index=False)
    except TypeError:
        csv_string = data.to_csv()

    header = f"# {title} for {symbol.upper()}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + csv_string


def get_stock_data(
    symbol: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    *,
    adjust: str | None = None,
) -> str:
    """Fetch OHLCV data for A-shares via AKShare."""
    normalized_symbol = _normalize_symbol(symbol)
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    try:
        data = ak.stock_zh_a_daily(
            symbol=normalized_symbol,
            start_date=start,
            end_date=end,
            adjust=adjust or "",
        )
    except Exception as exc:  # pragma: no cover - pass through vendor errors
        raise RuntimeError(f"AKShare get_stock_data failed: {exc}") from exc

    return _format_dataframe(
        data,
        title=f"AKShare daily data from {start_date} to {end_date}",
        symbol=symbol,
    )


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current trade date" ] | None = None,
) -> str:
    """Fetch fundamental snapshot using AKShare financial abstract."""
    normalized_symbol = _normalize_symbol(ticker, with_prefix=False)
    try:
        data = ak.stock_financial_abstract(symbol=normalized_symbol)
    except Exception as exc:  # pragma: no cover - pass through vendor errors
        raise RuntimeError(f"AKShare get_fundamentals failed: {exc}") from exc

    return _format_dataframe(data, title="AKShare financial abstract", symbol=ticker)


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency of data"] = "quarterly",
    curr_date: Annotated[str, "current trade date"] | None = None,
) -> str:
    normalized_symbol = _normalize_symbol(ticker, with_prefix=False)
    try:
        data = ak.stock_financial_report_sina(stock=normalized_symbol, symbol="资产负债表")
    except Exception as exc:  # pragma: no cover - pass through vendor errors
        raise RuntimeError(f"AKShare get_balance_sheet failed: {exc}") from exc

    return _format_dataframe(data, title="AKShare balance sheet", symbol=ticker)


def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency of data"] = "quarterly",
    curr_date: Annotated[str, "current trade date"] | None = None,
) -> str:
    normalized_symbol = _normalize_symbol(ticker, with_prefix=False)
    try:
        data = ak.stock_financial_report_sina(stock=normalized_symbol, symbol="现金流量表")
    except Exception as exc:  # pragma: no cover - pass through vendor errors
        raise RuntimeError(f"AKShare get_cashflow failed: {exc}") from exc

    return _format_dataframe(data, title="AKShare cashflow statement", symbol=ticker)


def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency of data"] = "quarterly",
    curr_date: Annotated[str, "current trade date"] | None = None,
) -> str:
    normalized_symbol = _normalize_symbol(ticker, with_prefix=False)
    try:
        data = ak.stock_financial_report_sina(stock=normalized_symbol, symbol="利润表")
    except Exception as exc:  # pragma: no cover - pass through vendor errors
        raise RuntimeError(f"AKShare get_income_statement failed: {exc}") from exc

    return _format_dataframe(data, title="AKShare income statement", symbol=ticker)
