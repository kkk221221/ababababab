"""BaoStock vendor integration for A-share market data."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from typing import Annotated, Iterable, List, Sequence
import importlib
import importlib.util
import io
import csv
import sys


@lru_cache(maxsize=1)
def _load_baostock():
    """Return the BaoStock module or raise an informative error."""

    module_name = "baostock"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ModuleNotFoundError(
            "The 'baostock' package is required for BaoStock data vendors. "
            "Install it with `pip install baostock`."
        )

    return importlib.import_module(module_name)


@contextmanager
def _baostock_session():
    module = _load_baostock()

    login_result = module.login()
    if login_result.error_code != "0":
        raise RuntimeError(
            f"BaoStock login failed: {login_result.error_code} {login_result.error_msg}"
        )
    try:
        yield
    finally:
        module.logout()


def _normalize_symbol(symbol: str) -> str:
    ticker = symbol.strip().upper()
    if ticker.startswith(("SH.", "SZ.", "BJ.")):
        return ticker.lower()
    if ticker.endswith((".SH", ".SS")) and len(ticker) >= 9:
        return f"sh.{ticker[:6]}"
    if ticker.endswith(".SZ") and len(ticker) >= 9:
        return f"sz.{ticker[:6]}"
    if ticker.endswith(".BJ") and len(ticker) >= 9:
        return f"bj.{ticker[:6]}"
    if ticker.lower().startswith(("sh", "sz", "bj")) and len(ticker) == 8:
        return f"{ticker[:2].lower()}.{ticker[2:]}"
    if len(ticker) == 6:
        if ticker.startswith("6"):
            return f"sh.{ticker}"
        if ticker.startswith(("0", "3")):
            return f"sz.{ticker}"
        if ticker.startswith(("4", "8")):
            return f"bj.{ticker}"
    return ticker.lower()


def _format_csv(fields: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(fields)
    writer.writerows(rows)
    return buffer.getvalue()


def _format_table(
    fields: Sequence[str],
    rows: List[Sequence[str]],
    *,
    title: str,
    symbol: str,
) -> str:
    if not rows:
        return f"No {title.lower()} available for symbol '{symbol}'"

    csv_content = _format_csv(fields, rows)
    header = f"# {title} for {symbol.upper()}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + csv_content


def _collect_query_rows(query) -> List[Sequence[str]]:
    rows: List[Sequence[str]] = []
    while query.error_code == "0" and query.next():
        rows.append(tuple(query.get_row_data()))
    if query.error_code != "0":
        raise RuntimeError(f"BaoStock query failed: {query.error_code} {query.error_msg}")
    return rows


def get_stock_data(
    symbol: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    *,
    frequency: str = "d",
    adjust: str = "3",
) -> str:
    code = _normalize_symbol(symbol)
    module = _load_baostock()

    with _baostock_session():
        query = module.query_history_k_data_plus(
            code,
            fields="date,open,high,low,close,volume,amount,turn,tradeStatus",
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjust,
        )
        rows = _collect_query_rows(query)
    return _format_table(
        query.fields,
        rows,
        title=f"BaoStock {frequency} price data from {start_date} to {end_date}",
        symbol=symbol,
    )


def _determine_period(curr_date: str | None) -> tuple[int, int]:
    if curr_date:
        dt = datetime.strptime(curr_date, "%Y-%m-%d")
    else:
        dt = datetime.utcnow()
    quarter = ((dt.month - 1) // 3) + 1
    return dt.year, quarter


def _query_statement(
    code: str,
    curr_date: str | None,
    *,
    title: str,
    symbol: str,
    query_name: str,
) -> str:
    year, quarter = _determine_period(curr_date)
    module = _load_baostock()

    with _baostock_session():
        query = getattr(module, query_name)(code=code, year=year, quarter=quarter)
    rows = _collect_query_rows(query)
    return _format_table(query.fields, rows, title=title, symbol=symbol)


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current trade date"] | None = None,
) -> str:
    code = _normalize_symbol(ticker)
    return _query_statement(
        code,
        curr_date,
        title="BaoStock growth indicators",
        symbol=ticker,
        query_name="query_growth_data",
    )


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency of data"] = "quarterly",
    curr_date: Annotated[str, "current trade date"] | None = None,
) -> str:
    code = _normalize_symbol(ticker)
    return _query_statement(
        code,
        curr_date,
        title="BaoStock balance sheet",
        symbol=ticker,
        query_name="query_balance_data",
    )


def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency of data"] = "quarterly",
    curr_date: Annotated[str, "current trade date"] | None = None,
) -> str:
    code = _normalize_symbol(ticker)
    return _query_statement(
        code,
        curr_date,
        title="BaoStock cashflow statement",
        symbol=ticker,
        query_name="query_cash_flow_data",
    )


def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency of data"] = "quarterly",
    curr_date: Annotated[str, "current trade date"] | None = None,
) -> str:
    code = _normalize_symbol(ticker)
    return _query_statement(
        code,
        curr_date,
        title="BaoStock income statement",
        symbol=ticker,
        query_name="query_profit_data",
    )
