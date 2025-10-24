"""Portfolio-level risk metrics utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, Mapping, Optional

from tradingagents.portfolio.state import PortfolioSnapshot


@dataclass(slots=True)
class PortfolioRiskMetrics:
    """Container for commonly used portfolio-level risk statistics."""

    as_of: datetime
    lookback_days: int
    benchmark_symbol: str
    beta: Optional[float]
    value_at_risk_pct: Optional[float]
    var_confidence: float
    sharpe_ratio: Optional[float]
    risk_free_rate: float
    sector_exposure: Dict[str, float]
    symbol_sectors: Dict[str, str]
    invested_market_value: float

    def to_dict(self) -> Dict[str, object]:
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


def _sanitize_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        if value != value:  # NaN check
            return None
        if value in (float("inf"), float("-inf")):
            return None
        return float(round(value, 6))
    except TypeError:
        return None


def _fetch_price_history(
    symbols: Iterable[str],
    end_date: datetime,
    *,
    lookback_days: int,
) -> Optional["pd.DataFrame"]:
    try:
        import pandas as pd
        import yfinance as yf
    except ImportError:
        return None

    unique_symbols = sorted({symbol.upper() for symbol in symbols if symbol})
    if not unique_symbols:
        return None

    start_date = end_date - timedelta(days=max(lookback_days, 1) * 2)
    frames = []
    for symbol in unique_symbols:
        try:
            history = yf.Ticker(symbol).history(
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
        except Exception:
            continue
        if history.empty or "Close" not in history:
            continue
        series = history["Close"].rename(symbol).dropna()
        if not series.empty:
            frames.append(series)

    if not frames:
        return None

    data = frames[0].to_frame()
    for series in frames[1:]:
        data = data.join(series, how="outer")

    return data.sort_index().dropna(how="any")


def _lookup_sector(symbol: str) -> str:
    try:
        import yfinance as yf
    except ImportError:
        return "Unclassified"

    try:
        info = yf.Ticker(symbol).get_info()
    except Exception:
        info = None
    if isinstance(info, Mapping):
        sector = info.get("sector") or info.get("industry")
        if isinstance(sector, str) and sector.strip():
            return sector.strip()
    return "Unclassified"


def _compute_sector_exposure(snapshot: PortfolioSnapshot) -> tuple[Dict[str, float], Dict[str, str], float]:
    totals: Dict[str, float] = defaultdict(float)
    symbol_sectors: Dict[str, str] = {}
    invested_value = 0.0

    for position in snapshot.iter_positions():
        market_value = float(position.market_value or 0.0)
        if market_value == 0.0:
            continue
        sector = _lookup_sector(position.symbol)
        totals[sector] += market_value
        symbol_sectors[position.symbol] = sector
        invested_value += market_value

    sector_exposure: Dict[str, float] = {}
    if invested_value > 0:
        for sector, value in totals.items():
            sector_exposure[sector] = round(value / invested_value, 6)

    return sector_exposure, symbol_sectors, invested_value


def compute_portfolio_risk_metrics(
    snapshot: PortfolioSnapshot,
    *,
    trade_date: str,
    benchmark_symbol: str,
    lookback_days: int,
    confidence: float,
    risk_free_rate: float,
) -> PortfolioRiskMetrics:
    """Calculate portfolio-level risk metrics using recent price history."""

    try:
        end_date = datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        end_date = snapshot.as_of

    sector_exposure, symbol_sectors, invested_value = _compute_sector_exposure(snapshot)

    positions = [pos for pos in snapshot.iter_positions() if float(pos.market_value or 0.0) > 0]
    symbols = [pos.symbol for pos in positions]

    beta: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    value_at_risk_pct: Optional[float] = None

    if symbols:
        price_history = _fetch_price_history([*symbols, benchmark_symbol], end_date, lookback_days=lookback_days)
    else:
        price_history = None

    if price_history is not None and not price_history.empty:
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            price_history = None

    if price_history is not None and not price_history.empty:
        returns = price_history.pct_change().dropna(how="any")
        if not returns.empty:
            benchmark_series = returns.get(benchmark_symbol)
            asset_returns = returns.drop(columns=[benchmark_symbol], errors="ignore")
            if benchmark_series is not None and not benchmark_series.empty and not asset_returns.empty:
                total_market_value = sum(float(pos.market_value or 0.0) for pos in positions)
                weights = []
                for pos in positions:
                    weight = 0.0
                    if total_market_value > 0:
                        weight = float(pos.market_value or 0.0) / total_market_value
                    weights.append(weight)
                weights_series = pd.Series(weights, index=asset_returns.columns)
                portfolio_returns = (asset_returns * weights_series).sum(axis=1)
                covariance = float(portfolio_returns.cov(benchmark_series)) if len(portfolio_returns) > 1 else None
                variance = float(benchmark_series.var()) if len(benchmark_series) > 1 else None
                if covariance is not None and variance and variance != 0:
                    beta = covariance / variance

                excess_returns = portfolio_returns - (risk_free_rate / 252.0)
                std_dev = float(portfolio_returns.std())
                if std_dev and std_dev != 0:
                    sharpe_ratio = (float(excess_returns.mean()) / std_dev) * (252 ** 0.5)

                if 0.0 < confidence < 1.0:
                    quantile = float(np.quantile(portfolio_returns, 1 - confidence))
                    value_at_risk_pct = -quantile

    metrics = PortfolioRiskMetrics(
        as_of=end_date,
        lookback_days=int(lookback_days),
        benchmark_symbol=benchmark_symbol,
        beta=_sanitize_float(beta),
        value_at_risk_pct=_sanitize_float(value_at_risk_pct),
        var_confidence=confidence,
        sharpe_ratio=_sanitize_float(sharpe_ratio),
        risk_free_rate=risk_free_rate,
        sector_exposure=sector_exposure,
        symbol_sectors=symbol_sectors,
        invested_market_value=float(round(invested_value, 2)),
    )
    return metrics


__all__ = ["PortfolioRiskMetrics", "compute_portfolio_risk_metrics"]
