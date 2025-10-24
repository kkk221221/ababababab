"""Core portfolio state definitions used across TradingAgents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional


def _as_float(value: object, default: float = 0.0) -> float:
    """Best-effort conversion to float with a safe default."""

    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class Position:
    """Represents the current holdings for a single symbol."""

    symbol: str
    quantity: float = 0.0
    average_cost: float = 0.0
    market_price: Optional[float] = None
    market_value: float = 0.0
    unrealized_pnl: float = 0.0

    def update_mark_to_market(self, price: float) -> None:
        """Refresh derived valuation fields using the latest traded price."""

        self.market_price = price
        self.market_value = price * self.quantity
        self.unrealized_pnl = (price - self.average_cost) * self.quantity

    def to_dict(self) -> Dict[str, float | str | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Position":
        return cls(
            symbol=str(payload.get("symbol", "")),
            quantity=_as_float(payload.get("quantity")),
            average_cost=_as_float(payload.get("average_cost")),
            market_price=(
                _as_float(payload.get("market_price"))
                if payload.get("market_price") not in (None, "")
                else None
            ),
            market_value=_as_float(payload.get("market_value")),
            unrealized_pnl=_as_float(payload.get("unrealized_pnl")),
        )


@dataclass(slots=True)
class TransactionRecord:
    """Captures a single executed trade for audit and attribution."""

    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    timestamp: datetime
    fees: float = 0.0
    slippage_bps: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TransactionRecord":
        timestamp_raw = payload.get("timestamp")
        timestamp = (
            datetime.fromisoformat(str(timestamp_raw))
            if timestamp_raw not in (None, "")
            else datetime.utcnow()
        )
        return cls(
            symbol=str(payload.get("symbol", "")),
            side=str(payload.get("side", "")).upper(),
            quantity=_as_float(payload.get("quantity")),
            price=_as_float(payload.get("price")),
            timestamp=timestamp,
            fees=_as_float(payload.get("fees")),
            slippage_bps=_as_float(payload.get("slippage_bps")),
        )


@dataclass(slots=True)
class PortfolioSnapshot:
    """Represents the full portfolio state at a moment in time."""

    as_of: datetime
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    total_market_value: float = 0.0
    total_equity: float = 0.0

    def update_totals(self) -> None:
        """Recalculate aggregate market value and equity from positions."""

        self.total_market_value = sum(position.market_value for position in self.positions.values())
        self.total_equity = self.cash + self.total_market_value + self.realized_pnl

    def update_position(self, position: Position) -> None:
        self.positions[position.symbol] = position
        self.update_totals()

    def remove_position(self, symbol: str) -> None:
        if symbol in self.positions:
            del self.positions[symbol]
            self.update_totals()

    def apply_transaction(
        self,
        transaction: TransactionRecord,
        *,
        commission_per_share: float = 0.0,
    ) -> None:
        """Apply a filled transaction to the portfolio state."""

        side = transaction.side.upper()
        quantity = transaction.quantity
        if quantity <= 0 or side not in {"BUY", "SELL"}:
            return

        position = self.positions.get(transaction.symbol)
        if position is None:
            position = Position(symbol=transaction.symbol)

        price = transaction.price
        total_fees = (transaction.fees or 0.0) + commission_per_share * quantity

        if side == "BUY":
            total_cost = price * quantity + total_fees
            self.cash -= total_cost
            prev_cost = position.average_cost * position.quantity
            new_quantity = position.quantity + quantity
            if new_quantity <= 0:
                position.quantity = 0.0
                position.average_cost = 0.0
            else:
                position.quantity = new_quantity
                position.average_cost = (
                    prev_cost + price * quantity
                ) / new_quantity
            position.update_mark_to_market(price)
            self.update_position(position)
        else:  # SELL
            if position.quantity <= 0:
                return
            sell_quantity = min(quantity, position.quantity)
            total_proceeds = price * quantity - total_fees
            self.cash += total_proceeds
            realized = (price - position.average_cost) * sell_quantity
            self.realized_pnl += realized
            new_quantity = position.quantity - quantity
            if new_quantity <= 0:
                self.remove_position(transaction.symbol)
            else:
                position.quantity = new_quantity
                position.update_mark_to_market(price)
                self.update_position(position)

        self.as_of = transaction.timestamp
        self.update_totals()

    def to_dict(self) -> Dict[str, object]:
        return {
            "as_of": self.as_of.isoformat(),
            "cash": self.cash,
            "positions": {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            "realized_pnl": self.realized_pnl,
            "total_market_value": self.total_market_value,
            "total_equity": self.total_equity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PortfolioSnapshot":
        positions_payload = payload.get("positions", {})
        positions: Dict[str, Position] = {}
        if isinstance(positions_payload, Mapping):
            positions = {
                symbol: Position.from_dict(dict(position_dict))
                for symbol, position_dict in positions_payload.items()
                if isinstance(position_dict, Mapping)
            }
        as_of_raw = payload.get("as_of")
        as_of = (
            datetime.fromisoformat(str(as_of_raw))
            if as_of_raw not in (None, "")
            else datetime.utcnow()
        )
        snapshot = cls(
            as_of=as_of,
            cash=_as_float(payload.get("cash")),
            positions=positions,
            realized_pnl=_as_float(payload.get("realized_pnl")),
            total_market_value=_as_float(payload.get("total_market_value")),
            total_equity=_as_float(payload.get("total_equity")),
        )
        snapshot.update_totals()
        return snapshot

    def iter_positions(self) -> Iterable[Position]:
        return self.positions.values()

    def list_positions(self) -> List[Position]:
        return list(self.iter_positions())


def empty_portfolio(starting_cash: float, as_of: Optional[datetime] = None) -> PortfolioSnapshot:
    """Helper to bootstrap a pristine portfolio state."""

    snapshot = PortfolioSnapshot(as_of=as_of or datetime.utcnow(), cash=starting_cash)
    snapshot.update_totals()
    return snapshot
