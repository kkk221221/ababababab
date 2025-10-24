"""Persistence helpers for portfolio state and transaction history."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from .state import PortfolioSnapshot, TransactionRecord


class PortfolioStorage:
    """Handles loading and saving portfolio artefacts under the results directory."""

    SNAPSHOT_FILENAME = "portfolio_snapshot.json"
    TRANSACTIONS_FILENAME = "transactions.json"

    def __init__(self, results_dir: str, *, snapshot_filename: Optional[str] = None, transactions_filename: Optional[str] = None) -> None:
        self.results_dir = Path(results_dir)
        self.portfolio_dir = self.results_dir / "portfolio"
        self.portfolio_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_filename = snapshot_filename or self.SNAPSHOT_FILENAME
        self._transactions_filename = transactions_filename or self.TRANSACTIONS_FILENAME

    def snapshot_path(self) -> Path:
        return self.portfolio_dir / self._snapshot_filename

    def transactions_path(self) -> Path:
        return self.portfolio_dir / self._transactions_filename

    def save_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        payload = snapshot.to_dict()
        with self.snapshot_path().open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_snapshot(self) -> Optional[PortfolioSnapshot]:
        path = self.snapshot_path()
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        snapshot = PortfolioSnapshot.from_dict(payload)
        if snapshot.as_of.tzinfo is not None:
            snapshot.as_of = snapshot.as_of.replace(tzinfo=None)
        return snapshot

    def ensure_snapshot(self, starting_cash: float) -> PortfolioSnapshot:
        snapshot = self.load_snapshot()
        if snapshot is None:
            from .state import empty_portfolio

            snapshot = empty_portfolio(starting_cash)
            self.save_snapshot(snapshot)
        return snapshot

    def append_transactions(self, transactions: Iterable[TransactionRecord]) -> None:
        existing = self.load_transactions()
        serialized = [transaction.to_dict() for transaction in transactions]
        if not serialized:
            return
        existing.extend(serialized)
        with self.transactions_path().open("w", encoding="utf-8") as handle:
            json.dump(existing, handle, indent=2)

    def record_transaction(self, transaction: TransactionRecord) -> None:
        self.append_transactions([transaction])

    def load_transactions(self) -> List[dict]:
        path = self.transactions_path()
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def load_transaction_records(self) -> List[TransactionRecord]:
        records: List[TransactionRecord] = []
        for item in self.load_transactions():
            try:
                item["timestamp"] = item.get("timestamp") or datetime.utcnow().isoformat()
                records.append(TransactionRecord.from_dict(item))
            except (KeyError, ValueError):
                continue
        return records
