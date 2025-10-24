"""Persistence helpers for portfolio state and transaction history."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Mapping

from .state import PortfolioSnapshot, TransactionRecord


class PortfolioStorage:
    """Handles loading and saving portfolio artefacts under the results directory."""

    SNAPSHOT_FILENAME = "portfolio_snapshot.json"
    TRANSACTIONS_FILENAME = "transactions.json"
    FEEDBACK_FILENAME = "portfolio_feedback.json"
    LESSONS_FILENAME = "portfolio_lessons.json"
    NAV_HISTORY_FILENAME = "nav_history.json"
    PERFORMANCE_FILENAME = "portfolio_performance.json"

    def __init__(
        self,
        results_dir: str,
        *,
        snapshot_filename: Optional[str] = None,
        transactions_filename: Optional[str] = None,
        feedback_filename: Optional[str] = None,
        lessons_filename: Optional[str] = None,
        nav_history_filename: Optional[str] = None,
        performance_filename: Optional[str] = None,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.portfolio_dir = self.results_dir / "portfolio"
        self.portfolio_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_filename = snapshot_filename or self.SNAPSHOT_FILENAME
        self._transactions_filename = transactions_filename or self.TRANSACTIONS_FILENAME
        self._feedback_filename = feedback_filename or self.FEEDBACK_FILENAME
        self._lessons_filename = lessons_filename or self.LESSONS_FILENAME
        self._nav_history_filename = nav_history_filename or self.NAV_HISTORY_FILENAME
        self._performance_filename = performance_filename or self.PERFORMANCE_FILENAME

    def snapshot_path(self) -> Path:
        return self.portfolio_dir / self._snapshot_filename

    def transactions_path(self) -> Path:
        return self.portfolio_dir / self._transactions_filename

    def feedback_path(self) -> Path:
        return self.portfolio_dir / self._feedback_filename

    def lessons_path(self) -> Path:
        return self.portfolio_dir / self._lessons_filename

    def nav_history_path(self) -> Path:
        return self.portfolio_dir / self._nav_history_filename

    def performance_path(self) -> Path:
        return self.portfolio_dir / self._performance_filename

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

    # Feedback and lessons -------------------------------------------------

    def load_feedback(self) -> List[dict]:
        path = self.feedback_path()
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def save_feedback(self, feedback_entries: List[dict]) -> None:
        with self.feedback_path().open("w", encoding="utf-8") as handle:
            json.dump(feedback_entries, handle, indent=2)

    def append_feedback(self, entry: dict, *, history_limit: Optional[int] = None) -> None:
        if not isinstance(entry, dict):
            return
        entries = self.load_feedback()
        entries.append(entry)
        if history_limit and history_limit > 0:
            entries = entries[-int(history_limit) :]
        self.save_feedback(entries)

    def load_latest_feedback(self) -> Optional[dict]:
        entries = self.load_feedback()
        if not entries:
            return None
        return entries[-1]

    def load_lessons(self) -> List[dict]:
        path = self.lessons_path()
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def save_lessons(self, lessons: List[dict]) -> None:
        with self.lessons_path().open("w", encoding="utf-8") as handle:
            json.dump(lessons, handle, indent=2)

    def append_lessons(self, lessons: Iterable[Mapping[str, object]]) -> None:
        serialized: List[dict] = []
        for lesson in lessons:
            if not isinstance(lesson, Mapping):
                continue
            situation = str(lesson.get("situation", "")).strip()
            recommendation = str(lesson.get("recommendation", "")).strip()
            if not situation or not recommendation:
                continue
            payload = dict(lesson)
            payload["situation"] = situation
            payload["recommendation"] = recommendation
            serialized.append(payload)
        if not serialized:
            return
        existing = self.load_lessons()
        existing.extend(serialized)
        self.save_lessons(existing)

    # Performance reporting ------------------------------------------------

    def load_nav_history(self) -> List[dict]:
        path = self.nav_history_path()
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def save_nav_history(self, history: List[dict]) -> None:
        with self.nav_history_path().open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    def load_performance_report(self) -> Optional[dict]:
        path = self.performance_path()
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return None
        return payload

    def save_performance_report(self, report: Mapping[str, object]) -> None:
        with self.performance_path().open("w", encoding="utf-8") as handle:
            json.dump(dict(report), handle, indent=2)
