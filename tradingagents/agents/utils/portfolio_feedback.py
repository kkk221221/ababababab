"""Helpers to expose portfolio feedback summaries to agent prompts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional


def _coerce_lines(payload: Any) -> Iterable[str]:
    if isinstance(payload, str):
        text = payload.strip()
        if text:
            yield text
    elif isinstance(payload, Iterable):
        for item in payload:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    yield text


def format_portfolio_feedback(
    feedback: Optional[Mapping[str, Any]],
    audience: str,
) -> str:
    """Return a newline separated summary tailored to the given audience."""

    if not isinstance(feedback, Mapping):
        return ""

    lines = []
    summary = feedback.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.append(summary.strip())

    alerts = feedback.get("alerts")
    lines.extend(f"Alert: {alert.strip()}" for alert in _coerce_lines(alerts))

    audience_map = feedback.get("audience")
    if isinstance(audience_map, Mapping):
        lines.extend(_coerce_lines(audience_map.get(audience)))

    return "\n".join(dict.fromkeys(lines))


__all__ = ["format_portfolio_feedback"]
