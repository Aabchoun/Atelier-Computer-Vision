"""Run report: events (info / warning / alert / error), summary figures and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from firespread.common.message import json_default
from firespread.evaluation import results_table

LEVELS = ("info", "warning", "alert", "error")


@dataclass
class Event:
    level: str
    source: str
    message: str


@dataclass
class Report:
    scene: str
    forecast_time: datetime
    events: list[Event] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, level: str, source: str, message: str) -> None:
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}")
        self.events.append(Event(level, source, message))

    def by_level(self, level: str) -> list[Event]:
        return [e for e in self.events if e.level == level]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "forecast_time": self.forecast_time.isoformat(),
            "summary": self.summary,
            "evaluation": self.evaluation,
            "events": [asdict(e) for e in self.events],
        }

    def to_markdown(self) -> str:
        lines = [f"# Fire spread report: {self.scene}", "", f"Forecast reference time: {self.forecast_time:%Y-%m-%d %H:%M} UTC", ""]
        alerts = self.by_level("alert")
        lines += ["## Alerts", ""] + ([f"- {e.message}" for e in alerts] or ["- none"]) + [""]
        if self.summary:
            lines += ["## Summary", "", "| item | value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in self.summary.items()] + [""]
        if self.evaluation:
            lines += ["## Evaluation against ground truth", "", results_table(self.evaluation), ""]
        lines += ["## Event log", ""] + [f"- **{e.level.upper()}** [{e.source}] {e.message}" for e in self.events]
        return "\n".join(lines) + "\n"

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text(json.dumps(self.to_dict(), indent=2, default=json_default), encoding="utf-8")
        (directory / "report.md").write_text(self.to_markdown(), encoding="utf-8")
