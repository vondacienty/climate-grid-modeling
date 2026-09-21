"""Multi-year trend summary of detected extreme events."""

from __future__ import annotations

import datetime
import math
import re
from typing import Any

_SCHEMA = "climate-grid/trends-v1"
_EXTREMES_SCHEMA = "climate-grid/extremes-v1"
_EXTREMES_KEYS = frozenset({"schema", "events"})
_EVENT_KEYS = frozenset(
    {"start", "end", "days", "cells", "peak", "mean", "uncertainty"}
)
_PERIOD_KEYS = frozenset({"year", "data"})
_METRICS = ("count", "days", "cells", "intensity", "uncertainty")
_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _round_output(value: float) -> float:
    value = round(value, 12)
    if value == 0.0:
        # Normalize negative zero.
        return 0.0
    return value


def _parse_date(value: Any, where: str) -> datetime.date:
    if not isinstance(value, str):
        raise TypeError(f"{where} must be a str")
    if _DATE_RE.fullmatch(value) is None:
        raise ValueError(f"{where} must be a YYYY-MM-DD date")
    year, month, day = (int(part) for part in value.split("-"))
    try:
        return datetime.date(year, month, day)
    except ValueError:
        raise ValueError(f"{where} must be a valid Gregorian date") from None


def _validate_number(cell: Any, target: str) -> None:
    if not isinstance(cell, (int, float)) or isinstance(cell, bool):
        raise TypeError(f"{target} must be a number")
    if not math.isfinite(cell):
        raise ValueError(f"{target} must be finite")


def _validate_event(event: Any, where: str) -> None:
    if not isinstance(event, dict):
        raise TypeError(f"{where} must be a dict")
    if set(event.keys()) != _EVENT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys start, end, days, cells, "
            "peak, mean, uncertainty"
        )

    start = _parse_date(event["start"], f"{where}.start")
    end = _parse_date(event["end"], f"{where}.end")
    if end < start:
        raise ValueError(f"{where}.end must not precede {where}.start")

    days = event["days"]
    if not isinstance(days, int) or isinstance(days, bool):
        raise TypeError(f"{where}.days must be a non-bool int")
    if days < 1:
        raise ValueError(f"{where}.days must be positive")

    cells = event["cells"]
    if not isinstance(cells, list):
        raise TypeError(f"{where}.cells must be a list")
    if len(cells) == 0:
        raise ValueError(f"{where}.cells must be non-empty")
    for index, cell in enumerate(cells):
        target = f"{where}.cells[{index}]"
        if not isinstance(cell, list):
            raise TypeError(f"{target} must be a list")
        if len(cell) != 2:
            raise ValueError(f"{target} must be a [i, j] pair")
        for coordinate in cell:
            if not isinstance(coordinate, int) or isinstance(coordinate, bool):
                raise TypeError(f"{target} entries must be non-bool ints")
            if coordinate < 0:
                raise ValueError(f"{target} entries must be non-negative")

    for member in ("peak", "mean", "uncertainty"):
        _validate_number(event[member], f"{where}.{member}")


def _validate_extremes(result: Any, where: str) -> list:
    if not isinstance(result, dict):
        raise TypeError(f"{where} must be a dict")
    if set(result.keys()) != _EXTREMES_KEYS:
        raise ValueError(f"{where} must have exactly the keys schema, events")
    if not isinstance(result["schema"], str):
        raise TypeError(f"{where}.schema must be a str")
    if result["schema"] != _EXTREMES_SCHEMA:
        raise ValueError(f"{where}.schema must be {_EXTREMES_SCHEMA!r}")

    events = result["events"]
    if not isinstance(events, list):
        raise TypeError(f"{where}.events must be a list")
    for index, event in enumerate(events):
        _validate_event(event, f"{where}.events[{index}]")

    return events


def _validate_data(data: Any, where: str) -> tuple[list[str], dict]:
    if not isinstance(data, dict):
        raise TypeError(f"{where} must be a dict")
    if len(data) == 0:
        raise ValueError(f"{where} must be non-empty")

    order: list[str] = []
    events_by_element: dict[str, list] = {}
    for element, result in data.items():
        if not isinstance(element, str):
            raise TypeError(f"{where} element names must be str")
        if element == "":
            raise ValueError(f"{where} element names must be non-empty")
        order.append(element)
        events_by_element[element] = _validate_extremes(
            result, f"{where}[{element!r}]"
        )

    return order, events_by_element


def _slope(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    n = len(points)
    x_mean = sum(x for x, _ in points) / n
    y_mean = sum(y for _, y in points) / n
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    return _round_output(numerator / denominator)


def summarize(periods) -> dict:
    """Summarize per-year extreme-event metrics and their OLS trends.

    ``periods`` must be a non-empty list of dicts with exactly the keys
    ``year, data``: ``year`` is a strictly increasing non-bool int and
    ``data`` a non-empty dict mapping non-empty str element names to complete
    :func:`climate_grid.extremes.detect` results (schema
    ``climate-grid/extremes-v1``).  Every period must contain the same
    elements in the same order.

    For each year and element, ``count`` is the number of events and
    ``days``, ``cells``, ``intensity`` and ``uncertainty`` are the arithmetic
    means of the events' ``days``, ``len(cells)``, ``mean`` and
    ``uncertainty`` respectively; a year without events yields ``0`` and four
    ``None``.  Each metric is regressed by ordinary least squares against
    ``year`` in input order: ``count`` uses every year, the other metrics
    ignore ``None`` years, and a slope is ``None`` when fewer than two valid
    points remain.

    The returned mapping uses the key order ``schema, years, data``;
    ``schema`` is ``climate-grid/trends-v1`` and ``years`` lists the input
    years in input order.  ``data`` follows the first period's element order
    and each item uses the key order ``count, days, cells, intensity,
    uncertainty, slopes``: the first five are per-year lists and ``slopes``
    is a dict keyed in the same five-key order.  Every mean and slope is
    ``round(x, 12)`` with negative zero normalized to ``0.0``; ``count``
    entries are ints.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    if not isinstance(periods, list):
        raise TypeError("periods must be a list")
    if len(periods) == 0:
        raise ValueError("periods must be non-empty")

    years: list[int] = []
    parsed: list[dict] = []
    reference_order: list[str] | None = None
    for index, period in enumerate(periods):
        where = f"periods[{index}]"
        if not isinstance(period, dict):
            raise TypeError(f"{where} must be a dict")
        if set(period.keys()) != _PERIOD_KEYS:
            raise ValueError(f"{where} must have exactly the keys year, data")

        year = period["year"]
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"{where}.year must be a non-bool int")
        if years and year <= years[-1]:
            raise ValueError("years must be strictly increasing")
        years.append(year)

        order, events_by_element = _validate_data(period["data"], f"{where}.data")
        if reference_order is None:
            reference_order = order
        elif order != reference_order:
            raise ValueError(
                "all periods must contain the same elements in the same order"
            )
        parsed.append(events_by_element)

    data: dict[str, dict] = {}
    for element in reference_order:
        per_year = [
            parsed[index][element] for index in range(len(years))
        ]
        count = [len(events) for events in per_year]
        days: list[float | None] = []
        cells: list[float | None] = []
        intensity: list[float | None] = []
        uncertainty: list[float | None] = []
        for events in per_year:
            if not events:
                days.append(None)
                cells.append(None)
                intensity.append(None)
                uncertainty.append(None)
                continue
            days.append(
                _round_output(
                    sum(event["days"] for event in events) / len(events)
                )
            )
            cells.append(
                _round_output(
                    sum(len(event["cells"]) for event in events) / len(events)
                )
            )
            intensity.append(
                _round_output(
                    sum(event["mean"] for event in events) / len(events)
                )
            )
            uncertainty.append(
                _round_output(
                    sum(event["uncertainty"] for event in events) / len(events)
                )
            )

        series = {
            "count": count,
            "days": days,
            "cells": cells,
            "intensity": intensity,
            "uncertainty": uncertainty,
        }
        slopes = {}
        for metric in _METRICS:
            points = [
                (year, value)
                for year, value in zip(years, series[metric])
                if value is not None
            ]
            slopes[metric] = _slope(points)

        data[element] = {
            "count": count,
            "days": days,
            "cells": cells,
            "intensity": intensity,
            "uncertainty": uncertainty,
            "slopes": slopes,
        }

    return {"schema": _SCHEMA, "years": list(years), "data": data}
