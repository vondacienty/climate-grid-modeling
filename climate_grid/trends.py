"""Yearly trend summaries of detected extreme events."""

from __future__ import annotations

import datetime
import math
import re
from typing import Any

_SCHEMA = "climate-grid/trends-v1"
_EXTREMES_SCHEMA = "climate-grid/extremes-v1"
_PERIOD_KEYS = frozenset({"year", "data"})
_EXTREMES_KEYS = frozenset({"schema", "events"})
_EVENT_KEYS = frozenset(
    {"start", "end", "days", "cells", "peak", "mean", "uncertainty"}
)
_METRIC_KEYS = ("count", "days", "cells", "intensity", "uncertainty")
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


def _validate_number(value: Any, target: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{target} must be a finite non-bool int or float")
    if not math.isfinite(value):
        raise ValueError(f"{target} must be finite")


def _validate_event(event: Any, where: str) -> None:
    if not isinstance(event, dict):
        raise TypeError(f"{where} must be a dict")
    if set(event.keys()) != _EVENT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys "
            "start, end, days, cells, peak, mean, uncertainty"
        )

    start = _parse_date(event["start"], f"{where}.start")
    end = _parse_date(event["end"], f"{where}.end")
    if end < start:
        raise ValueError(f"{where}.end must not be earlier than {where}.start")

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
    previous: tuple[int, int] | None = None
    seen: set[tuple[int, int]] = set()
    for index, cell in enumerate(cells):
        target = f"{where}.cells[{index}]"
        if not isinstance(cell, list) or len(cell) != 2:
            raise TypeError(f"{target} must be a two-item list [i, j]")
        i, j = cell
        if not isinstance(i, int) or isinstance(i, bool):
            raise TypeError(f"{target}[0] must be a non-bool int")
        if not isinstance(j, int) or isinstance(j, bool):
            raise TypeError(f"{target}[1] must be a non-bool int")
        pair = (i, j)
        if pair in seen:
            raise ValueError(f"{where}.cells must be deduplicated")
        if previous is not None and pair <= previous:
            raise ValueError(f"{where}.cells must be in ascending order")
        seen.add(pair)
        previous = pair

    _validate_number(event["peak"], f"{where}.peak")
    _validate_number(event["mean"], f"{where}.mean")
    _validate_number(event["uncertainty"], f"{where}.uncertainty")


def _validate_detection(detection: Any, where: str) -> None:
    if not isinstance(detection, dict):
        raise TypeError(f"{where} must be a dict")
    if set(detection.keys()) != _EXTREMES_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, events"
        )
    schema = detection["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _EXTREMES_SCHEMA:
        raise ValueError(f"{where}.schema must be {_EXTREMES_SCHEMA!r}")
    events = detection["events"]
    if not isinstance(events, list):
        raise TypeError(f"{where}.events must be a list")
    for index, event in enumerate(events):
        _validate_event(event, f"{where}.events[{index}]")


def _slope(years: list[int], values: list) -> float | None:
    points = [
        (year, value)
        for year, value in zip(years, values)
        if value is not None
    ]
    if len(points) < 2:
        return None
    n = len(points)
    mean_year = sum(year for year, _ in points) / n
    mean_value = sum(value for _, value in points) / n
    denominator = sum((year - mean_year) ** 2 for year, _ in points)
    numerator = sum(
        (year - mean_year) * (value - mean_value)
        for year, value in points
    )
    return _round_output(numerator / denominator)


def summarize(periods) -> dict:
    """Summarize per-element event metrics and their yearly linear trends.

    ``periods`` must be a non-empty list of dicts, each with exactly the
    keys ``year`` and ``data``.  ``year`` is a non-bool int and the years
    must be strictly increasing.  ``data`` is a non-empty dict mapping
    non-empty str element names to complete :func:`climate_grid.extremes.detect`
    results (schema ``climate-grid/extremes-v1``); every year must expose
    the same element keys in the same order.

    For each year and element five yearly metrics are produced: ``count``
    is the number of events, while ``days``, ``cells``, ``intensity`` and
    ``uncertainty`` are the arithmetic means of the events' ``days``,
    ``len(cells)``, ``mean`` and ``uncertainty`` values.  Years without
    events yield a count of ``0`` and ``None`` for the other four metrics.

    Each metric gets an ordinary least-squares slope with the year as x
    in input order.  The ``count`` slope uses every year; the other
    slopes ignore ``None`` points and become ``None`` when fewer than two
    valid points remain.

    The returned mapping uses the key order ``schema, years, data``;
    ``schema`` is ``climate-grid/trends-v1`` and ``years`` preserves the
    input years.  ``data`` follows the first year's element order and each
    element uses the key order ``count, days, cells, intensity,
    uncertainty, slopes``; the first five are per-year lists (``count``
    values are ints) and ``slopes`` is a dict keyed in that same five-key
    order.  Every mean and slope float is ``round(x, 12)`` with negative
    zero normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    if not isinstance(periods, list):
        raise TypeError("periods must be a list")
    if len(periods) == 0:
        raise ValueError("periods must be non-empty")

    years: list[int] = []
    element_order: list[str] | None = None
    yearly_data: list[dict] = []
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
            raise ValueError("periods years must be strictly increasing")

        data = period["data"]
        if not isinstance(data, dict):
            raise TypeError(f"{where}.data must be a dict")
        if len(data) == 0:
            raise ValueError(f"{where}.data must be non-empty")
        for element, detection in data.items():
            if not isinstance(element, str):
                raise TypeError(f"{where}.data element names must be str")
            if element == "":
                raise ValueError(f"{where}.data element names must be non-empty")
            _validate_detection(detection, f"{where}.data[{element!r}]")

        keys = list(data.keys())
        if element_order is None:
            element_order = keys
        elif keys != element_order:
            raise ValueError(
                f"{where}.data element keys and order must match the first year"
            )

        years.append(year)
        yearly_data.append(data)

    result_data: dict = {}
    for element in element_order:
        counts: list[int] = []
        days_series: list = []
        cells_series: list = []
        intensity_series: list = []
        uncertainty_series: list = []
        for data in yearly_data:
            events = data[element]["events"]
            count = len(events)
            counts.append(count)
            if count == 0:
                days_series.append(None)
                cells_series.append(None)
                intensity_series.append(None)
                uncertainty_series.append(None)
            else:
                days_series.append(
                    _round_output(sum(event["days"] for event in events) / count)
                )
                cells_series.append(
                    _round_output(
                        sum(len(event["cells"]) for event in events) / count
                    )
                )
                intensity_series.append(
                    _round_output(sum(event["mean"] for event in events) / count)
                )
                uncertainty_series.append(
                    _round_output(
                        sum(event["uncertainty"] for event in events) / count
                    )
                )

        series = {
            "count": counts,
            "days": days_series,
            "cells": cells_series,
            "intensity": intensity_series,
            "uncertainty": uncertainty_series,
        }
        slopes = {key: _slope(years, series[key]) for key in _METRIC_KEYS}
        series["slopes"] = slopes
        result_data[element] = series

    return {"schema": _SCHEMA, "years": years, "data": result_data}
