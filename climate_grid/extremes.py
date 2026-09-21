"""Detection of extreme events in reconstructed daily climate grids."""

from __future__ import annotations

import datetime
import math
import re
from typing import Any

_SCHEMA = "climate-grid/extremes-v1"
_TEMPORAL_SCHEMA = "climate-grid/temporal-v1"
_TEMPORAL_KEYS = frozenset({"schema", "times", "lats", "lons", "data"})
_SERIES_KEYS = frozenset({"values", "status", "uncertainty"})
_STATUSES = frozenset({"observed", "interpolated", "missing"})
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


def _validate_axis(axis: Any, name: str) -> None:
    if not isinstance(axis, list):
        raise TypeError(f"{name} must be a list")
    if len(axis) == 0:
        raise ValueError(f"{name} must be non-empty")
    for index, value in enumerate(axis):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"{name}[{index}] must be a finite non-bool int or float"
            )
        if not math.isfinite(value):
            raise ValueError(f"{name}[{index}] must be finite")
    for index in range(1, len(axis)):
        if axis[index] <= axis[index - 1]:
            raise ValueError(f"{name} must be strictly increasing")


def _validate_number(cell: Any, target: str, *, nullable: bool) -> None:
    if nullable and cell is None:
        return
    if not isinstance(cell, (int, float)) or isinstance(cell, bool):
        raise TypeError(f"{target} must be a number" + (" or None" if nullable else ""))
    if not math.isfinite(cell):
        raise ValueError(f"{target} must be finite")


def _validate_temporal(temporal: Any) -> tuple[list[str], list, list, dict]:
    if not isinstance(temporal, dict):
        raise TypeError("temporal must be a dict")
    if set(temporal.keys()) != _TEMPORAL_KEYS:
        raise ValueError(
            "temporal must have exactly the keys schema, times, lats, lons, data"
        )
    if not isinstance(temporal["schema"], str):
        raise TypeError("temporal.schema must be a str")
    if temporal["schema"] != _TEMPORAL_SCHEMA:
        raise ValueError(f"temporal.schema must be {_TEMPORAL_SCHEMA!r}")

    times = temporal["times"]
    if not isinstance(times, list):
        raise TypeError("temporal.times must be a list")
    if len(times) == 0:
        raise ValueError("temporal.times must be non-empty")
    days = [
        _parse_date(value, f"temporal.times[{index}]")
        for index, value in enumerate(times)
    ]
    for index in range(1, len(days)):
        if (days[index] - days[index - 1]).days != 1:
            raise ValueError(
                "temporal.times must be strictly increasing consecutive days"
            )

    lats = temporal["lats"]
    lons = temporal["lons"]
    _validate_axis(lats, "temporal.lats")
    _validate_axis(lons, "temporal.lons")

    data = temporal["data"]
    if not isinstance(data, dict):
        raise TypeError("temporal.data must be a dict")
    if len(data) == 0:
        raise ValueError("temporal.data must be non-empty")

    n_times = len(times)
    n_lat = len(lats)
    n_lon = len(lons)
    for element, series in data.items():
        if not isinstance(element, str):
            raise TypeError("temporal.data element names must be str")
        if element == "":
            raise ValueError("temporal.data element names must be non-empty")
        where = f"temporal.data[{element!r}]"
        if not isinstance(series, dict):
            raise TypeError(f"{where} must be a dict")
        if set(series.keys()) != _SERIES_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys values, status, uncertainty"
            )
        _validate_series(
            series, n_times, n_lat, n_lon, where
        )

    return times, lats, lons, data


def _validate_series(
    series: dict, n_times: int, n_lat: int, n_lon: int, where: str
) -> None:
    for member in ("values", "status", "uncertainty"):
        _validate_member(
            series[member], n_times, n_lat, n_lon, f"{where}.{member}", member
        )
    values = series["values"]
    status = series["status"]
    uncertainty = series["uncertainty"]
    for t in range(n_times):
        for i in range(n_lat):
            for j in range(n_lon):
                target = f"{where}[{t}][{i}][{j}]"
                if status[t][i][j] == "missing":
                    if values[t][i][j] is not None:
                        raise ValueError(
                            f"{target}: missing cells must have a None value"
                        )
                    if uncertainty[t][i][j] is not None:
                        raise ValueError(
                            f"{target}: missing cells must have a None uncertainty"
                        )
                else:
                    if values[t][i][j] is None:
                        raise ValueError(
                            f"{target}: non-missing cells must have a value"
                        )
                    if uncertainty[t][i][j] is None:
                        raise ValueError(
                            f"{target}: non-missing cells must have an uncertainty"
                        )


def _validate_member(
    member: Any, n_times: int, n_lat: int, n_lon: int, name: str, kind: str
) -> None:
    if not isinstance(member, list):
        raise TypeError(f"{name} must be a list")
    if len(member) != n_times:
        raise ValueError(f"{name} must have {n_times} frames (one per time)")
    for t, frame in enumerate(member):
        frame_where = f"{name}[{t}]"
        if not isinstance(frame, list):
            raise TypeError(f"{frame_where} must be a list")
        if len(frame) != n_lat:
            raise ValueError(f"{frame_where} must have {n_lat} rows (one per lat)")
        for i, row in enumerate(frame):
            row_where = f"{frame_where}[{i}]"
            if not isinstance(row, list):
                raise TypeError(f"{row_where} must be a list")
            if len(row) != n_lon:
                raise ValueError(
                    f"{row_where} must have {n_lon} cells (one per lon)"
                )
            for j, cell in enumerate(row):
                target = f"{row_where}[{j}]"
                if kind == "status":
                    if not isinstance(cell, str):
                        raise TypeError(f"{target} must be a str")
                    if cell not in _STATUSES:
                        raise ValueError(
                            f"{target} must be one of observed, interpolated, missing"
                        )
                else:
                    _validate_number(cell, target, nullable=True)


def detect(temporal, element, threshold, *, min_duration: int = 2) -> dict:
    """Detect threshold-exceeding extreme events in a reconstructed series.

    ``temporal`` must be a complete :func:`climate_grid.temporal.reconstruct`
    result (schema ``climate-grid/temporal-v1``); ``element`` a non-empty str
    naming one of its elements; ``threshold`` a finite non-bool number; and
    ``min_duration`` a non-bool positive int (default 2).

    A voxel ``(t, i, j)`` is every time/lat/lon cell of the element whose
    value exceeds ``threshold`` and whose status is not ``missing``.  Voxels
    are connected when they share a calendar day and are four-neighbours on
    the grid, or share a grid cell on adjacent days; an event is a
    three-dimensional connected component spanning at least ``min_duration``
    distinct dates.  Events are ordered ascending by their smallest
    ``(t, i, j)`` in lexicographic order.

    The returned mapping uses the key order ``schema, events``; ``schema`` is
    ``climate-grid/extremes-v1``.  Each event uses the key order ``start,
    end, days, cells, peak, mean, uncertainty``: first and last date
    (``YYYY-MM-DD``), number of distinct dates, the deduplicated ``[i, j]``
    cells in ascending order, the maximum and mean of ``value - threshold``
    over the event's voxels, and the mean of their uncertainties.  Every
    output float is ``round(x, 12)`` with negative zero normalized to
    ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    times, lats, lons, data = _validate_temporal(temporal)

    if not isinstance(element, str):
        raise TypeError("element must be a str")
    if element == "":
        raise ValueError("element must be non-empty")
    if element not in data:
        raise ValueError(f"unknown element: {element!r}")

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a finite non-bool int or float")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")

    if not isinstance(min_duration, int) or isinstance(min_duration, bool):
        raise TypeError("min_duration must be a non-bool int")
    if min_duration < 1:
        raise ValueError("min_duration must be positive")

    series = data[element]
    values = series["values"]
    status = series["status"]
    uncertainty = series["uncertainty"]
    n_times = len(times)
    n_lat = len(lats)
    n_lon = len(lons)

    voxels: set[tuple[int, int, int]] = set()
    for t in range(n_times):
        for i in range(n_lat):
            for j in range(n_lon):
                if (
                    status[t][i][j] != "missing"
                    and values[t][i][j] > threshold
                ):
                    voxels.add((t, i, j))

    components: list[list[tuple[int, int, int]]] = []
    remaining = set(voxels)
    while remaining:
        seed = min(remaining)
        remaining.discard(seed)
        stack = [seed]
        component = []
        while stack:
            t, i, j = stack.pop()
            component.append((t, i, j))
            for neighbour in (
                (t - 1, i, j),
                (t + 1, i, j),
                (t, i - 1, j),
                (t, i + 1, j),
                (t, i, j - 1),
                (t, i, j + 1),
            ):
                if neighbour in remaining:
                    remaining.discard(neighbour)
                    stack.append(neighbour)
        if len({t for t, _, _ in component}) >= min_duration:
            components.append(component)

    components.sort(key=min)

    events = []
    for component in components:
        dates = sorted({t for t, _, _ in component})
        cells = sorted({(i, j) for _, i, j in component})
        excesses = [
            values[t][i][j] - threshold for t, i, j in component
        ]
        uncertainties = [uncertainty[t][i][j] for t, i, j in component]
        events.append(
            {
                "start": times[dates[0]],
                "end": times[dates[-1]],
                "days": len(dates),
                "cells": [[i, j] for i, j in cells],
                "peak": _round_output(max(excesses)),
                "mean": _round_output(sum(excesses) / len(excesses)),
                "uncertainty": _round_output(
                    sum(uncertainties) / len(uncertainties)
                ),
            }
        )

    return {"schema": _SCHEMA, "events": events}
