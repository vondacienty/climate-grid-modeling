"""Extreme-event detection on reconstructed daily gridded datasets."""

from __future__ import annotations

import datetime
import math
import re
from typing import Any

_SCHEMA = "climate-grid/extremes-v1"
_TEMPORAL_SCHEMA = "climate-grid/temporal-v1"
_RESULT_KEYS = frozenset({"schema", "times", "lats", "lons", "data"})
_ELEMENT_KEYS = frozenset({"values", "status", "uncertainty"})
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
        raise TypeError(f"{where} entries must be str")
    if _DATE_RE.fullmatch(value) is None:
        raise ValueError(f"{where} entries must be YYYY-MM-DD dates")
    year, month, day = (int(part) for part in value.split("-"))
    try:
        return datetime.date(year, month, day)
    except ValueError:
        raise ValueError(f"{where} entries must be valid Gregorian dates") from None


def _validate_axis(axis: Any, name: str) -> None:
    if not isinstance(axis, list):
        raise TypeError(f"{name} must be a list")
    if len(axis) == 0:
        raise ValueError(f"{name} must be non-empty")
    for index, value in enumerate(axis):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name}[{index}] must be a finite non-bool int or float")
        if not math.isfinite(value):
            raise ValueError(f"{name}[{index}] must be finite")
    for index in range(1, len(axis)):
        if axis[index] <= axis[index - 1]:
            raise ValueError(f"{name} must be strictly increasing")


def _validate_temporal(temporal: Any) -> tuple[int, int, int, dict]:
    if not isinstance(temporal, dict):
        raise TypeError("temporal must be a dict")
    if set(temporal.keys()) != _RESULT_KEYS:
        raise ValueError(
            "temporal must have exactly the keys schema, times, lats, lons, data"
        )

    schema = temporal["schema"]
    if not isinstance(schema, str):
        raise TypeError("temporal.schema must be a str")
    if schema != _TEMPORAL_SCHEMA:
        raise ValueError(f"temporal.schema must be {_TEMPORAL_SCHEMA!r}")

    times = temporal["times"]
    if not isinstance(times, list):
        raise TypeError("temporal.times must be a list")
    if len(times) == 0:
        raise ValueError("temporal.times must be non-empty")
    previous_day = None
    seen_days: set[datetime.date] = set()
    for index, value in enumerate(times):
        day = _parse_date(value, "temporal.times")
        if day in seen_days:
            raise ValueError(f"duplicate temporal.times entry: {value!r}")
        if previous_day is not None and day <= previous_day:
            raise ValueError("temporal.times must be strictly increasing")
        seen_days.add(day)
        previous_day = day

    _validate_axis(temporal["lats"], "temporal.lats")
    _validate_axis(temporal["lons"], "temporal.lons")

    data = temporal["data"]
    if not isinstance(data, dict):
        raise TypeError("temporal.data must be a dict")
    if len(data) == 0:
        raise ValueError("temporal.data must be non-empty")

    n_times = len(times)
    n_lat = len(temporal["lats"])
    n_lon = len(temporal["lons"])
    for element, item in data.items():
        if not isinstance(element, str):
            raise TypeError("temporal.data element names must be str")
        if element == "":
            raise ValueError("temporal.data element names must be non-empty")
        _validate_element(item, element, n_times, n_lat, n_lon)

    return n_times, n_lat, n_lon, data


def _check_cell_number(cell: Any, target: str) -> None:
    if cell is not None:
        if not isinstance(cell, (int, float)) or isinstance(cell, bool):
            raise TypeError(f"{target} must be a number or None")
        if not math.isfinite(cell):
            raise ValueError(f"{target} must be finite")


def _validate_element(
    item: Any, element: str, n_times: int, n_lat: int, n_lon: int
) -> None:
    where = f"temporal.data[{element!r}]"
    if not isinstance(item, dict):
        raise TypeError(f"{where} must be a dict")
    if set(item.keys()) != _ELEMENT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys values, status, uncertainty"
        )

    values = item["values"]
    status = item["status"]
    uncertainty = item["uncertainty"]
    members = (
        ("values", values),
        ("status", status),
        ("uncertainty", uncertainty),
    )
    for name, member in members:
        if not isinstance(member, list):
            raise TypeError(f"{where}.{name} must be a list")
        if len(member) != n_times:
            raise ValueError(f"{where}.{name} must have {n_times} entries (one per time)")

    for t in range(n_times):
        for name, member in members:
            if not isinstance(member[t], list):
                raise TypeError(f"{where}.{name}[{t}] must be a list")
            if len(member[t]) != n_lat:
                raise ValueError(f"{where}.{name}[{t}] must have {n_lat} rows (one per lat)")

        for i in range(n_lat):
            for name, member in members:
                if not isinstance(member[t][i], list):
                    raise TypeError(f"{where}.{name}[{t}][{i}] must be a list")
                if len(member[t][i]) != n_lon:
                    raise ValueError(
                        f"{where}.{name}[{t}][{i}] must have {n_lon} cells (one per lon)"
                    )

            for j in range(n_lon):
                cell_status = status[t][i][j]
                if not isinstance(cell_status, str):
                    raise TypeError(f"{where}.status[{t}][{i}][{j}] must be a str")
                if cell_status not in _STATUSES:
                    raise ValueError(
                        f"{where}.status[{t}][{i}][{j}] must be one of "
                        "observed, interpolated, missing"
                    )
                _check_cell_number(values[t][i][j], f"{where}.values[{t}][{i}][{j}]")
                _check_cell_number(
                    uncertainty[t][i][j], f"{where}.uncertainty[{t}][{i}][{j}]"
                )

                value_cell = values[t][i][j]
                uncertainty_cell = uncertainty[t][i][j]
                if cell_status == "missing":
                    if value_cell is not None or uncertainty_cell is not None:
                        raise ValueError(
                            f"{where}: missing status at ({t}, {i}, {j}) requires "
                            "values and uncertainty to be None"
                        )
                elif value_cell is None or uncertainty_cell is None:
                    raise ValueError(
                        f"{where}: non-missing status at ({t}, {i}, {j}) requires "
                        "numeric values and uncertainty"
                    )


def detect(temporal, element, threshold, *, min_duration: int = 2) -> dict:
    """Detect threshold-exceeding 3-D connected extreme events.

    ``temporal`` must be a complete :func:`climate_grid.temporal.reconstruct`
    result (schema ``climate-grid/temporal-v1``); ``element`` must be a
    non-empty str naming one of its ``data`` entries; ``threshold`` must be a
    finite non-bool int or float; ``min_duration`` must be a non-bool positive
    int (default ``2``).

    A grid cell at ``(t, i, j)`` is an event voxel when its ``value`` is
    strictly greater than ``threshold`` and its status is not ``missing``.
    Voxels connect when they are spatial four-neighbours on the same day or
    the same cell on adjacent days.  An event is a 3-D connected component
    spanning at least ``min_duration`` distinct dates.

    The returned mapping uses the key order ``schema, events``; ``schema`` is
    ``climate-grid/extremes-v1``.  Events are ordered by the minimum
    ``(t, i, j)`` voxel coordinate lexicographically, and each event uses the
    key order ``start, end, days, cells, peak, mean, uncertainty``: ``start``
    and ``end`` are the first/last date strings, ``days`` is the number of
    distinct dates, ``cells`` is the sorted de-duplicated list of ``[i, j]``
    cell pairs, ``peak`` is the maximum and ``mean`` the mean of
    ``value - threshold`` over the event's voxels, and ``uncertainty`` is the
    mean voxel uncertainty.  Every output float is ``round(x, 12)`` with
    negative zero normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong argument/container/item types and
    ``ValueError`` for any other contract violation.
    """
    n_times, n_lat, n_lon, data = _validate_temporal(temporal)

    if not isinstance(element, str):
        raise TypeError("element must be a str")
    if element == "" or element not in data:
        raise ValueError("element must be a non-empty str present in temporal.data")

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a finite non-bool int or float")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")

    if not isinstance(min_duration, int) or isinstance(min_duration, bool):
        raise TypeError("min_duration must be a non-bool int")
    if min_duration < 1:
        raise ValueError("min_duration must be a positive integer")

    times = temporal["times"]
    item = data[element]
    values = item["values"]
    status = item["status"]
    uncertainty = item["uncertainty"]

    voxel: set[tuple[int, int, int]] = set()
    for t in range(n_times):
        for i in range(n_lat):
            for j in range(n_lon):
                if (
                    status[t][i][j] != "missing"
                    and values[t][i][j] is not None
                    and values[t][i][j] > threshold
                ):
                    voxel.add((t, i, j))

    events: list[dict] = []
    anchors: list[tuple[int, int, int]] = []
    visited: set[tuple[int, int, int]] = set()
    for seed in sorted(voxel):
        if seed in visited:
            continue

        stack = [seed]
        visited.add(seed)
        component: list[tuple[int, int, int]] = []
        while stack:
            t, i, j = stack.pop()
            component.append((t, i, j))

            neighbours = (
                (t, i - 1, j),
                (t, i + 1, j),
                (t, i, j - 1),
                (t, i, j + 1),
                (t - 1, i, j),
                (t + 1, i, j),
            )
            for candidate in neighbours:
                ct, ci, cj = candidate
                if (
                    0 <= ct < n_times
                    and 0 <= ci < n_lat
                    and 0 <= cj < n_lon
                    and candidate in voxel
                    and candidate not in visited
                ):
                    visited.add(candidate)
                    stack.append(candidate)

        distinct_days = {t for t, _, _ in component}
        if len(distinct_days) < min_duration:
            continue

        anchor = min(component)
        excesses = [float(values[t][i][j] - threshold) for t, i, j in component]
        uncertainties = [float(uncertainty[t][i][j]) for t, i, j in component]
        cells = sorted({(i, j) for _, i, j in component})
        first_t = min(distinct_days)
        last_t = max(distinct_days)
        size = len(component)

        anchors.append(anchor)
        events.append(
            {
                "start": times[first_t],
                "end": times[last_t],
                "days": len(distinct_days),
                "cells": [[i, j] for i, j in cells],
                "peak": _round_output(max(excesses)),
                "mean": _round_output(sum(excesses) / size),
                "uncertainty": _round_output(sum(uncertainties) / size),
            }
        )

    ordered = [event for _, event in sorted(zip(anchors, events))]
    return {"schema": _SCHEMA, "events": ordered}
