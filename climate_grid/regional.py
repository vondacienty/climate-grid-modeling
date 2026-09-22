"""Regional aggregation of reconstructed daily climate grids."""

from __future__ import annotations

import datetime
import math
import re
from typing import Any

_SCHEMA = "climate-grid/regional-v1"
_TEMPORAL_SCHEMA = "climate-grid/temporal-v1"
_TEMPORAL_KEYS = frozenset({"schema", "times", "lats", "lons", "data"})
_SERIES_KEYS = frozenset({"values", "status", "uncertainty"})
_REGION_KEYS = frozenset({"name", "cells"})
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


def _validate_regions(
    regions: Any, n_lat: int, n_lon: int
) -> list[tuple[str, list[tuple[int, int]]]]:
    if not isinstance(regions, list):
        raise TypeError("regions must be a list")
    if len(regions) == 0:
        raise ValueError("regions must be non-empty")

    parsed: list[tuple[str, list[tuple[int, int]]]] = []
    seen_names: set[str] = set()
    for index, region in enumerate(regions):
        where = f"regions[{index}]"
        if not isinstance(region, dict):
            raise TypeError(f"{where} must be a dict")
        if set(region.keys()) != _REGION_KEYS:
            raise ValueError(f"{where} must have exactly the keys name, cells")

        name = region["name"]
        if not isinstance(name, str):
            raise TypeError(f"{where}.name must be a str")
        if name == "":
            raise ValueError(f"{where}.name must be non-empty")
        if name in seen_names:
            raise ValueError(f"duplicate region name: {name!r}")
        seen_names.add(name)

        cells = region["cells"]
        if not isinstance(cells, list):
            raise TypeError(f"{where}.cells must be a list")
        if len(cells) == 0:
            raise ValueError(f"{where}.cells must be non-empty")

        parsed_cells: list[tuple[int, int]] = []
        previous: tuple[int, int] | None = None
        for cell_index, cell in enumerate(cells):
            target = f"{where}.cells[{cell_index}]"
            if not isinstance(cell, list):
                raise TypeError(f"{target} must be a [i, j] list")
            if len(cell) != 2:
                raise ValueError(f"{target} must have exactly 2 items [i, j]")
            i, j = cell
            for axis, value, limit in (
                ("i", i, n_lat),
                ("j", j, n_lon),
            ):
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"{target}: {axis} must be a non-bool int")
                if value < 0:
                    raise ValueError(f"{target}: {axis} must be non-negative")
                if value >= limit:
                    raise ValueError(f"{target}: {axis} must be within the grid")
            pair = (i, j)
            if previous is not None and pair <= previous:
                raise ValueError(
                    f"{where}.cells must be ascending and non-repeating"
                )
            previous = pair
            parsed_cells.append(pair)

        parsed.append((name, parsed_cells))

    return parsed


def aggregate(temporal, element, regions, *, min_count: int = 1) -> dict:
    """Aggregate a reconstructed daily series over named regions.

    ``temporal`` must be a complete :func:`climate_grid.temporal.reconstruct`
    result (schema ``climate-grid/temporal-v1``); ``element`` a non-empty str
    naming one of its elements; ``regions`` a non-empty list of region dicts
    with exactly the keys ``name`` and ``cells``, where ``name`` is a unique
    non-empty str and ``cells`` a non-empty list of ascending, non-repeating
    ``[i, j]`` pairs whose ``i``/``j`` are non-bool non-negative ints within
    the grid; and ``min_count`` a non-bool positive int (default 1).

    For each region and each day, the region's cells whose status is not
    ``missing`` are collected; ``count`` is their number.  When fewer than
    ``min_count`` cells are available, ``mean``, ``min``, ``max`` and
    ``uncertainty`` are all ``None`` for that day; otherwise they are the
    mean, minimum and maximum of the cell values and
    ``sqrt(sum(u**2)) / count`` over the cell uncertainties.

    The returned mapping uses the key order ``schema, element, times,
    regions, data``; ``schema`` is ``climate-grid/regional-v1``, ``element``
    and ``times`` echo the input, and ``regions`` is the list of region
    names in input order.  ``data`` maps each region name, in input order,
    to a mapping with the key order ``count, mean, min, max, uncertainty``,
    each member a day-aligned sequence as long as ``times``.  Every output
    float is ``round(x, 12)`` with negative zero normalized to ``0.0``.
    Inputs are not modified.

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

    parsed_regions = _validate_regions(regions, len(lats), len(lons))

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    series = data[element]
    values = series["values"]
    status = series["status"]
    uncertainty = series["uncertainty"]
    n_times = len(times)

    out_data: dict[str, dict] = {}
    for name, cells in parsed_regions:
        counts: list[int] = []
        means: list[float | None] = []
        minimums: list[float | None] = []
        maximums: list[float | None] = []
        uncertainties: list[float | None] = []
        for t in range(n_times):
            day_values = [
                values[t][i][j]
                for i, j in cells
                if status[t][i][j] != "missing"
            ]
            count = len(day_values)
            counts.append(count)
            if count < min_count:
                means.append(None)
                minimums.append(None)
                maximums.append(None)
                uncertainties.append(None)
                continue
            day_uncertainties = [
                uncertainty[t][i][j]
                for i, j in cells
                if status[t][i][j] != "missing"
            ]
            means.append(_round_output(sum(day_values) / count))
            minimums.append(_round_output(min(day_values)))
            maximums.append(_round_output(max(day_values)))
            uncertainties.append(
                _round_output(
                    math.sqrt(sum(u * u for u in day_uncertainties)) / count
                )
            )
        out_data[name] = {
            "count": counts,
            "mean": means,
            "min": minimums,
            "max": maximums,
            "uncertainty": uncertainties,
        }

    return {
        "schema": _SCHEMA,
        "element": element,
        "times": list(times),
        "regions": [name for name, _ in parsed_regions],
        "data": out_data,
    }
