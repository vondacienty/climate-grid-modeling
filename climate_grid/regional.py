"""Regional aggregation of reconstructed daily climate grids."""

from __future__ import annotations

import datetime
import math
import re
from typing import Any

_SCHEMA = "climate-grid/regional-v1"
_MULTI_SCHEMA = "climate-grid/regional-multi-v1"
_WINDOW_SCHEMA = "climate-grid/window-v1"
_TEMPORAL_SCHEMA = "climate-grid/temporal-v1"
_TEMPORAL_KEYS = frozenset({"schema", "times", "lats", "lons", "data"})
_SERIES_KEYS = frozenset({"values", "status", "uncertainty"})
_STATUSES = frozenset({"observed", "interpolated", "missing"})
_REGION_KEYS = frozenset({"name", "cells"})
_WINDOW_KEYS = frozenset({"name", "start", "end"})
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


def _validate_number(cell: Any, target: str) -> None:
    if cell is None:
        return
    if not isinstance(cell, (int, float)) or isinstance(cell, bool):
        raise TypeError(f"{target} must be a number or None")
    if not math.isfinite(cell):
        raise ValueError(f"{target} must be finite")


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
                    _validate_number(cell, target)


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


def _validate_temporal(temporal: Any) -> tuple[list, list, list, dict]:
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
        _validate_series(series, n_times, n_lat, n_lon, where)

    return times, lats, lons, data


def _validate_regions(
    regions: Any, n_lat: int, n_lon: int
) -> list[tuple[str, list[tuple[int, int]]]]:
    if not isinstance(regions, list):
        raise TypeError("regions must be a list")
    if len(regions) == 0:
        raise ValueError("regions must be non-empty")

    validated: list[tuple[str, list[tuple[int, int]]]] = []
    seen_names: set[str] = set()
    for r_index, region in enumerate(regions):
        where = f"regions[{r_index}]"
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
        for c_index, cell in enumerate(cells):
            target = f"{where}.cells[{c_index}]"
            if not isinstance(cell, list):
                raise TypeError(f"{target} must be a list")
            if len(cell) != 2:
                raise ValueError(f"{target} must have exactly two items [i, j]")
            for axis, coordinate in enumerate(cell):
                label = "i" if axis == 0 else "j"
                if not isinstance(coordinate, int) or isinstance(coordinate, bool):
                    raise TypeError(f"{target}.{label} must be a non-bool int")
                if coordinate < 0:
                    raise ValueError(f"{target}.{label} must be non-negative")
                limit = n_lat if axis == 0 else n_lon
                if coordinate >= limit:
                    raise ValueError(
                        f"{target}.{label} must be within the grid "
                        f"(0 <= {label} < {limit})"
                    )
            pair = (cell[0], cell[1])
            if previous is not None and pair <= previous:
                raise ValueError(
                    f"{where}.cells must be strictly ascending and unique"
                )
            previous = pair
            parsed_cells.append(pair)

        validated.append((name, parsed_cells))

    return validated


def _region_series(
    values: list,
    status: list,
    uncertainty: list,
    cells: list[tuple[int, int]],
    n_times: int,
    min_count: int,
) -> dict:
    counts: list[int] = []
    means: list[float | None] = []
    mins: list[float | None] = []
    maxs: list[float | None] = []
    uncertainties: list[float | None] = []

    for t in range(n_times):
        day_values = [
            values[t][i][j] for i, j in cells if status[t][i][j] != "missing"
        ]
        count = len(day_values)
        counts.append(count)
        if count < min_count:
            means.append(None)
            mins.append(None)
            maxs.append(None)
            uncertainties.append(None)
            continue

        day_uncertainties = [
            uncertainty[t][i][j]
            for i, j in cells
            if status[t][i][j] != "missing"
        ]
        means.append(_round_output(sum(day_values) / count))
        mins.append(_round_output(min(day_values)))
        maxs.append(_round_output(max(day_values)))
        uncertainties.append(
            _round_output(
                math.sqrt(sum(u ** 2 for u in day_uncertainties)) / count
            )
        )

    return {
        "count": counts,
        "mean": means,
        "min": mins,
        "max": maxs,
        "uncertainty": uncertainties,
    }


def _validate_windows(
    windows: Any, times: list
) -> list[tuple[str, str, str, int, int]]:
    if not isinstance(windows, list):
        raise TypeError("windows must be a list")
    if len(windows) == 0:
        raise ValueError("windows must be non-empty")

    day_index = {day: index for index, day in enumerate(times)}
    validated: list[tuple[str, str, str, int, int]] = []
    seen_names: set[str] = set()
    for w_index, window in enumerate(windows):
        where = f"windows[{w_index}]"
        if not isinstance(window, dict):
            raise TypeError(f"{where} must be a dict")
        if list(window.keys()) != ["name", "start", "end"]:
            raise ValueError(
                f"{where} must have exactly the keys name, start, end in order"
            )

        name = window["name"]
        if not isinstance(name, str):
            raise TypeError(f"{where}.name must be a str")
        if name == "":
            raise ValueError(f"{where}.name must be non-empty")
        if name in seen_names:
            raise ValueError(f"duplicate window name: {name!r}")
        seen_names.add(name)

        start = window["start"]
        end = window["end"]
        start_day = _parse_date(start, f"{where}.start")
        end_day = _parse_date(end, f"{where}.end")
        if start not in day_index:
            raise ValueError(f"{where}.start must be within temporal.times")
        if end not in day_index:
            raise ValueError(f"{where}.end must be within temporal.times")
        if start_day > end_day:
            raise ValueError(f"{where}.start must be on or before {where}.end")

        validated.append((name, start, end, day_index[start], day_index[end]))

    return validated


def _window_region(
    values: list,
    status: list,
    uncertainty: list,
    cells: list[tuple[int, int]],
    t_start: int,
    t_end: int,
    min_count: int,
) -> dict:
    window_values: list[float] = []
    window_uncertainties: list[float] = []
    for t in range(t_start, t_end + 1):
        for i, j in cells:
            if status[t][i][j] != "missing":
                window_values.append(values[t][i][j])
                window_uncertainties.append(uncertainty[t][i][j])

    count = len(window_values)
    if count < min_count:
        return {
            "count": count,
            "mean": None,
            "min": None,
            "max": None,
            "uncertainty": None,
        }

    return {
        "count": count,
        "mean": _round_output(sum(window_values) / count),
        "min": _round_output(min(window_values)),
        "max": _round_output(max(window_values)),
        "uncertainty": _round_output(
            math.sqrt(sum(u ** 2 for u in window_uncertainties)) / count
        ),
    }


def aggregate(temporal, element, regions, *, min_count: int = 1) -> dict:
    """Aggregate a reconstructed grid series into per-region daily statistics.

    ``temporal`` must be a complete :func:`climate_grid.temporal.reconstruct`
    result (schema ``climate-grid/temporal-v1``); ``element`` a non-empty str
    naming one of its elements; ``regions`` a non-empty list of dicts, each
    with exactly the keys ``name`` and ``cells``.  ``name`` is a unique
    non-empty str and ``cells`` a non-empty list of unique ``[i, j]`` pairs
    in strictly ascending (lexicographic) order, where ``i`` and ``j`` are
    non-bool non-negative ints inside the grid.  ``min_count`` is a non-bool
    positive int.

    For every region and day, the cells whose status is not ``missing`` are
    retained.  ``count`` is their number; when it is below ``min_count``,
    ``mean``, ``min``, ``max`` and ``uncertainty`` are all ``None``.
    Otherwise they are the arithmetic mean, minimum and maximum of the cell
    values, and ``sqrt(sum(u ** 2)) / count`` over the cells' uncertainties.

    The returned mapping uses the key order ``schema, element, times,
    regions, data``; ``schema`` is ``climate-grid/regional-v1``, ``element``
    echoes the argument, ``times`` is passed through unchanged and
    ``regions`` lists the region names in input order.  ``data`` follows the
    same order; each entry uses the key order ``count, mean, min, max,
    uncertainty``, all five members being equally long day series.  Every
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

    validated_regions = _validate_regions(regions, len(lats), len(lons))

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    series = data[element]
    values = series["values"]
    status = series["status"]
    uncertainty = series["uncertainty"]
    n_times = len(times)

    result_data = []
    for name, cells in validated_regions:
        result_data.append(
            _region_series(
                values, status, uncertainty, cells, n_times, min_count
            )
        )

    return {
        "schema": _SCHEMA,
        "element": element,
        "times": times,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


def aggregate_multi(temporal, elements, regions, *, min_count: int = 1) -> dict:
    """Aggregate a reconstructed grid series for several elements at once.

    Behaves like :func:`aggregate` but ``elements`` is a non-empty list of
    unique non-empty str element names that must all exist in
    ``temporal.data``; wrong element item types raise ``TypeError`` while an
    empty, duplicate or unknown element raises ``ValueError``.

    The returned mapping uses the key order ``schema, elements, times,
    regions, data``; ``schema`` is ``climate-grid/regional-multi-v1``,
    ``elements`` and ``times`` are passed through unchanged and ``regions``
    lists the region names in input order.  ``data`` follows the region
    order; each entry uses the key order ``name, elements``, where
    ``elements`` follows the requested element order and each entry uses the
    key order ``element, count, mean, min, max, uncertainty``, all six
    members being equally long day series.  Per-day statistics and float
    rounding match :func:`aggregate`.  Inputs are not modified.
    """
    times, lats, lons, data = _validate_temporal(temporal)

    if not isinstance(elements, list):
        raise TypeError("elements must be a list")
    if len(elements) == 0:
        raise ValueError("elements must be non-empty")
    seen_elements: set[str] = set()
    for index, element in enumerate(elements):
        if not isinstance(element, str):
            raise TypeError(f"elements[{index}] must be a str")
        if element == "":
            raise ValueError(f"elements[{index}] must be non-empty")
        if element in seen_elements:
            raise ValueError(f"duplicate element: {element!r}")
        seen_elements.add(element)
        if element not in data:
            raise ValueError(f"unknown element: {element!r}")

    validated_regions = _validate_regions(regions, len(lats), len(lons))

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    n_times = len(times)
    result_data = []
    for name, cells in validated_regions:
        element_series = []
        for element in elements:
            series = data[element]
            stats = _region_series(
                series["values"],
                series["status"],
                series["uncertainty"],
                cells,
                n_times,
                min_count,
            )
            element_series.append({"element": element, **stats})
        result_data.append({"name": name, "elements": element_series})

    return {
        "schema": _MULTI_SCHEMA,
        "elements": elements,
        "times": times,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


def aggregate_window(temporal, element, regions, windows, *, min_count: int = 1) -> dict:
    """Aggregate a reconstructed grid series into per-window region statistics.

    Behaves like :func:`aggregate`, but instead of per-day statistics every
    statistic is computed once per region over a closed calendar window.
    ``windows`` is a non-empty list of dicts, each with exactly the keys
    ``name``, ``start`` and ``end`` in that order: ``name`` is a unique
    non-empty str and ``start``/``end`` are valid ``YYYY-MM-DD`` dates that
    occur in ``temporal.times`` with ``start <= end``.  Wrong container,
    ``name`` or date types raise ``TypeError``; every other window contract
    violation raises ``ValueError``.

    For every window (in window order) and region (in region order), every
    non-``missing`` cell of every day of the inclusive interval contributes a
    sample.  ``count`` is the number of samples; when it is below
    ``min_count``, ``mean``, ``min``, ``max`` and ``uncertainty`` are all
    ``None``.  Otherwise they are the arithmetic mean, minimum and maximum of
    the sample values, and ``sqrt(sum(u ** 2)) / count`` over the samples'
    uncertainties.

    The returned mapping uses the key order ``schema, element, windows,
    regions, data``; ``schema`` is ``climate-grid/window-v1``, ``element``
    echoes the argument, ``windows`` lists three-key dicts in input order and
    ``regions`` lists the region names in input order.  ``data`` follows the
    window order; each entry uses the key order ``name, regions``, where each
    region entry uses the key order ``name, count, mean, min, max,
    uncertainty``.  ``count`` is an int and every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs are
    not modified.
    """
    times, lats, lons, data = _validate_temporal(temporal)

    if not isinstance(element, str):
        raise TypeError("element must be a str")
    if element == "":
        raise ValueError("element must be non-empty")
    if element not in data:
        raise ValueError(f"unknown element: {element!r}")

    validated_regions = _validate_regions(regions, len(lats), len(lons))
    validated_windows = _validate_windows(windows, times)

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    series = data[element]
    values = series["values"]
    status = series["status"]
    uncertainty = series["uncertainty"]

    result_data = []
    for w_name, _start, _end, t_start, t_end in validated_windows:
        region_stats = []
        for r_name, cells in validated_regions:
            stats = _window_region(
                values,
                status,
                uncertainty,
                cells,
                t_start,
                t_end,
                min_count,
            )
            region_stats.append({"name": r_name, **stats})
        result_data.append({"name": w_name, "regions": region_stats})

    return {
        "schema": _WINDOW_SCHEMA,
        "element": element,
        "windows": [
            {"name": name, "start": start, "end": end}
            for name, start, end, _t_start, _t_end in validated_windows
        ],
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }
