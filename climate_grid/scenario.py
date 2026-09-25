"""Scenario downscaling of reconstructed daily climate grids."""

from __future__ import annotations

import datetime
import math
import re
from typing import Any

__all__ = ["downscale"]

_SCHEMA = "climate-grid/scenario-v1"
_TEMPORAL_SCHEMA = "climate-grid/temporal-v1"
_TEMPORAL_KEYS = frozenset({"schema", "times", "lats", "lons", "data"})
_SERIES_KEYS = frozenset({"values", "status", "uncertainty"})
_STATUSES = frozenset({"observed", "interpolated", "missing"})
_MODES = frozenset({"additive", "multiplicative"})
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


def _validate_temporal(temporal: Any) -> tuple[list[str], list, list, list, dict]:
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
    order: list[str] = []
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
        order.append(element)

    return order, times, lats, lons, data


def _validate_deltas(
    deltas: Any, order: list[str], n_times: int, n_lat: int, n_lon: int
) -> None:
    if not isinstance(deltas, dict):
        raise TypeError("deltas must be a dict")
    if list(deltas.keys()) != order:
        raise ValueError(
            "deltas must have exactly the same element keys and order as temporal.data"
        )
    for element in order:
        grid = deltas[element]
        name = f"deltas[{element!r}]"
        if not isinstance(grid, list):
            raise TypeError(f"{name} must be a list")
        if len(grid) != n_times:
            raise ValueError(f"{name} must have {n_times} frames (one per time)")
        for t, frame in enumerate(grid):
            frame_where = f"{name}[{t}]"
            if not isinstance(frame, list):
                raise TypeError(f"{frame_where} must be a list")
            if len(frame) != n_lat:
                raise ValueError(
                    f"{frame_where} must have {n_lat} rows (one per lat)"
                )
            for i, row in enumerate(frame):
                row_where = f"{frame_where}[{i}]"
                if not isinstance(row, list):
                    raise TypeError(f"{row_where} must be a list")
                if len(row) != n_lon:
                    raise ValueError(
                        f"{row_where} must have {n_lon} cells (one per lon)"
                    )
                for j, cell in enumerate(row):
                    _validate_number(cell, f"{row_where}[{j}]", nullable=True)


def downscale(temporal, deltas, *, mode: str = "additive") -> dict:
    """Apply per-cell scenario deltas to a reconstructed daily series.

    ``temporal`` must be a complete :func:`climate_grid.temporal.reconstruct`
    result (schema ``climate-grid/temporal-v1``).  ``deltas`` must be a dict
    with exactly the same element keys in exactly the same order as
    ``temporal.data``; each value is a three-dimensional ``[time][lat][lon]``
    list whose cells are finite non-bool numbers or ``None``.  ``mode`` is
    ``"additive"`` (``value + delta``) or ``"multiplicative"``
    (``value * (1 + delta)``).

    For every cell, a missing original status or a ``None`` delta yields
    ``values=None``, ``status="missing"`` and ``uncertainty=None``;
    otherwise the adjusted value is computed per ``mode`` while the original
    status and uncertainty are kept.

    The returned mapping uses the key order ``schema, times, lats, lons,
    data``; ``schema`` is ``climate-grid/scenario-v1`` and the axes are
    returned as-is.  ``data`` follows the temporal element order and each
    item uses the key order ``values, status, uncertainty``.  Every output
    value float is ``round(x, 12)`` with negative zero normalized to
    ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    order, times, lats, lons, data = _validate_temporal(temporal)

    if not isinstance(mode, str):
        raise TypeError("mode must be a str")
    if mode not in _MODES:
        raise ValueError("mode must be one of additive, multiplicative")

    n_times = len(times)
    n_lat = len(lats)
    n_lon = len(lons)
    _validate_deltas(deltas, order, n_times, n_lat, n_lon)

    result_data: dict[str, dict] = {}
    for element in order:
        series = data[element]
        values = series["values"]
        status = series["status"]
        uncertainty = series["uncertainty"]
        element_deltas = deltas[element]

        out_values = [
            [[None for _ in range(n_lon)] for _ in range(n_lat)]
            for _ in range(n_times)
        ]
        out_status = [
            [[None for _ in range(n_lon)] for _ in range(n_lat)]
            for _ in range(n_times)
        ]
        out_uncertainty = [
            [[None for _ in range(n_lon)] for _ in range(n_lat)]
            for _ in range(n_times)
        ]

        for t in range(n_times):
            for i in range(n_lat):
                for j in range(n_lon):
                    delta = element_deltas[t][i][j]
                    if status[t][i][j] == "missing" or delta is None:
                        out_status[t][i][j] = "missing"
                        continue
                    value = values[t][i][j]
                    if mode == "additive":
                        adjusted = value + delta
                    else:
                        adjusted = value * (1.0 + delta)
                    out_values[t][i][j] = _round_output(adjusted)
                    out_status[t][i][j] = status[t][i][j]
                    out_uncertainty[t][i][j] = uncertainty[t][i][j]

        result_data[element] = {
            "values": out_values,
            "status": out_status,
            "uncertainty": out_uncertainty,
        }

    return {
        "schema": _SCHEMA,
        "times": times,
        "lats": lats,
        "lons": lons,
        "data": result_data,
    }
