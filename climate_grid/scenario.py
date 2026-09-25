"""Scenario downscaling of reconstructed daily climate grids."""

from __future__ import annotations

import datetime
import math
import re
from typing import Any

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


def _validate_temporal(
    temporal: Any, *, schema: str = _TEMPORAL_SCHEMA, where: str = "temporal"
) -> tuple[list[str], list, list, dict]:
    if not isinstance(temporal, dict):
        raise TypeError(f"{where} must be a dict")
    if set(temporal.keys()) != _TEMPORAL_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, times, lats, lons, data"
        )
    if not isinstance(temporal["schema"], str):
        raise TypeError(f"{where}.schema must be a str")
    if temporal["schema"] != schema:
        raise ValueError(f"{where}.schema must be {schema!r}")

    times = temporal["times"]
    if not isinstance(times, list):
        raise TypeError(f"{where}.times must be a list")
    if len(times) == 0:
        raise ValueError(f"{where}.times must be non-empty")
    days = [
        _parse_date(value, f"{where}.times[{index}]")
        for index, value in enumerate(times)
    ]
    for index in range(1, len(days)):
        if (days[index] - days[index - 1]).days != 1:
            raise ValueError(
                f"{where}.times must be strictly increasing consecutive days"
            )

    lats = temporal["lats"]
    lons = temporal["lons"]
    _validate_axis(lats, f"{where}.lats")
    _validate_axis(lons, f"{where}.lons")

    data = temporal["data"]
    if not isinstance(data, dict):
        raise TypeError(f"{where}.data must be a dict")
    if len(data) == 0:
        raise ValueError(f"{where}.data must be non-empty")

    n_times = len(times)
    n_lat = len(lats)
    n_lon = len(lons)
    for element, series in data.items():
        if not isinstance(element, str):
            raise TypeError(f"{where}.data element names must be str")
        if element == "":
            raise ValueError(f"{where}.data element names must be non-empty")
        series_where = f"{where}.data[{element!r}]"
        if not isinstance(series, dict):
            raise TypeError(f"{series_where} must be a dict")
        if set(series.keys()) != _SERIES_KEYS:
            raise ValueError(
                f"{series_where} must have exactly the keys values, status, uncertainty"
            )
        _validate_series(
            series, n_times, n_lat, n_lon, series_where
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


def _validate_deltas(
    deltas: Any, elements: list[str], n_times: int, n_lat: int, n_lon: int
) -> None:
    if not isinstance(deltas, dict):
        raise TypeError("deltas must be a dict")
    if list(deltas.keys()) != elements:
        raise ValueError(
            "deltas keys must match temporal.data elements exactly and in order"
        )
    for element in elements:
        _validate_member(
            deltas[element],
            n_times,
            n_lat,
            n_lon,
            f"deltas[{element!r}]",
            "values",
        )


def downscale(temporal, deltas, *, mode: str = "additive") -> dict:
    """Apply per-cell scenario deltas to a reconstructed daily series.

    ``temporal`` must be a complete
    :func:`climate_grid.temporal.reconstruct` result (schema
    ``climate-grid/temporal-v1``).  ``deltas`` is a dict whose keys are
    exactly the ``temporal.data`` elements in the same order; each value is a
    nested ``[time][lat][lon]`` list whose cells are finite non-bool numbers
    or ``None``.  ``mode`` is ``"additive"`` (default) or
    ``"multiplicative"``.

    For every element and grid cell, a cell whose original status is
    ``missing`` or whose delta is ``None`` keeps ``values=None``,
    ``status="missing"`` and ``uncertainty=None``; otherwise the new value is
    ``value + delta`` (additive) or ``value * (1 + delta)`` (multiplicative)
    and the original ``status`` and ``uncertainty`` are carried over
    unchanged.

    The returned mapping uses the key order ``schema, times, lats, lons,
    data``; ``schema`` is ``climate-grid/scenario-v1`` and the axes are
    carried over as-is.  ``data`` follows the ``temporal.data`` element order
    and each item uses the key order ``values, status, uncertainty``; the
    three members are nested ``[time][lat][lon]``.  Every computed output
    number is ``round(x, 12)`` with negative zero normalized to ``0.0``.
    Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    times, lats, lons, data = _validate_temporal(temporal)

    n_times = len(times)
    n_lat = len(lats)
    n_lon = len(lons)
    elements = list(data.keys())
    _validate_deltas(deltas, elements, n_times, n_lat, n_lon)

    if not isinstance(mode, str):
        raise TypeError("mode must be a str")
    if mode not in _MODES:
        raise ValueError("mode must be 'additive' or 'multiplicative'")

    out_data: dict[str, dict] = {}
    for element in elements:
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
                    if mode == "additive":
                        adjusted = values[t][i][j] + delta
                    else:
                        adjusted = values[t][i][j] * (1 + delta)
                    out_values[t][i][j] = _round_output(adjusted)
                    out_status[t][i][j] = status[t][i][j]
                    out_uncertainty[t][i][j] = uncertainty[t][i][j]

        out_data[element] = {
            "values": out_values,
            "status": out_status,
            "uncertainty": out_uncertainty,
        }

    return {
        "schema": _SCHEMA,
        "times": list(times),
        "lats": list(lats),
        "lons": list(lons),
        "data": out_data,
    }


_ENSEMBLE_SCHEMA = "climate-grid/ensemble-v1"
_DEFAULT_QUANTILES = [0.05, 0.5, 0.95]


def _validate_quantiles(quantiles: Any) -> list:
    if quantiles is None:
        return list(_DEFAULT_QUANTILES)
    if not isinstance(quantiles, list):
        raise TypeError("quantiles must be a list")
    if len(quantiles) == 0:
        raise ValueError("quantiles must be non-empty")
    for index, q in enumerate(quantiles):
        if not isinstance(q, (int, float)) or isinstance(q, bool):
            raise TypeError(
                f"quantiles[{index}] must be a finite non-bool int or float"
            )
        if not math.isfinite(q):
            raise ValueError(f"quantiles[{index}] must be finite")
        if q < 0.0 or q > 1.0:
            raise ValueError(f"quantiles[{index}] must be between 0 and 1")
    for index in range(1, len(quantiles)):
        if quantiles[index] <= quantiles[index - 1]:
            raise ValueError("quantiles must be strictly increasing")
    return list(quantiles)


def _validate_scenarios(scenarios: Any) -> tuple[list, list, list, list[str], list]:
    if not isinstance(scenarios, list):
        raise TypeError("scenarios must be a list")
    if len(scenarios) == 0:
        raise ValueError("scenarios must be non-empty")

    validated = [
        _validate_temporal(
            scenario, schema=_SCHEMA, where=f"scenarios[{index}]"
        )
        for index, scenario in enumerate(scenarios)
    ]
    times, lats, lons, data = validated[0]
    elements = list(data.keys())
    for index in range(1, len(validated)):
        other_times, other_lats, other_lons, other_data = validated[index]
        if other_times != times or other_lats != lats or other_lons != lons:
            raise ValueError(
                "all scenarios must share identical times, lats and lons"
            )
        if list(other_data.keys()) != elements:
            raise ValueError(
                "all scenarios must have the same data elements in the same order"
            )
    return times, lats, lons, elements, validated


def _combine_cells(
    series_list: list, qs: list, n_times: int, n_lat: int, n_lon: int
) -> tuple[list, list, list, list]:
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
    out_quantiles = [
        [
            [[None for _ in range(n_lon)] for _ in range(n_lat)]
            for _ in range(n_times)
        ]
        for _ in qs
    ]

    for t in range(n_times):
        for i in range(n_lat):
            for j in range(n_lon):
                samples = [
                    (
                        series["values"][t][i][j],
                        series["uncertainty"][t][i][j],
                        series["status"][t][i][j],
                    )
                    for series in series_list
                    if series["status"][t][i][j] != "missing"
                ]
                if not samples:
                    out_status[t][i][j] = "missing"
                    continue
                n = len(samples)
                out_values[t][i][j] = _round_output(
                    sum(sample[0] for sample in samples) / n
                )
                out_status[t][i][j] = (
                    "observed"
                    if all(sample[2] == "observed" for sample in samples)
                    else "interpolated"
                )
                out_uncertainty[t][i][j] = _round_output(
                    math.sqrt(sum(sample[1] ** 2 for sample in samples)) / n
                )
                ordered = sorted(sample[0] for sample in samples)
                for q_index, q in enumerate(qs):
                    h = (n - 1) * q
                    lower = int(math.floor(h))
                    upper = min(lower + 1, n - 1)
                    fraction = h - lower
                    interpolated = ordered[lower] + (
                        ordered[upper] - ordered[lower]
                    ) * fraction
                    out_quantiles[q_index][t][i][j] = _round_output(interpolated)

    return out_values, out_status, out_uncertainty, out_quantiles


def ensemble(scenarios, element, *, quantiles=None) -> dict:
    """Combine scenario-v1 downscale results into a per-cell ensemble.

    ``scenarios`` is a non-empty list of :func:`downscale` results (schema
    ``climate-grid/scenario-v1``); every member is validated against that
    contract and all members must share identical ``times``, ``lats``,
    ``lons`` axes and the same ``data`` elements in the same order.
    ``element`` is a non-empty str naming one of those elements.
    ``quantiles`` is ``None`` (default ``[0.05, 0.5, 0.95]``) or a non-empty
    list of finite non-bool numbers in ``[0, 1]``, strictly increasing.

    For every ``[time][lat][lon]`` cell the non-missing ``(value,
    uncertainty)`` samples across the scenarios are combined: with no
    samples the cell gets ``values=None``, ``status="missing"`` and
    ``uncertainty=None``; otherwise ``values`` is the sample mean
    ``Σv / n``, ``status`` is ``"observed"`` when every sample is observed
    and ``"interpolated"`` otherwise, and ``uncertainty`` is
    ``√(Σu²) / n``.  Each requested quantile is computed from the ascending
    sample values by linear interpolation on the position ``h = (n - 1) q``.

    The returned mapping uses the key order ``schema, element, times, lats,
    lons, quantiles, values, status, uncertainty, quantile_values``;
    ``schema`` is ``climate-grid/ensemble-v1`` and the axes are carried over
    from the scenarios.  ``values``, ``status`` and ``uncertainty`` are
    nested ``[time][lat][lon]`` lists; ``quantile_values`` is nested
    ``[quantile][time][lat][lon]`` with ``None`` for sample-less cells.
    Every computed output number is ``round(x, 12)`` with negative zero
    normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    times, lats, lons, elements, validated = _validate_scenarios(scenarios)

    if not isinstance(element, str):
        raise TypeError("element must be a str")
    if element == "":
        raise ValueError("element must be non-empty")
    if element not in elements:
        raise ValueError(f"element {element!r} is not present in the scenarios")

    qs = _validate_quantiles(quantiles)

    series_list = [member[3][element] for member in validated]
    out_values, out_status, out_uncertainty, out_quantiles = _combine_cells(
        series_list, qs, len(times), len(lats), len(lons)
    )

    return {
        "schema": _ENSEMBLE_SCHEMA,
        "element": element,
        "times": list(times),
        "lats": list(lats),
        "lons": list(lons),
        "quantiles": qs,
        "values": out_values,
        "status": out_status,
        "uncertainty": out_uncertainty,
        "quantile_values": out_quantiles,
    }


_ENSEMBLE_MULTI_SCHEMA = "climate-grid/ensemble-multi-v1"


def ensemble_multi(scenarios, *, quantiles=None) -> dict:
    """Combine scenario-v1 downscale results into a per-element ensemble.

    ``scenarios`` is a non-empty list of :func:`downscale` results (schema
    ``climate-grid/scenario-v1``); every member is validated against that
    contract and all members must share identical ``times``, ``lats``,
    ``lons`` axes and the same ``data`` elements in the same order.
    ``quantiles`` is ``None`` (default ``[0.05, 0.5, 0.95]``) or a non-empty
    list of finite non-bool numbers in ``[0, 1]``, strictly increasing.

    For every element and every ``[time][lat][lon]`` cell the non-missing
    ``(value, uncertainty)`` samples across the scenarios are combined: with
    no samples the cell gets ``values=None``, ``status="missing"`` and
    ``uncertainty=None``; otherwise ``values`` is the sample mean
    ``Σv / n``, ``status`` is ``"observed"`` when every sample is observed
    and ``"interpolated"`` otherwise, and ``uncertainty`` is
    ``√(Σu²) / n``.  Each requested quantile is computed from the ascending
    sample values by linear interpolation on the position ``h = (n - 1) q``.

    The returned mapping uses the key order ``schema, elements, times, lats,
    lons, quantiles, data``; ``schema`` is ``climate-grid/ensemble-multi-v1``
    and the axes are carried over from the scenarios.  ``data`` follows the
    scenario element order and each item uses the key order ``values,
    status, uncertainty, quantile_values``; the first three members are
    nested ``[time][lat][lon]`` lists and ``quantile_values`` is nested
    ``[quantile][time][lat][lon]`` with ``None`` for sample-less cells.
    Every computed output number is ``round(x, 12)`` with negative zero
    normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    times, lats, lons, elements, validated = _validate_scenarios(scenarios)
    qs = _validate_quantiles(quantiles)

    n_times = len(times)
    n_lat = len(lats)
    n_lon = len(lons)

    out_data: dict[str, dict] = {}
    for element in elements:
        series_list = [member[3][element] for member in validated]
        out_values, out_status, out_uncertainty, out_quantiles = _combine_cells(
            series_list, qs, n_times, n_lat, n_lon
        )
        out_data[element] = {
            "values": out_values,
            "status": out_status,
            "uncertainty": out_uncertainty,
            "quantile_values": out_quantiles,
        }

    return {
        "schema": _ENSEMBLE_MULTI_SCHEMA,
        "elements": list(elements),
        "times": list(times),
        "lats": list(lats),
        "lons": list(lons),
        "quantiles": qs,
        "data": out_data,
    }
