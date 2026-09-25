"""Scenario downscaling of reconstructed daily climate grids."""

from __future__ import annotations

import datetime
import math
import re
from typing import Any

from .regional import (
    _coverage_window_region,
    _entropy_window_region,
    _exceedance_window_region,
    _histogram_window_region,
    _quantile_window_region,
    _validate_quantiles as _regional_validate_quantiles,
    _validate_regions,
    _validate_windows,
    _window_region,
)

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


def _validate_quantiles(quantiles: Any, *, allow_none: bool = True) -> list:
    if quantiles is None:
        if allow_none:
            return list(_DEFAULT_QUANTILES)
        raise TypeError("quantiles must be a list")
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


_SCENARIO_COVERAGE_SCHEMA = "climate-grid/scenario-cov-v1"
_ENSEMBLE_MULTI_KEYS = (
    "schema",
    "elements",
    "times",
    "lats",
    "lons",
    "quantiles",
    "data",
)
_ENSEMBLE_DELTA_MULTI_SCHEMA = "climate-grid/ensemble-delta-multi-v1"
_ENSEMBLE_MULTI_SERIES_KEYS = (
    "values",
    "status",
    "uncertainty",
    "quantile_values",
)
_ENSEMBLE_MULTI_SERIES_KEY_SET = frozenset(_ENSEMBLE_MULTI_SERIES_KEYS)


def _validate_ensemble_multi(
    ensemble: Any,
    *,
    schema: str = _ENSEMBLE_MULTI_SCHEMA,
    where: str = "ensemble",
) -> tuple[list[str], list, list, list, list, dict]:
    if not isinstance(ensemble, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(ensemble.keys()) != _ENSEMBLE_MULTI_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, elements, times, "
            "lats, lons, quantiles, data in order"
        )

    schema_value = ensemble["schema"]
    if not isinstance(schema_value, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema_value != schema:
        raise ValueError(f"{where}.schema must be {schema!r}")

    elements = ensemble["elements"]
    if not isinstance(elements, list):
        raise TypeError(f"{where}.elements must be a list")
    if len(elements) == 0:
        raise ValueError(f"{where}.elements must be non-empty")
    seen_elements: set[str] = set()
    for index, element in enumerate(elements):
        if not isinstance(element, str):
            raise TypeError(f"{where}.elements[{index}] must be a str")
        if element == "":
            raise ValueError(f"{where}.elements[{index}] must be non-empty")
        if element in seen_elements:
            raise ValueError(f"duplicate {where} element: {element!r}")
        seen_elements.add(element)

    times = ensemble["times"]
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

    lats = ensemble["lats"]
    lons = ensemble["lons"]
    _validate_axis(lats, f"{where}.lats")
    _validate_axis(lons, f"{where}.lons")

    quantiles = _validate_quantiles(ensemble["quantiles"], allow_none=False)

    data = ensemble["data"]
    if not isinstance(data, dict):
        raise TypeError(f"{where}.data must be a dict")
    if len(data) == 0:
        raise ValueError(f"{where}.data must be non-empty")
    if list(data.keys()) != elements:
        raise ValueError(
            f"{where}.data keys must match {where}.elements exactly and in order"
        )

    n_times = len(times)
    n_lat = len(lats)
    n_lon = len(lons)
    for element in elements:
        series_where = f"{where}.data[{element!r}]"
        series = data[element]
        if not isinstance(series, dict):
            raise TypeError(f"{series_where} must be a dict")
        if set(series.keys()) != _ENSEMBLE_MULTI_SERIES_KEY_SET:
            raise ValueError(
                f"{series_where} must have exactly the keys values, status, "
                "uncertainty, quantile_values"
            )
        if tuple(series.keys()) != _ENSEMBLE_MULTI_SERIES_KEYS:
            raise ValueError(
                f"{series_where} must have the keys values, status, "
                "uncertainty, quantile_values in order"
            )
        _validate_series(series, n_times, n_lat, n_lon, series_where)
        quantile_values = series["quantile_values"]
        if not isinstance(quantile_values, list):
            raise TypeError(f"{series_where}.quantile_values must be a list")
        if len(quantile_values) != len(quantiles):
            raise ValueError(
                f"{series_where}.quantile_values must have {len(quantiles)} "
                "frames (one per quantile)"
            )
        for q_index, frame in enumerate(quantile_values):
            frame_where = f"{series_where}.quantile_values[{q_index}]"
            if not isinstance(frame, list):
                raise TypeError(f"{frame_where} must be a list")
            if len(frame) != n_times:
                raise ValueError(
                    f"{frame_where} must have {n_times} frames (one per time)"
                )
            for t, row in enumerate(frame):
                row_where = f"{frame_where}[{t}]"
                if not isinstance(row, list):
                    raise TypeError(f"{row_where} must be a list")
                if len(row) != n_lat:
                    raise ValueError(
                        f"{row_where} must have {n_lat} rows (one per lat)"
                    )
                for i, cells in enumerate(row):
                    cells_where = f"{row_where}[{i}]"
                    if not isinstance(cells, list):
                        raise TypeError(f"{cells_where} must be a list")
                    if len(cells) != n_lon:
                        raise ValueError(
                            f"{cells_where} must have {n_lon} cells (one per lon)"
                        )
                    for j, cell in enumerate(cells):
                        target = f"{cells_where}[{j}]"
                        _validate_number(cell, target, nullable=True)
                        if (cell is None) != (
                            series["status"][t][i][j] == "missing"
                        ):
                            raise ValueError(
                                f"{target}: quantile cells must be None exactly "
                                f"where the cell status is missing"
                            )

    return elements, times, lats, lons, quantiles, data


def coverage_multi(
    ensemble, regions, windows, *, min_count: int = 1
) -> dict:
    """Aggregate an ensemble-multi grid into per-window coverage stats.

    ``ensemble`` must be an :func:`ensemble_multi` result (schema
    ``climate-grid/ensemble-multi-v1``) with exactly the keys ``schema,
    elements, times, lats, lons, quantiles, data`` in that order; every
    member is validated against the ensemble-multi contract, including the
    nested ``[quantile][time][lat][lon]`` quantile frames (finite non-bool
    numbers or ``None``).  ``regions``, ``windows`` and ``min_count`` follow
    :func:`climate_grid.regional.aggregate_window` exactly: region
    ``cells`` are checked against the ensemble grid and window
    ``start``/``end`` must occur in ``ensemble.times``.

    For every window (in window order), region (in region order) and
    element (in ensemble element order), the closed calendar interval is
    swept as a grid: ``total`` is the number of days in the inclusive
    interval times the number of cells in the region.  ``available``
    counts non-``missing`` cells, split into ``observed`` and
    ``interpolated``; all four counts are ints.  When ``available`` is
    below ``min_count``, ``rate``, ``observed_rate``,
    ``interpolated_rate`` and ``uncertainty`` are all ``None``.
    Otherwise the three rates are ``available / total``, ``observed /
    total`` and ``interpolated / total`` and ``uncertainty`` is
    ``sqrt(sum(u ** 2)) / available`` over the available cells'
    uncertainties.

    The returned mapping uses the key order ``schema, elements, windows,
    regions, data``; ``schema`` is ``climate-grid/scenario-cov-v1``,
    ``elements`` and ``windows`` echo the arguments (``elements`` taken
    from the ensemble) and ``regions`` lists the region names in input
    order.  ``data`` is a flat list of rows in
    window-then-region-then-element order; each row uses the key order
    ``window, region, element, total, available, observed, interpolated,
    rate, observed_rate, interpolated_rate, uncertainty``.  The four
    counts are ints and every output float is ``round(x, 12)`` with
    negative zero normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    elements, times, lats, lons, _quantiles, data = _validate_ensemble_multi(
        ensemble
    )

    validated_regions = _validate_regions(regions, len(lats), len(lons))
    validated_windows = _validate_windows(windows, times)

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    result_data = []
    for w_name, _start, _end, t_start, t_end in validated_windows:
        for r_name, cells in validated_regions:
            for element in elements:
                series = data[element]
                stats = _coverage_window_region(
                    series["status"],
                    series["uncertainty"],
                    cells,
                    t_start,
                    t_end,
                    min_count,
                )
                result_data.append(
                    {
                        "window": w_name,
                        "region": r_name,
                        "element": element,
                        **stats,
                    }
                )

    return {
        "schema": _SCENARIO_COVERAGE_SCHEMA,
        "elements": list(elements),
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


_SCENARIO_COVERAGE_SUMMARY_SCHEMA = "climate-grid/scenario-cov-summary-v1"
_SCENARIO_COVERAGE_KEYS = (
    "schema",
    "elements",
    "windows",
    "regions",
    "data",
)
_SCENARIO_COVERAGE_ROW_KEYS = (
    "window",
    "region",
    "element",
    "total",
    "available",
    "observed",
    "interpolated",
    "rate",
    "observed_rate",
    "interpolated_rate",
    "uncertainty",
)
_SCENARIO_COVERAGE_SUMMARY_KEYS = (
    "schema",
    "elements",
    "windows",
    "regions",
    "data",
)
_SCENARIO_COVERAGE_SUMMARY_ROW_KEYS = (
    "element",
    "region",
    "total",
    "available",
    "observed",
    "interpolated",
    "rate",
    "observed_rate",
    "interpolated_rate",
    "uncertainty",
)


def _validate_coverage_windows(windows: Any, *, prefix: str = "coverage.windows") -> list:
    if not isinstance(windows, list):
        raise TypeError(f"{prefix} must be a list")
    if len(windows) == 0:
        raise ValueError(f"{prefix} must be non-empty")

    seen_names: set[str] = set()
    for w_index, window in enumerate(windows):
        where = f"{prefix}[{w_index}]"
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
            raise ValueError(f"duplicate coverage window name: {name!r}")
        seen_names.add(name)

        start_day = _parse_date(window["start"], f"{where}.start")
        end_day = _parse_date(window["end"], f"{where}.end")
        if start_day > end_day:
            raise ValueError(f"{where}.start must be on or before {where}.end")

    return windows


def _validate_coverage(
    coverage: Any, *, where: str = "coverage", schema: str = _SCENARIO_COVERAGE_SCHEMA
) -> tuple[list[str], list, list[str], list]:
    if not isinstance(coverage, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(coverage.keys()) != _SCENARIO_COVERAGE_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, elements, windows, "
            "regions, data in order"
        )

    schema_value = coverage["schema"]
    if not isinstance(schema_value, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema_value != schema:
        raise ValueError(f"{where}.schema must be {schema!r}")

    elements = coverage["elements"]
    if not isinstance(elements, list):
        raise TypeError(f"{where}.elements must be a list")
    if len(elements) == 0:
        raise ValueError(f"{where}.elements must be non-empty")
    seen_elements: set[str] = set()
    for index, element in enumerate(elements):
        if not isinstance(element, str):
            raise TypeError(f"{where}.elements[{index}] must be a str")
        if element == "":
            raise ValueError(f"{where}.elements[{index}] must be non-empty")
        if element in seen_elements:
            raise ValueError(f"duplicate {where} element: {element!r}")
        seen_elements.add(element)

    windows = _validate_coverage_windows(
        coverage["windows"], prefix=f"{where}.windows"
    )

    regions = coverage["regions"]
    if not isinstance(regions, list):
        raise TypeError(f"{where}.regions must be a list")
    if len(regions) == 0:
        raise ValueError(f"{where}.regions must be non-empty")
    seen_regions: set[str] = set()
    for index, region in enumerate(regions):
        if not isinstance(region, str):
            raise TypeError(f"{where}.regions[{index}] must be a str")
        if region == "":
            raise ValueError(f"{where}.regions[{index}] must be non-empty")
        if region in seen_regions:
            raise ValueError(f"duplicate {where} region: {region!r}")
        seen_regions.add(region)

    data = coverage["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")

    n_windows = len(windows)
    n_regions = len(regions)
    n_elements = len(elements)
    expected_rows = n_windows * n_regions * n_elements
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows "
            "(one per window/region/element combination, in "
            "window-then-region-then-element order)"
        )

    row_index = 0
    for w_index, window in enumerate(windows):
        for r_index, region in enumerate(regions):
            for e_index, element in enumerate(elements):
                target = f"{where}.data[{row_index}]"
                row = data[row_index]
                if not isinstance(row, dict):
                    raise TypeError(f"{target} must be a dict")
                if tuple(row.keys()) != _SCENARIO_COVERAGE_ROW_KEYS:
                    raise ValueError(
                        f"{target} must have exactly the keys window, region, "
                        "element, total, available, observed, interpolated, "
                        "rate, observed_rate, interpolated_rate, uncertainty "
                        "in order"
                    )

                if not isinstance(row["window"], str):
                    raise TypeError(f"{target}.window must be a str")
                if row["window"] != window["name"]:
                    raise ValueError(
                        f"{target}.window must be {window['name']!r} for its "
                        "window-then-region-then-element position"
                    )
                if not isinstance(row["region"], str):
                    raise TypeError(f"{target}.region must be a str")
                if row["region"] != region:
                    raise ValueError(
                        f"{target}.region must be {region!r} for its "
                        "window-then-region-then-element position"
                    )
                if not isinstance(row["element"], str):
                    raise TypeError(f"{target}.element must be a str")
                if row["element"] != element:
                    raise ValueError(
                        f"{target}.element must be {element!r} for its "
                        "window-then-region-then-element position"
                    )

                for count_name in ("total", "available", "observed", "interpolated"):
                    count = row[count_name]
                    if not isinstance(count, int) or isinstance(count, bool):
                        raise TypeError(f"{target}.{count_name} must be a non-bool int")
                    if count < 0:
                        raise ValueError(f"{target}.{count_name} must be non-negative")
                if row["observed"] + row["interpolated"] != row["available"]:
                    raise ValueError(
                        f"{target}: observed + interpolated must equal available"
                    )
                if row["available"] > row["total"]:
                    raise ValueError(f"{target}: available must not exceed total")

                rate_fields = (
                    "rate",
                    "observed_rate",
                    "interpolated_rate",
                    "uncertainty",
                )
                rates_are_none = [row[name] is None for name in rate_fields]
                if not all(rates_are_none) and any(rates_are_none):
                    raise ValueError(
                        f"{target}: rate, observed_rate, interpolated_rate and "
                        "uncertainty must be all None or all present"
                    )
                if not all(rates_are_none):
                    for name in rate_fields:
                        _validate_number(
                            row[name], f"{target}.{name}", nullable=False
                        )
                    if row["total"] <= 0 or row["available"] <= 0:
                        raise ValueError(
                            f"{target}: rates require positive total and available"
                        )
                    expected = {
                        "rate": row["available"] / row["total"],
                        "observed_rate": row["observed"] / row["total"],
                        "interpolated_rate": row["interpolated"] / row["total"],
                    }
                    for name, value in expected.items():
                        if row[name] != _round_output(value):
                            raise ValueError(
                                f"{target}.{name} must equal the count over total"
                            )
                    if row["uncertainty"] < 0:
                        raise ValueError(f"{target}.uncertainty must be non-negative")

                row_index += 1

    return elements, windows, regions, data


def _coverage_summary_rows(
    elements: list[str], windows: list, regions: list[str], data: list
) -> list:
    n_windows = len(windows)
    n_regions = len(regions)
    n_elements = len(elements)

    result_data = []
    for e_index, element in enumerate(elements):
        for r_index, region in enumerate(regions):
            rows = [
                data[w_index * n_regions * n_elements + r_index * n_elements + e_index]
                for w_index in range(n_windows)
            ]

            total = sum(row["total"] for row in rows)
            available = sum(row["available"] for row in rows)
            observed = sum(row["observed"] for row in rows)
            interpolated = sum(row["interpolated"] for row in rows)

            if total > 0:
                rate = _round_output(available / total)
                observed_rate = _round_output(observed / total)
                interpolated_rate = _round_output(interpolated / total)
            else:
                rate = None
                observed_rate = None
                interpolated_rate = None

            if available == 0 or any(
                row["available"] > 0 and row["uncertainty"] is None for row in rows
            ):
                uncertainty = None
            else:
                weighted_sq_sum = sum(
                    (row["available"] * row["uncertainty"]) ** 2
                    for row in rows
                    if row["available"] > 0
                )
                uncertainty = _round_output(
                    math.sqrt(weighted_sq_sum) / available
                )

            result_data.append(
                {
                    "element": element,
                    "region": region,
                    "total": total,
                    "available": available,
                    "observed": observed,
                    "interpolated": interpolated,
                    "rate": rate,
                    "observed_rate": observed_rate,
                    "interpolated_rate": interpolated_rate,
                    "uncertainty": uncertainty,
                }
            )

    return result_data


def coverage_summary(coverage) -> dict:
    """Summarize a scenario coverage result across all windows.

    ``coverage`` must be a complete :func:`coverage_multi` result (schema
    ``climate-grid/scenario-cov-v1``) with exactly the keys ``schema,
    elements, windows, regions, data`` in that order; every member is
    validated against that contract, including the flat window-then-region-
    then-element ``data`` row order, each row's key order
    ``window, region, element, total, available, observed, interpolated,
    rate, observed_rate, interpolated_rate, uncertainty`` and the count /
    rate invariants.

    Rows are aggregated across all windows, in element order and then region
    order, by summing the four counts.  When the summed ``total`` is
    positive, ``rate``, ``observed_rate`` and ``interpolated_rate`` are the
    summed counts over the summed total; otherwise the three rates are
    ``None``.  The aggregated ``uncertainty`` is ``None`` when the summed
    ``available`` is zero or any contributing row (a row with positive
    ``available``) has an ``uncertainty`` of ``None``; otherwise it is
    ``sqrt(sum(available * uncertainty) ** 2) / sum(available)`` over the
    windows.

    The returned mapping uses the key order ``schema, elements, windows,
    regions, data``; ``schema`` is ``climate-grid/scenario-cov-summary-v1``
    and ``elements``, ``windows`` and ``regions`` echo the coverage arrays.
    Each ``data`` row uses the key order ``element, region, total,
    available, observed, interpolated, rate, observed_rate,
    interpolated_rate, uncertainty``; the counts are ints and every output
    float is ``round(x, 12)`` with negative zero normalized to ``0.0``.
    Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    elements, windows, regions, data = _validate_coverage(coverage)
    result_data = _coverage_summary_rows(elements, windows, regions, data)

    return {
        "schema": _SCENARIO_COVERAGE_SUMMARY_SCHEMA,
        "elements": list(elements),
        "windows": windows,
        "regions": list(regions),
        "data": result_data,
    }


_SCENARIO_COVERAGE_DELTA_SCHEMA = "climate-grid/scenario-cov-delta-v1"
_SCENARIO_COVERAGE_DELTA_ROW_KEYS = (
    "element",
    "region",
    "total_delta",
    "available_delta",
    "observed_delta",
    "interpolated_delta",
    "rate_delta",
    "observed_rate_delta",
    "interpolated_rate_delta",
    "uncertainty",
)


def _validate_coverage_summary(
    summary: Any, where: str
) -> tuple[list[str], list, list[str], list]:
    if not isinstance(summary, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(summary.keys()) != _SCENARIO_COVERAGE_SUMMARY_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, elements, windows, "
            "regions, data in order"
        )

    schema = summary["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _SCENARIO_COVERAGE_SUMMARY_SCHEMA:
        raise ValueError(
            f"{where}.schema must be {_SCENARIO_COVERAGE_SUMMARY_SCHEMA!r}"
        )

    elements = summary["elements"]
    if not isinstance(elements, list):
        raise TypeError(f"{where}.elements must be a list")
    if len(elements) == 0:
        raise ValueError(f"{where}.elements must be non-empty")
    seen_elements: set[str] = set()
    for index, element in enumerate(elements):
        if not isinstance(element, str):
            raise TypeError(f"{where}.elements[{index}] must be a str")
        if element == "":
            raise ValueError(f"{where}.elements[{index}] must be non-empty")
        if element in seen_elements:
            raise ValueError(f"duplicate {where}.elements entry: {element!r}")
        seen_elements.add(element)

    windows = _validate_coverage_windows(
        summary["windows"], prefix=f"{where}.windows"
    )

    regions = summary["regions"]
    if not isinstance(regions, list):
        raise TypeError(f"{where}.regions must be a list")
    if len(regions) == 0:
        raise ValueError(f"{where}.regions must be non-empty")
    seen_regions: set[str] = set()
    for index, region in enumerate(regions):
        if not isinstance(region, str):
            raise TypeError(f"{where}.regions[{index}] must be a str")
        if region == "":
            raise ValueError(f"{where}.regions[{index}] must be non-empty")
        if region in seen_regions:
            raise ValueError(f"duplicate {where}.regions entry: {region!r}")
        seen_regions.add(region)

    data = summary["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")

    n_regions = len(regions)
    n_elements = len(elements)
    expected_rows = n_elements * n_regions
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows "
            "(one per element/region combination, in "
            "element-then-region order)"
        )

    row_index = 0
    for element in elements:
        for region in regions:
            row_where = f"{where}.data[{row_index}]"
            row = data[row_index]
            if not isinstance(row, dict):
                raise TypeError(f"{row_where} must be a dict")
            if tuple(row.keys()) != _SCENARIO_COVERAGE_SUMMARY_ROW_KEYS:
                raise ValueError(
                    f"{row_where} must have exactly the keys element, region, "
                    "total, available, observed, interpolated, rate, "
                    "observed_rate, interpolated_rate, uncertainty in order"
                )

            if not isinstance(row["element"], str):
                raise TypeError(f"{row_where}.element must be a str")
            if row["element"] != element:
                raise ValueError(
                    f"{row_where}.element must be {element!r} for its "
                    "element-then-region position"
                )
            if not isinstance(row["region"], str):
                raise TypeError(f"{row_where}.region must be a str")
            if row["region"] != region:
                raise ValueError(
                    f"{row_where}.region must be {region!r} for its "
                    "element-then-region position"
                )

            for count_name in ("total", "available", "observed", "interpolated"):
                count = row[count_name]
                if not isinstance(count, int) or isinstance(count, bool):
                    raise TypeError(
                        f"{row_where}.{count_name} must be a non-bool int"
                    )
                if count < 0:
                    raise ValueError(
                        f"{row_where}.{count_name} must be non-negative"
                    )
            if row["observed"] + row["interpolated"] != row["available"]:
                raise ValueError(
                    f"{row_where}: observed + interpolated must equal available"
                )
            if row["available"] > row["total"]:
                raise ValueError(f"{row_where}: available must not exceed total")

            rate_fields = ("rate", "observed_rate", "interpolated_rate")
            rates_are_none = [row[name] is None for name in rate_fields]
            if not all(rates_are_none) and any(rates_are_none):
                raise ValueError(
                    f"{row_where}: rate, observed_rate and interpolated_rate "
                    "must be all None or all present"
                )
            if all(rates_are_none):
                if row["total"] > 0:
                    raise ValueError(
                        f"{row_where}: rates must be present when total is "
                        "positive"
                    )
            else:
                if row["total"] <= 0:
                    raise ValueError(
                        f"{row_where}: rates require a positive total"
                    )
                expected = {
                    "rate": row["available"] / row["total"],
                    "observed_rate": row["observed"] / row["total"],
                    "interpolated_rate": row["interpolated"] / row["total"],
                }
                for name, value in expected.items():
                    _validate_number(
                        row[name], f"{row_where}.{name}", nullable=False
                    )
                    if row[name] != _round_output(value):
                        raise ValueError(
                            f"{row_where}.{name} must equal the count over total"
                        )

            uncertainty = row["uncertainty"]
            if uncertainty is not None:
                _validate_number(
                    uncertainty, f"{row_where}.uncertainty", nullable=False
                )
                if uncertainty < 0:
                    raise ValueError(
                        f"{row_where}.uncertainty must be non-negative"
                    )
                if row["available"] == 0:
                    raise ValueError(
                        f"{row_where}.uncertainty must be None when available "
                        "is zero"
                    )

            row_index += 1

    return elements, windows, regions, data


def coverage_delta(left, right) -> dict:
    """Compute the right-minus-left delta of two coverage summaries.

    ``left`` and ``right`` must be complete :func:`coverage_summary` results
    (schema ``climate-grid/scenario-cov-summary-v1``) with exactly the keys
    ``schema, elements, windows, regions, data`` in that order; every member
    is validated against that contract, including the flat
    element-then-region ``data`` row order, each row's key order ``element,
    region, total, available, observed, interpolated, rate, observed_rate,
    interpolated_rate, uncertainty`` and the count / rate invariants.  The
    two summaries must share equal ``elements``, ``windows`` and ``regions``
    in the same order.

    The returned mapping uses the key order ``schema, elements, windows,
    regions, data``; ``schema`` is ``climate-grid/scenario-cov-delta-v1``
    and ``elements``, ``windows`` and ``regions`` echo the (shared) input
    arrays.  ``data`` follows the element-then-region order of the inputs;
    each row uses the key order ``element, region, total_delta,
    available_delta, observed_delta, interpolated_delta, rate_delta,
    observed_rate_delta, interpolated_rate_delta, uncertainty``.  The four
    count deltas are ``right - left`` ints.  Each rate delta is
    ``right - left`` when both rates are not ``None`` and ``None``
    otherwise.  ``uncertainty`` is ``sqrt(u_left ** 2 + u_right ** 2)``
    when both uncertainties are not ``None`` and ``None`` otherwise.
    Every output float is ``round(x, 12)`` with negative zero normalized
    to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    left_elements, left_windows, left_regions, left_data = (
        _validate_coverage_summary(left, "left")
    )
    right_elements, _right_windows, right_regions, right_data = (
        _validate_coverage_summary(right, "right")
    )

    if right_elements != left_elements:
        raise ValueError(
            "left and right must have equal elements in the same order"
        )
    if _right_windows != left_windows:
        raise ValueError(
            "left and right must have equal windows in the same order"
        )
    if right_regions != left_regions:
        raise ValueError(
            "left and right must have equal regions in the same order"
        )

    result_data = []
    for left_row, right_row in zip(left_data, right_data):
        rate_deltas = {}
        for name in ("rate", "observed_rate", "interpolated_rate"):
            left_rate = left_row[name]
            right_rate = right_row[name]
            if left_rate is None or right_rate is None:
                rate_deltas[name] = None
            else:
                rate_deltas[name] = _round_output(right_rate - left_rate)

        left_uncertainty = left_row["uncertainty"]
        right_uncertainty = right_row["uncertainty"]
        if left_uncertainty is None or right_uncertainty is None:
            uncertainty = None
        else:
            uncertainty = _round_output(
                math.sqrt(left_uncertainty ** 2 + right_uncertainty ** 2)
            )

        result_data.append(
            {
                "element": left_row["element"],
                "region": left_row["region"],
                "total_delta": right_row["total"] - left_row["total"],
                "available_delta": right_row["available"] - left_row["available"],
                "observed_delta": right_row["observed"] - left_row["observed"],
                "interpolated_delta": (
                    right_row["interpolated"] - left_row["interpolated"]
                ),
                "rate_delta": rate_deltas["rate"],
                "observed_rate_delta": rate_deltas["observed_rate"],
                "interpolated_rate_delta": rate_deltas["interpolated_rate"],
                "uncertainty": uncertainty,
            }
        )

    return {
        "schema": _SCENARIO_COVERAGE_DELTA_SCHEMA,
        "elements": list(left_elements),
        "windows": left_windows,
        "regions": list(left_regions),
        "data": result_data,
    }


_SCENARIO_COVERAGE_DELTA_MULTI_SCHEMA = "climate-grid/scenario-cov-delta-multi-v1"
_SCENARIO_COVERAGE_DELTA_MULTI_ROW_KEYS = (
    "window",
    "region",
    "element",
    "total_delta",
    "available_delta",
    "observed_delta",
    "interpolated_delta",
    "rate_delta",
    "observed_rate_delta",
    "interpolated_rate_delta",
    "uncertainty",
)


def coverage_delta_multi(left, right) -> dict:
    """Compute the right-minus-left delta of two per-window coverage results.

    ``left`` and ``right`` must be complete :func:`coverage_multi` results
    (schema ``climate-grid/scenario-cov-v1``) with exactly the keys
    ``schema, elements, windows, regions, data`` in that order; every member
    is validated against that contract, including the flat
    window-then-region-then-element ``data`` row order, each row's key order
    ``window, region, element, total, available, observed, interpolated,
    rate, observed_rate, interpolated_rate, uncertainty`` and the count /
    rate invariants.  The two results must share equal ``elements``,
    ``windows`` and ``regions`` in the same order.

    The returned mapping uses the key order ``schema, elements, windows,
    regions, data``; ``schema`` is
    ``climate-grid/scenario-cov-delta-multi-v1`` and ``elements``,
    ``windows`` and ``regions`` echo the (shared) input arrays.  ``data``
    follows the window-then-region-then-element order of the inputs; each
    row uses the key order ``window, region, element, total_delta,
    available_delta, observed_delta, interpolated_delta, rate_delta,
    observed_rate_delta, interpolated_rate_delta, uncertainty``.  The four
    count deltas are ``right - left`` ints.  Each rate delta is
    ``right - left`` when both rates are not ``None`` and ``None``
    otherwise.  ``uncertainty`` is ``sqrt(u_left ** 2 + u_right ** 2)``
    when both uncertainties are not ``None`` and ``None`` otherwise.
    Every output float is ``round(x, 12)`` with negative zero normalized
    to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    left_elements, left_windows, left_regions, left_data = _validate_coverage(
        left, where="left"
    )
    right_elements, right_windows, right_regions, right_data = (
        _validate_coverage(right, where="right")
    )

    if right_elements != left_elements:
        raise ValueError(
            "left and right must have equal elements in the same order"
        )
    if right_windows != left_windows:
        raise ValueError(
            "left and right must have equal windows in the same order"
        )
    if right_regions != left_regions:
        raise ValueError(
            "left and right must have equal regions in the same order"
        )

    result_data = []
    for left_row, right_row in zip(left_data, right_data):
        rate_deltas = {}
        for name in ("rate", "observed_rate", "interpolated_rate"):
            left_rate = left_row[name]
            right_rate = right_row[name]
            if left_rate is None or right_rate is None:
                rate_deltas[name] = None
            else:
                rate_deltas[name] = _round_output(right_rate - left_rate)

        left_uncertainty = left_row["uncertainty"]
        right_uncertainty = right_row["uncertainty"]
        if left_uncertainty is None or right_uncertainty is None:
            uncertainty = None
        else:
            uncertainty = _round_output(
                math.sqrt(left_uncertainty ** 2 + right_uncertainty ** 2)
            )

        result_data.append(
            {
                "window": left_row["window"],
                "region": left_row["region"],
                "element": left_row["element"],
                "total_delta": right_row["total"] - left_row["total"],
                "available_delta": right_row["available"] - left_row["available"],
                "observed_delta": right_row["observed"] - left_row["observed"],
                "interpolated_delta": (
                    right_row["interpolated"] - left_row["interpolated"]
                ),
                "rate_delta": rate_deltas["rate"],
                "observed_rate_delta": rate_deltas["observed_rate"],
                "interpolated_rate_delta": rate_deltas["interpolated_rate"],
                "uncertainty": uncertainty,
            }
        )

    return {
        "schema": _SCENARIO_COVERAGE_DELTA_MULTI_SCHEMA,
        "elements": list(left_elements),
        "windows": left_windows,
        "regions": list(left_regions),
        "data": result_data,
    }


_SCENARIO_COVERAGE_DELTA_SUMMARY_MULTI_SCHEMA = (
    "climate-grid/scenario-cov-delta-summary-multi-v1"
)
_SCENARIO_COVERAGE_DELTA_MULTI_KEYS = (
    "schema",
    "elements",
    "windows",
    "regions",
    "data",
)
_SCENARIO_COVERAGE_DELTA_SUMMARY_MULTI_ROW_KEYS = (
    "element",
    "region",
    "total_delta",
    "available_delta",
    "observed_delta",
    "interpolated_delta",
    "rate_delta",
    "observed_rate_delta",
    "interpolated_rate_delta",
    "uncertainty",
)


def _validate_coverage_delta_multi(
    delta: Any, *, where: str = "delta"
) -> tuple[list[str], list, list[str], list]:
    if not isinstance(delta, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(delta.keys()) != _SCENARIO_COVERAGE_DELTA_MULTI_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, elements, windows, "
            "regions, data in order"
        )

    schema = delta["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _SCENARIO_COVERAGE_DELTA_MULTI_SCHEMA:
        raise ValueError(
            f"{where}.schema must be {_SCENARIO_COVERAGE_DELTA_MULTI_SCHEMA!r}"
        )

    elements = delta["elements"]
    if not isinstance(elements, list):
        raise TypeError(f"{where}.elements must be a list")
    if len(elements) == 0:
        raise ValueError(f"{where}.elements must be non-empty")
    seen_elements: set[str] = set()
    for index, element in enumerate(elements):
        if not isinstance(element, str):
            raise TypeError(f"{where}.elements[{index}] must be a str")
        if element == "":
            raise ValueError(f"{where}.elements[{index}] must be non-empty")
        if element in seen_elements:
            raise ValueError(f"duplicate {where}.elements entry: {element!r}")
        seen_elements.add(element)

    windows = _validate_coverage_windows(
        delta["windows"], prefix=f"{where}.windows"
    )

    regions = delta["regions"]
    if not isinstance(regions, list):
        raise TypeError(f"{where}.regions must be a list")
    if len(regions) == 0:
        raise ValueError(f"{where}.regions must be non-empty")
    seen_regions: set[str] = set()
    for index, region in enumerate(regions):
        if not isinstance(region, str):
            raise TypeError(f"{where}.regions[{index}] must be a str")
        if region == "":
            raise ValueError(f"{where}.regions[{index}] must be non-empty")
        if region in seen_regions:
            raise ValueError(f"duplicate {where}.regions entry: {region!r}")
        seen_regions.add(region)

    data = delta["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")

    n_windows = len(windows)
    n_regions = len(regions)
    n_elements = len(elements)
    expected_rows = n_windows * n_regions * n_elements
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows "
            "(one per window/region/element combination, in "
            "window-then-region-then-element order)"
        )

    row_index = 0
    for window in windows:
        for region in regions:
            for element in elements:
                target = f"{where}.data[{row_index}]"
                row = data[row_index]
                if not isinstance(row, dict):
                    raise TypeError(f"{target} must be a dict")
                if tuple(row.keys()) != _SCENARIO_COVERAGE_DELTA_MULTI_ROW_KEYS:
                    raise ValueError(
                        f"{target} must have exactly the keys window, region, "
                        "element, total_delta, available_delta, observed_delta, "
                        "interpolated_delta, rate_delta, observed_rate_delta, "
                        "interpolated_rate_delta, uncertainty in order"
                    )

                if not isinstance(row["window"], str):
                    raise TypeError(f"{target}.window must be a str")
                if row["window"] != window["name"]:
                    raise ValueError(
                        f"{target}.window must be {window['name']!r} for its "
                        "window-then-region-then-element position"
                    )
                if not isinstance(row["region"], str):
                    raise TypeError(f"{target}.region must be a str")
                if row["region"] != region:
                    raise ValueError(
                        f"{target}.region must be {region!r} for its "
                        "window-then-region-then-element position"
                    )
                if not isinstance(row["element"], str):
                    raise TypeError(f"{target}.element must be a str")
                if row["element"] != element:
                    raise ValueError(
                        f"{target}.element must be {element!r} for its "
                        "window-then-region-then-element position"
                    )

                for count_name in (
                    "total_delta",
                    "available_delta",
                    "observed_delta",
                    "interpolated_delta",
                ):
                    count = row[count_name]
                    if not isinstance(count, int) or isinstance(count, bool):
                        raise TypeError(
                            f"{target}.{count_name} must be a non-bool int"
                        )

                for name in (
                    "rate_delta",
                    "observed_rate_delta",
                    "interpolated_rate_delta",
                    "uncertainty",
                ):
                    _validate_number(row[name], f"{target}.{name}", nullable=True)

                row_index += 1

    return elements, windows, regions, data


def coverage_delta_multi_summary(delta) -> dict:
    """Summarize a per-window coverage delta result across all windows.

    ``delta`` must be a complete :func:`coverage_delta_multi` result (schema
    ``climate-grid/scenario-cov-delta-multi-v1``) with exactly the keys
    ``schema, elements, windows, regions, data`` in that order; every member
    is validated against that contract, including the flat
    window-then-region-then-element ``data`` row order, each row's key order
    ``window, region, element, total_delta, available_delta, observed_delta,
    interpolated_delta, rate_delta, observed_rate_delta,
    interpolated_rate_delta, uncertainty`` and the item types (the four
    count deltas are non-bool ints; the rate deltas and ``uncertainty`` are
    finite non-bool numbers or ``None``).

    Rows are aggregated across all windows, in element order and then region
    order.  The four count deltas are summed.  Each rate delta is the sum of
    the per-window rate deltas when every window's value is not ``None``
    and ``None`` otherwise.  ``uncertainty`` is ``sqrt(sum(u ** 2))`` over
    the windows when every window's uncertainty is not ``None`` and
    ``None`` otherwise.

    The returned mapping uses the key order ``schema, elements, windows,
    regions, data``; ``schema`` is
    ``climate-grid/scenario-cov-delta-summary-multi-v1`` and ``elements``,
    ``windows`` and ``regions`` echo the delta arrays.  Each ``data`` row
    uses the key order ``element, region, total_delta, available_delta,
    observed_delta, interpolated_delta, rate_delta, observed_rate_delta,
    interpolated_rate_delta, uncertainty``; the count deltas are ints and
    every output float is ``round(x, 12)`` with negative zero normalized to
    ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    elements, windows, regions, data = _validate_coverage_delta_multi(delta)

    n_windows = len(windows)
    n_regions = len(regions)
    n_elements = len(elements)

    result_data = []
    for e_index, element in enumerate(elements):
        for r_index, region in enumerate(regions):
            rows = [
                data[w_index * n_regions * n_elements + r_index * n_elements + e_index]
                for w_index in range(n_windows)
            ]

            rate_deltas = {}
            for name in (
                "rate_delta",
                "observed_rate_delta",
                "interpolated_rate_delta",
            ):
                if any(row[name] is None for row in rows):
                    rate_deltas[name] = None
                else:
                    rate_deltas[name] = _round_output(
                        sum(row[name] for row in rows)
                    )

            if any(row["uncertainty"] is None for row in rows):
                uncertainty = None
            else:
                uncertainty = _round_output(
                    math.sqrt(sum(row["uncertainty"] ** 2 for row in rows))
                )

            result_data.append(
                {
                    "element": element,
                    "region": region,
                    "total_delta": sum(row["total_delta"] for row in rows),
                    "available_delta": sum(row["available_delta"] for row in rows),
                    "observed_delta": sum(row["observed_delta"] for row in rows),
                    "interpolated_delta": sum(
                        row["interpolated_delta"] for row in rows
                    ),
                    "rate_delta": rate_deltas["rate_delta"],
                    "observed_rate_delta": rate_deltas["observed_rate_delta"],
                    "interpolated_rate_delta": rate_deltas[
                        "interpolated_rate_delta"
                    ],
                    "uncertainty": uncertainty,
                }
            )

    return {
        "schema": _SCENARIO_COVERAGE_DELTA_SUMMARY_MULTI_SCHEMA,
        "elements": list(elements),
        "windows": windows,
        "regions": list(regions),
        "data": result_data,
    }


def value_delta_multi(left, right) -> dict:
    """Compute the right-minus-left value delta of two ensemble-multi grids.

    ``left`` and ``right`` must be complete :func:`ensemble_multi` results
    (schema ``climate-grid/ensemble-multi-v1``) with exactly the keys
    ``schema, elements, times, lats, lons, quantiles, data`` in that order;
    every member is validated against the ensemble-multi contract, including
    each item's key order ``values, status, uncertainty, quantile_values``
    and the nested ``[quantile][time][lat][lon]`` quantile frames.  The two
    inputs must share equal ``elements``, ``times``, ``lats``, ``lons`` and
    ``quantiles`` arrays in the same order.

    For every element and every ``[time][lat][lon]`` cell whose status is
    ``missing`` in either input, the output cell gets ``values=None``,
    ``status="missing"``, ``uncertainty=None`` and ``None`` for every
    quantile.  Otherwise ``values`` is ``right - left``, ``status`` is
    ``"observed"`` only when both input statuses are observed and
    ``"interpolated"`` otherwise, and ``uncertainty`` is
    ``sqrt(u_left ** 2 + u_right ** 2)``.  Each quantile value is
    ``right - left`` when both input quantile cells are not ``None`` and
    ``None`` otherwise.

    The returned mapping uses the key order ``schema, elements, times, lats,
    lons, quantiles, data``; ``schema`` is
    ``climate-grid/ensemble-delta-multi-v1`` and the elements, axes and
    quantile labels are carried over (echoed) from the inputs.  ``data``
    follows the shared element order and each item uses the key order
    ``values, status, uncertainty, quantile_values``; the first three
    members are nested ``[time][lat][lon]`` lists and ``quantile_values`` is
    nested ``[quantile][time][lat][lon]``.  Every computed output number is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs are
    not modified.

    Raises ``TypeError`` for wrong container/item types and ``ValueError``
    for any other contract violation (schema, key order, shape or ordering).
    """
    left_elements, left_times, left_lats, left_lons, left_quantiles, left_data = (
        _validate_ensemble_multi(left, where="left")
    )
    (
        right_elements,
        right_times,
        right_lats,
        right_lons,
        right_quantiles,
        right_data,
    ) = _validate_ensemble_multi(right, where="right")

    if right_elements != left_elements:
        raise ValueError(
            "left and right must have equal elements in the same order"
        )
    if right_times != left_times:
        raise ValueError(
            "left and right must have equal times in the same order"
        )
    if right_lats != left_lats:
        raise ValueError(
            "left and right must have equal lats in the same order"
        )
    if right_lons != left_lons:
        raise ValueError(
            "left and right must have equal lons in the same order"
        )
    if right_quantiles != left_quantiles:
        raise ValueError(
            "left and right must have equal quantiles in the same order"
        )

    n_quantiles = len(left_quantiles)
    n_times = len(left_times)
    n_lat = len(left_lats)
    n_lon = len(left_lons)

    out_data: dict[str, dict] = {}
    for element in left_elements:
        left_series = left_data[element]
        right_series = right_data[element]
        left_status = left_series["status"]
        right_status = right_series["status"]
        left_values = left_series["values"]
        right_values = right_series["values"]
        left_uncertainty = left_series["uncertainty"]
        right_uncertainty = right_series["uncertainty"]
        left_quantile_values = left_series["quantile_values"]
        right_quantile_values = right_series["quantile_values"]

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
            for _ in range(n_quantiles)
        ]

        for t in range(n_times):
            for i in range(n_lat):
                for j in range(n_lon):
                    if (
                        left_status[t][i][j] == "missing"
                        or right_status[t][i][j] == "missing"
                    ):
                        out_status[t][i][j] = "missing"
                        continue
                    out_values[t][i][j] = _round_output(
                        right_values[t][i][j] - left_values[t][i][j]
                    )
                    out_status[t][i][j] = (
                        "observed"
                        if left_status[t][i][j] == "observed"
                        and right_status[t][i][j] == "observed"
                        else "interpolated"
                    )
                    out_uncertainty[t][i][j] = _round_output(
                        math.sqrt(
                            left_uncertainty[t][i][j] ** 2
                            + right_uncertainty[t][i][j] ** 2
                        )
                    )
                    for q_index in range(n_quantiles):
                        left_q = left_quantile_values[q_index][t][i][j]
                        right_q = right_quantile_values[q_index][t][i][j]
                        if left_q is not None and right_q is not None:
                            out_quantiles[q_index][t][i][j] = _round_output(
                                right_q - left_q
                            )

        out_data[element] = {
            "values": out_values,
            "status": out_status,
            "uncertainty": out_uncertainty,
            "quantile_values": out_quantiles,
        }

    return {
        "schema": _ENSEMBLE_DELTA_MULTI_SCHEMA,
        "elements": list(left_elements),
        "times": list(left_times),
        "lats": list(left_lats),
        "lons": list(left_lons),
        "quantiles": list(left_quantiles),
        "data": out_data,
    }


_ENSEMBLE_DELTA_REGION_SCHEMA = "climate-grid/ensemble-delta-region-v1"


def aggregate_value_delta_multi(delta, regions, windows, *, min_count: int = 1) -> dict:
    """Aggregate an ensemble-delta-multi grid into per-window region stats.

    ``delta`` must be a complete :func:`value_delta_multi` result (schema
    ``climate-grid/ensemble-delta-multi-v1``) with exactly the keys
    ``schema, elements, times, lats, lons, quantiles, data`` in that order;
    every member is validated against that contract, including each item's
    key order ``values, status, uncertainty, quantile_values`` and the
    nested ``[quantile][time][lat][lon]`` quantile frames.  ``regions``,
    ``windows`` and ``min_count`` follow
    :func:`climate_grid.regional.aggregate_window` exactly: region
    ``cells`` are checked against the delta grid and window
    ``start``/``end`` must occur in ``delta.times``.

    For every window (in window order), region (in region order) and
    element (in delta element order), the non-missing ``(value,
    uncertainty)`` samples of the closed calendar interval are collected
    across the region's cells; ``count`` is the sample count ``n``.  When
    ``n`` is below ``min_count``, ``mean``, ``min``, ``max`` and
    ``uncertainty`` are all ``None``; otherwise they are the sample mean
    ``Σv / n``, the sample minimum, the sample maximum and
    ``sqrt(Σu²) / n`` over the sampled uncertainties.

    The returned mapping uses the key order ``schema, elements, windows,
    regions, data``; ``schema`` is ``climate-grid/ensemble-delta-region-v1``,
    ``elements`` and ``windows`` echo the delta elements and the ``windows``
    argument, and ``regions`` lists the region names in input order.
    ``data`` is a flat list of rows in window-then-region-then-element
    order; each row uses the key order ``window, region, element, count,
    mean, min, max, uncertainty``.  ``count`` is an int and every output
    float is ``round(x, 12)`` with negative zero normalized to ``0.0``.
    Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    elements, times, lats, lons, _quantiles, data = _validate_ensemble_multi(
        delta, schema=_ENSEMBLE_DELTA_MULTI_SCHEMA, where="delta"
    )

    validated_regions = _validate_regions(regions, len(lats), len(lons))
    validated_windows = _validate_windows(windows, times)

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    result_data = []
    for w_name, _start, _end, t_start, t_end in validated_windows:
        for r_name, cells in validated_regions:
            for element in elements:
                series = data[element]
                stats = _window_region(
                    series["values"],
                    series["status"],
                    series["uncertainty"],
                    cells,
                    t_start,
                    t_end,
                    min_count,
                )
                result_data.append(
                    {
                        "window": w_name,
                        "region": r_name,
                        "element": element,
                        **stats,
                    }
                )

    return {
        "schema": _ENSEMBLE_DELTA_REGION_SCHEMA,
        "elements": list(elements),
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


_DELTA_QUANTILE_REGION_SCHEMA = "climate-grid/dq-region-v1"


def aggregate_value_delta_quantile(
    delta, regions, windows, quantiles, *, min_count: int = 1
) -> dict:
    """Aggregate an ensemble-delta-multi grid into per-window region quantiles.

    ``delta`` must be a complete :func:`value_delta_multi` result (schema
    ``climate-grid/ensemble-delta-multi-v1``) with exactly the keys
    ``schema, elements, times, lats, lons, quantiles, data`` in that order;
    every member is validated against that contract, including each item's
    key order ``values, status, uncertainty, quantile_values`` and the
    nested ``[quantile][time][lat][lon]`` quantile frames.  ``regions``,
    ``windows`` and ``min_count`` follow
    :func:`climate_grid.regional.aggregate_window` exactly: region
    ``cells`` are checked against the delta grid and window
    ``start``/``end`` must occur in ``delta.times``.  ``quantiles`` follows
    :func:`climate_grid.regional.aggregate_quantile` exactly: a non-empty
    list of finite non-bool numbers with ``0 <= q <= 1`` in strictly
    increasing order.

    For every window (in window order), region (in region order) and
    element (in delta element order), the non-missing ``(value,
    uncertainty)`` samples of the closed calendar interval are collected
    across the region's cells; ``count`` is the sample count ``n``.  When
    ``n`` is below ``min_count``, ``quantiles`` is a list of ``None`` of
    the same length as the argument and ``uncertainty`` is ``None``.
    Otherwise the samples are sorted ascending and each quantile ``q`` is
    computed as ``h = (n - 1) * q``, ``a = floor(h)``, ``b = ceil(h)``,
    ``v[a] + (h - a) * (v[b] - v[a])``; ``uncertainty`` is
    ``sqrt(sum(u ** 2)) / n`` over the sampled uncertainties.

    The returned mapping uses the key order ``schema, elements, quantiles,
    windows, regions, data``; ``schema`` is ``climate-grid/dq-region-v1``,
    ``elements``, ``quantiles`` and ``windows`` echo the delta elements and
    the arguments, and ``regions`` lists the region names in input order.
    ``data`` is a flat list of rows in window-then-region-then-element
    order; each row uses the key order ``window, region, element, count,
    quantiles, uncertainty``.  ``count`` is an int and every output float
    is ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs
    are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    elements, times, lats, lons, _delta_quantiles, data = _validate_ensemble_multi(
        delta, schema=_ENSEMBLE_DELTA_MULTI_SCHEMA, where="delta"
    )

    validated_regions = _validate_regions(regions, len(lats), len(lons))
    validated_windows = _validate_windows(windows, times)
    validated_quantiles = _regional_validate_quantiles(quantiles)

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    result_data = []
    for w_name, _start, _end, t_start, t_end in validated_windows:
        for r_name, cells in validated_regions:
            for element in elements:
                series = data[element]
                stats = _quantile_window_region(
                    series["values"],
                    series["status"],
                    series["uncertainty"],
                    cells,
                    t_start,
                    t_end,
                    validated_quantiles,
                    min_count,
                )
                result_data.append(
                    {
                        "window": w_name,
                        "region": r_name,
                        "element": element,
                        **stats,
                    }
                )

    return {
        "schema": _DELTA_QUANTILE_REGION_SCHEMA,
        "elements": list(elements),
        "quantiles": quantiles,
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


_ENSEMBLE_DELTA_COVERAGE_SCHEMA = "climate-grid/ensemble-delta-coverage-v1"


def aggregate_value_delta_coverage(
    delta, regions, windows, *, min_count: int = 1
) -> dict:
    """Aggregate an ensemble-delta-multi grid into per-window coverage stats.

    ``delta`` must be a complete :func:`value_delta_multi` result (schema
    ``climate-grid/ensemble-delta-multi-v1``) with exactly the keys
    ``schema, elements, times, lats, lons, quantiles, data`` in that order;
    every member is validated against that contract, including each item's
    key order ``values, status, uncertainty, quantile_values`` and the
    nested ``[quantile][time][lat][lon]`` quantile frames.  ``regions``,
    ``windows`` and ``min_count`` follow
    :func:`climate_grid.regional.aggregate_window` exactly: region
    ``cells`` are checked against the delta grid and window
    ``start``/``end`` must occur in ``delta.times``.

    For every window (in window order), region (in region order) and
    element (in delta element order), the closed calendar interval is
    swept as a grid: ``total`` is the number of days in the inclusive
    interval times the number of cells in the region.  ``available``
    counts non-``missing`` cells, split into ``observed`` and
    ``interpolated``; all four counts are ints.  When ``available`` is
    below ``min_count``, ``rate``, ``observed_rate``,
    ``interpolated_rate`` and ``uncertainty`` are all ``None``.
    Otherwise the three rates are ``available / total``, ``observed /
    total`` and ``interpolated / total`` and ``uncertainty`` is
    ``sqrt(sum(u ** 2)) / available`` over the available cells'
    uncertainties.

    The returned mapping uses the key order ``schema, elements, windows,
    regions, data``; ``schema`` is
    ``climate-grid/ensemble-delta-coverage-v1``, ``elements`` and
    ``windows`` echo the delta elements and the ``windows`` argument, and
    ``regions`` lists the region names in input order.  ``data`` is a
    flat list of rows in window-then-region-then-element order; each row
    uses the key order ``window, region, element, total, available,
    observed, interpolated, rate, observed_rate, interpolated_rate,
    uncertainty``.  The four counts are ints and every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs
    are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    elements, times, lats, lons, _quantiles, data = _validate_ensemble_multi(
        delta, schema=_ENSEMBLE_DELTA_MULTI_SCHEMA, where="delta"
    )

    validated_regions = _validate_regions(regions, len(lats), len(lons))
    validated_windows = _validate_windows(windows, times)

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    result_data = []
    for w_name, _start, _end, t_start, t_end in validated_windows:
        for r_name, cells in validated_regions:
            for element in elements:
                series = data[element]
                stats = _coverage_window_region(
                    series["status"],
                    series["uncertainty"],
                    cells,
                    t_start,
                    t_end,
                    min_count,
                )
                result_data.append(
                    {
                        "window": w_name,
                        "region": r_name,
                        "element": element,
                        **stats,
                    }
                )

    return {
        "schema": _ENSEMBLE_DELTA_COVERAGE_SCHEMA,
        "elements": list(elements),
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


_ENSEMBLE_DELTA_COVERAGE_SUMMARY_SCHEMA = (
    "climate-grid/ensemble-delta-coverage-summary-v1"
)


def aggregate_value_delta_coverage_summary(coverage) -> dict:
    """Summarize an ensemble-delta coverage result across all windows.

    ``coverage`` must be a complete :func:`aggregate_value_delta_coverage`
    result (schema ``climate-grid/ensemble-delta-coverage-v1``) with exactly
    the keys ``schema, elements, windows, regions, data`` in that order;
    every member is validated against that contract, including the flat
    window-then-region-then-element ``data`` row order, each row's key order
    ``window, region, element, total, available, observed, interpolated,
    rate, observed_rate, interpolated_rate, uncertainty`` and the count /
    rate invariants.

    Rows are aggregated across all windows, in element order and then region
    order, by summing the four counts.  When the summed ``total`` is
    positive, ``rate``, ``observed_rate`` and ``interpolated_rate`` are the
    summed counts over the summed total; otherwise the three rates are
    ``None``.  The aggregated ``uncertainty`` is ``None`` when the summed
    ``available`` is zero or any contributing row (a row with positive
    ``available``) has an ``uncertainty`` of ``None``; otherwise it is
    ``sqrt(sum(available * uncertainty) ** 2) / sum(available)`` over the
    windows.

    The returned mapping uses the key order ``schema, elements, windows,
    regions, data``; ``schema`` is
    ``climate-grid/ensemble-delta-coverage-summary-v1`` and ``elements``,
    ``windows`` and ``regions`` echo the coverage arrays.  Each ``data`` row
    uses the key order ``element, region, total, available, observed,
    interpolated, rate, observed_rate, interpolated_rate, uncertainty``;
    the counts are ints and every output float is ``round(x, 12)`` with
    negative zero normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    elements, windows, regions, data = _validate_coverage(
        coverage, schema=_ENSEMBLE_DELTA_COVERAGE_SCHEMA
    )
    result_data = _coverage_summary_rows(elements, windows, regions, data)

    return {
        "schema": _ENSEMBLE_DELTA_COVERAGE_SUMMARY_SCHEMA,
        "elements": list(elements),
        "windows": windows,
        "regions": list(regions),
        "data": result_data,
    }


_ENSEMBLE_DELTA_HISTOGRAM_REGION_SCHEMA = (
    "climate-grid/ensemble-delta-histogram-region-v1"
)


def aggregate_value_delta_histogram(
    delta, regions, windows, edges, *, min_count: int = 1
) -> dict:
    """Aggregate an ensemble-delta-multi grid into per-window value histograms.

    ``delta`` must be a complete :func:`value_delta_multi` result (schema
    ``climate-grid/ensemble-delta-multi-v1``) with exactly the keys
    ``schema, elements, times, lats, lons, quantiles, data`` in that order;
    every member is validated against that contract, including each item's
    key order ``values, status, uncertainty, quantile_values`` and the
    nested ``[quantile][time][lat][lon]`` quantile frames.  ``regions``,
    ``windows`` and ``min_count`` follow
    :func:`climate_grid.regional.aggregate_window` exactly: region
    ``cells`` are checked against the delta grid and window
    ``start``/``end`` must occur in ``delta.times``.  ``edges`` is a
    non-empty list of finite non-bool int or float bin edges in strictly
    increasing order; a wrong container or item type raises ``TypeError``
    while an empty, non-finite or non-increasing list raises
    ``ValueError``.

    For every window (in window order), region (in region order) and
    element (in delta element order), the non-missing ``(value,
    uncertainty)`` samples of the closed calendar interval are collected
    across the region's cells.  With ``B = len(edges) + 1`` bins, a value
    is counted in the underflow bin when ``v < edges[0]``, in the overflow
    bin when ``v >= edges[-1]``, and otherwise in the bin
    ``edges[k - 1] <= v < edges[k]``; a value equal to an edge falls into
    the bin on its right.  ``count`` is the sample count ``n`` and
    ``bin_counts`` lists the ``B`` int bin counts.  When ``n`` is below
    ``min_count``, ``rates`` is a list of ``None`` of length ``B`` and
    ``uncertainty`` is ``None``.  Otherwise ``rates[i]`` is
    ``bin_counts[i] / n`` and ``uncertainty`` is ``sqrt(sum(u ** 2)) / n``
    over the sampled uncertainties.

    The returned mapping uses the key order ``schema, elements, edges,
    windows, regions, data``; ``schema`` is
    ``climate-grid/ensemble-delta-histogram-region-v1``, ``elements``,
    ``edges`` and ``windows`` echo the delta elements and the arguments in
    input order, and ``regions`` lists the region names in input order.
    ``data`` is a flat list of rows in window-then-region-then-element
    order; each row uses the key order ``window, region, element, count,
    bin_counts, rates, uncertainty``.  ``count`` and the members of
    ``bin_counts`` are ints and every output float is ``round(x, 12)``
    with negative zero normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    elements, times, lats, lons, _quantiles, data = _validate_ensemble_multi(
        delta, schema=_ENSEMBLE_DELTA_MULTI_SCHEMA, where="delta"
    )

    validated_regions = _validate_regions(regions, len(lats), len(lons))
    validated_windows = _validate_windows(windows, times)
    _validate_axis(edges, "edges")

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    result_data = []
    for w_name, _start, _end, t_start, t_end in validated_windows:
        for r_name, cells in validated_regions:
            for element in elements:
                series = data[element]
                stats = _histogram_window_region(
                    series["values"],
                    series["status"],
                    series["uncertainty"],
                    cells,
                    t_start,
                    t_end,
                    edges,
                    min_count,
                )
                result_data.append(
                    {
                        "window": w_name,
                        "region": r_name,
                        "element": element,
                        **stats,
                    }
                )

    return {
        "schema": _ENSEMBLE_DELTA_HISTOGRAM_REGION_SCHEMA,
        "elements": list(elements),
        "edges": edges,
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


_ENSEMBLE_DELTA_ENTROPY_REGION_SCHEMA = (
    "climate-grid/ensemble-delta-entropy-region-v1"
)


def aggregate_value_delta_entropy(
    delta, regions, windows, edges, *, min_count: int = 1
) -> dict:
    """Aggregate an ensemble-delta-multi grid into per-window value entropy.

    ``delta`` must be a complete :func:`value_delta_multi` result (schema
    ``climate-grid/ensemble-delta-multi-v1``) with exactly the keys
    ``schema, elements, times, lats, lons, quantiles, data`` in that order;
    every member is validated against that contract, including each item's
    key order ``values, status, uncertainty, quantile_values`` and the
    nested ``[quantile][time][lat][lon]`` quantile frames.  ``regions``,
    ``windows`` and ``min_count`` follow
    :func:`climate_grid.regional.aggregate_window` exactly: region
    ``cells`` are checked against the delta grid and window
    ``start``/``end`` must occur in ``delta.times``.  ``edges`` is a
    non-empty list of finite non-bool int or float bin edges in strictly
    increasing order; a wrong container or item type raises ``TypeError``
    while an empty, non-finite or non-increasing list raises
    ``ValueError``.

    For every window (in window order), region (in region order) and
    element (in delta element order), the non-missing ``(value,
    uncertainty)`` samples of the closed calendar interval are collected
    across the region's cells.  With ``B = len(edges) + 1`` bins, the
    samples are binned exactly as in
    :func:`climate_grid.regional.aggregate_histogram`; ``count`` is the
    sample count ``n`` and ``bin_counts`` lists the ``B`` int bin counts.
    When ``n`` is below ``min_count``, ``entropy`` and ``uncertainty`` are
    both ``None``.  Otherwise, with ``p = bin_counts[i] / n``, ``entropy``
    is ``-sum(p * log2(p))`` over the bins with ``p > 0`` and
    ``uncertainty`` is ``sqrt(sum(u ** 2)) / n`` over the sampled
    uncertainties.

    The returned mapping uses the key order ``schema, elements, edges,
    windows, regions, data``; ``schema`` is
    ``climate-grid/ensemble-delta-entropy-region-v1``, ``elements``,
    ``edges`` and ``windows`` echo the delta elements and the arguments in
    input order, and ``regions`` lists the region names in input order.
    ``data`` is a flat list of rows in window-then-region-then-element
    order; each row uses the key order ``window, region, element, count,
    bin_counts, entropy, uncertainty``.  ``count`` and the members of
    ``bin_counts`` are ints and every output float is ``round(x, 12)``
    with negative zero normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    elements, times, lats, lons, _quantiles, data = _validate_ensemble_multi(
        delta, schema=_ENSEMBLE_DELTA_MULTI_SCHEMA, where="delta"
    )

    validated_regions = _validate_regions(regions, len(lats), len(lons))
    validated_windows = _validate_windows(windows, times)
    _validate_axis(edges, "edges")

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    result_data = []
    for w_name, _start, _end, t_start, t_end in validated_windows:
        for r_name, cells in validated_regions:
            for element in elements:
                series = data[element]
                stats = _entropy_window_region(
                    series["values"],
                    series["status"],
                    series["uncertainty"],
                    cells,
                    t_start,
                    t_end,
                    edges,
                    min_count,
                )
                result_data.append(
                    {
                        "window": w_name,
                        "region": r_name,
                        "element": element,
                        **stats,
                    }
                )

    return {
        "schema": _ENSEMBLE_DELTA_ENTROPY_REGION_SCHEMA,
        "elements": list(elements),
        "edges": edges,
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


_ENSEMBLE_DELTA_EXCEEDANCE_SCHEMA = "climate-grid/ensemble-delta-exceedance-v1"


def aggregate_value_delta_exceedance(
    delta, regions, windows, threshold, *, min_count: int = 1
) -> dict:
    """Aggregate an ensemble-delta-multi grid into per-window exceedance stats.

    ``delta`` must be a complete :func:`value_delta_multi` result (schema
    ``climate-grid/ensemble-delta-multi-v1``) with exactly the keys
    ``schema, elements, times, lats, lons, quantiles, data`` in that order;
    every member is validated against that contract, including each item's
    key order ``values, status, uncertainty, quantile_values`` and the
    nested ``[quantile][time][lat][lon]`` quantile frames.  ``regions``,
    ``windows`` and ``min_count`` follow
    :func:`climate_grid.regional.aggregate_window` exactly: region
    ``cells`` are checked against the delta grid and window
    ``start``/``end`` must occur in ``delta.times``.  ``threshold`` must be
    a finite non-bool int or float; a wrong type raises ``TypeError`` while
    a non-finite value raises ``ValueError``.

    For every window (in window order), region (in region order) and
    element (in delta element order), the non-missing ``(value,
    uncertainty)`` samples of the closed calendar interval are collected
    across the region's cells; ``count`` is the sample count ``n``.  When
    ``n`` is below ``min_count``, ``exceed``, ``rate``, ``mean_excess`` and
    ``uncertainty`` are all ``None``.  Otherwise ``exceed`` is the count of
    samples with ``v > threshold``, ``rate`` is ``exceed / n``,
    ``mean_excess`` is ``sum(max(v - threshold, 0)) / n`` and
    ``uncertainty`` is ``sqrt(sum(u ** 2)) / n`` over the sampled
    uncertainties.

    The returned mapping uses the key order ``schema, elements, threshold,
    windows, regions, data``; ``schema`` is
    ``climate-grid/ensemble-delta-exceedance-v1``, ``elements`` echoes the
    delta elements, ``threshold`` and ``windows`` echo the arguments, and
    ``regions`` lists the region names in input order.  ``data`` is a flat
    list of rows in window-then-region-then-element order; each row uses
    the key order ``window, region, element, count, exceed, rate,
    mean_excess, uncertainty``.  ``count`` and ``exceed`` are ints and
    every output float is ``round(x, 12)`` with negative zero normalized to
    ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    elements, times, lats, lons, _quantiles, data = _validate_ensemble_multi(
        delta, schema=_ENSEMBLE_DELTA_MULTI_SCHEMA, where="delta"
    )

    validated_regions = _validate_regions(regions, len(lats), len(lons))
    validated_windows = _validate_windows(windows, times)

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a finite non-bool int or float")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    result_data = []
    for w_name, _start, _end, t_start, t_end in validated_windows:
        for r_name, cells in validated_regions:
            for element in elements:
                series = data[element]
                stats = _exceedance_window_region(
                    series["values"],
                    series["status"],
                    series["uncertainty"],
                    cells,
                    t_start,
                    t_end,
                    threshold,
                    min_count,
                )
                result_data.append(
                    {
                        "window": w_name,
                        "region": r_name,
                        "element": element,
                        **stats,
                    }
                )

    return {
        "schema": _ENSEMBLE_DELTA_EXCEEDANCE_SCHEMA,
        "elements": list(elements),
        "threshold": threshold,
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


_ENSEMBLE_DELTA_EXCEEDANCE_SUMMARY_SCHEMA = "climate-grid/delta-exceed-summary-v1"

_ENSEMBLE_DELTA_EXCEEDANCE_KEYS = (
    "schema",
    "elements",
    "threshold",
    "windows",
    "regions",
    "data",
)
_ENSEMBLE_DELTA_EXCEEDANCE_ROW_KEYS = (
    "window",
    "region",
    "element",
    "count",
    "exceed",
    "rate",
    "mean_excess",
    "uncertainty",
)


def _validate_exceedance(
    exceedance: Any, *, where: str = "exceedance"
) -> tuple[list[str], Any, list, list[str], list]:
    if not isinstance(exceedance, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(exceedance.keys()) != _ENSEMBLE_DELTA_EXCEEDANCE_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, elements, threshold, "
            "windows, regions, data in order"
        )

    schema_value = exceedance["schema"]
    if not isinstance(schema_value, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema_value != _ENSEMBLE_DELTA_EXCEEDANCE_SCHEMA:
        raise ValueError(f"{where}.schema must be {_ENSEMBLE_DELTA_EXCEEDANCE_SCHEMA!r}")

    elements = exceedance["elements"]
    if not isinstance(elements, list):
        raise TypeError(f"{where}.elements must be a list")
    if len(elements) == 0:
        raise ValueError(f"{where}.elements must be non-empty")
    seen_elements: set[str] = set()
    for index, element in enumerate(elements):
        if not isinstance(element, str):
            raise TypeError(f"{where}.elements[{index}] must be a str")
        if element == "":
            raise ValueError(f"{where}.elements[{index}] must be non-empty")
        if element in seen_elements:
            raise ValueError(f"duplicate {where} element: {element!r}")
        seen_elements.add(element)

    threshold = exceedance["threshold"]
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError(f"{where}.threshold must be a finite non-bool int or float")
    if not math.isfinite(threshold):
        raise ValueError(f"{where}.threshold must be finite")

    windows = _validate_coverage_windows(
        exceedance["windows"], prefix=f"{where}.windows"
    )

    regions = exceedance["regions"]
    if not isinstance(regions, list):
        raise TypeError(f"{where}.regions must be a list")
    if len(regions) == 0:
        raise ValueError(f"{where}.regions must be non-empty")
    seen_regions: set[str] = set()
    for index, region in enumerate(regions):
        if not isinstance(region, str):
            raise TypeError(f"{where}.regions[{index}] must be a str")
        if region == "":
            raise ValueError(f"{where}.regions[{index}] must be non-empty")
        if region in seen_regions:
            raise ValueError(f"duplicate {where} region: {region!r}")
        seen_regions.add(region)

    data = exceedance["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")

    n_windows = len(windows)
    n_regions = len(regions)
    n_elements = len(elements)
    expected_rows = n_windows * n_regions * n_elements
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows "
            "(one per window/region/element combination, in "
            "window-then-region-then-element order)"
        )

    row_index = 0
    for window in windows:
        for region in regions:
            for element in elements:
                target = f"{where}.data[{row_index}]"
                row = data[row_index]
                if not isinstance(row, dict):
                    raise TypeError(f"{target} must be a dict")
                if tuple(row.keys()) != _ENSEMBLE_DELTA_EXCEEDANCE_ROW_KEYS:
                    raise ValueError(
                        f"{target} must have exactly the keys window, region, "
                        "element, count, exceed, rate, mean_excess, uncertainty "
                        "in order"
                    )

                if not isinstance(row["window"], str):
                    raise TypeError(f"{target}.window must be a str")
                if row["window"] != window["name"]:
                    raise ValueError(
                        f"{target}.window must be {window['name']!r} for its "
                        "window-then-region-then-element position"
                    )
                if not isinstance(row["region"], str):
                    raise TypeError(f"{target}.region must be a str")
                if row["region"] != region:
                    raise ValueError(
                        f"{target}.region must be {region!r} for its "
                        "window-then-region-then-element position"
                    )
                if not isinstance(row["element"], str):
                    raise TypeError(f"{target}.element must be a str")
                if row["element"] != element:
                    raise ValueError(
                        f"{target}.element must be {element!r} for its "
                        "window-then-region-then-element position"
                    )

                count = row["count"]
                if not isinstance(count, int) or isinstance(count, bool):
                    raise TypeError(f"{target}.count must be a non-bool int")
                if count < 0:
                    raise ValueError(f"{target}.count must be non-negative")

                stat_fields = ("exceed", "rate", "mean_excess", "uncertainty")
                stats_are_none = [row[name] is None for name in stat_fields]
                if not all(stats_are_none) and any(stats_are_none):
                    raise ValueError(
                        f"{target}: exceed, rate, mean_excess and uncertainty "
                        "must be all None or all present"
                    )
                if all(stats_are_none):
                    if count == 0:
                        row_index += 1
                        continue
                else:
                    if count <= 0:
                        raise ValueError(
                            f"{target}: present stats require a positive count"
                        )
                    exceed = row["exceed"]
                    if not isinstance(exceed, int) or isinstance(exceed, bool):
                        raise TypeError(f"{target}.exceed must be a non-bool int")
                    if exceed < 0:
                        raise ValueError(f"{target}.exceed must be non-negative")
                    if exceed > count:
                        raise ValueError(
                            f"{target}.exceed must not exceed {target}.count"
                        )
                    for name in ("rate", "mean_excess", "uncertainty"):
                        _validate_number(
                            row[name], f"{target}.{name}", nullable=False
                        )
                    if row["rate"] != _round_output(exceed / count):
                        raise ValueError(
                            f"{target}.rate must equal exceed over count"
                        )
                    if row["mean_excess"] < 0:
                        raise ValueError(
                            f"{target}.mean_excess must be non-negative"
                        )
                    if row["uncertainty"] < 0:
                        raise ValueError(
                            f"{target}.uncertainty must be non-negative"
                        )

                row_index += 1

    return elements, threshold, windows, regions, data


def aggregate_value_delta_exceedance_summary(exceedance) -> dict:
    """Summarize an ensemble-delta exceedance result across all windows.

    ``exceedance`` must be a complete :func:`aggregate_value_delta_exceedance`
    result (schema ``climate-grid/ensemble-delta-exceedance-v1``) with exactly
    the keys ``schema, elements, threshold, windows, regions, data`` in that
    order; every member is validated against that contract, including the
    flat window-then-region-then-element ``data`` row order, each row's key
    order ``window, region, element, count, exceed, rate, mean_excess,
    uncertainty`` and the count / exceedance invariants.

    Rows are aggregated across all windows, in element order and then region
    order, by summing ``count``.  When the summed ``count`` is zero,
    ``exceed`` is ``0`` and ``rate``, ``mean_excess`` and ``uncertainty``
    are all ``None``.  When the summed ``count`` is positive but any
    contributing row (a row with positive ``count``) has ``exceed``,
    ``rate``, ``mean_excess`` or ``uncertainty`` of ``None``, ``exceed``
    and all three statistics are ``None``.  Otherwise ``exceed`` is the
    summed exceedance count, ``rate`` is the summed ``exceed`` over the
    summed ``count``, ``mean_excess`` is
    ``sum(mean_excess * count) / sum(count)`` and ``uncertainty`` is
    ``sqrt(sum((uncertainty * count) ** 2)) / sum(count)`` over the
    contributing rows.

    The returned mapping uses the key order ``schema, elements, threshold,
    windows, regions, data``; ``schema`` is
    ``climate-grid/delta-exceed-summary-v1`` and ``elements``, ``threshold``,
    ``windows`` and ``regions`` echo the exceedance members.  Each ``data``
    row uses the key order ``element, region, count, exceed, rate,
    mean_excess, uncertainty``; ``count`` is an int, ``exceed`` is an int or
    ``None`` and the three statistics are floats or ``None``, with every
    output float ``round(x, 12)`` and negative zero normalized to ``0.0``.
    Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    elements, threshold, windows, regions, data = _validate_exceedance(exceedance)

    n_windows = len(windows)
    n_regions = len(regions)
    n_elements = len(elements)

    result_data = []
    for e_index, element in enumerate(elements):
        for r_index, region in enumerate(regions):
            rows = [
                data[w_index * n_regions * n_elements + r_index * n_elements + e_index]
                for w_index in range(n_windows)
            ]

            count = sum(row["count"] for row in rows)
            contributing = [row for row in rows if row["count"] > 0]

            if count == 0:
                exceed = 0
                rate = None
                mean_excess = None
                uncertainty = None
            elif any(row["exceed"] is None for row in contributing):
                exceed = None
                rate = None
                mean_excess = None
                uncertainty = None
            else:
                exceed = sum(row["exceed"] for row in contributing)
                rate = _round_output(exceed / count)
                mean_excess = _round_output(
                    sum(row["mean_excess"] * row["count"] for row in contributing)
                    / count
                )
                uncertainty = _round_output(
                    math.sqrt(
                        sum(
                            (row["uncertainty"] * row["count"]) ** 2
                            for row in contributing
                        )
                    )
                    / count
                )

            result_data.append(
                {
                    "element": element,
                    "region": region,
                    "count": count,
                    "exceed": exceed,
                    "rate": rate,
                    "mean_excess": mean_excess,
                    "uncertainty": uncertainty,
                }
            )

    return {
        "schema": _ENSEMBLE_DELTA_EXCEEDANCE_SUMMARY_SCHEMA,
        "elements": list(elements),
        "threshold": threshold,
        "windows": windows,
        "regions": list(regions),
        "data": result_data,
    }
