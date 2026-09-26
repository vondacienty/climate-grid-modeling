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


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        # Huge ints cannot be represented as finite floats.
        return False


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
    if not _is_finite(cell):
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


_DELTA_EXCEED_SUMMARY_SCHEMA = "climate-grid/delta-exceed-summary-v1"
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
        raise ValueError(
            f"{where}.schema must be {_ENSEMBLE_DELTA_EXCEEDANCE_SCHEMA!r}"
        )

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
                        "element, count, exceed, rate, mean_excess, "
                        "uncertainty in order"
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

                exceed = row["exceed"]
                if exceed is not None:
                    if not isinstance(exceed, int) or isinstance(exceed, bool):
                        raise TypeError(
                            f"{target}.exceed must be a non-bool int or None"
                        )
                    if exceed < 0:
                        raise ValueError(f"{target}.exceed must be non-negative")
                    if exceed > count:
                        raise ValueError(
                            f"{target}.exceed must not exceed count"
                        )

                stat_fields = ("rate", "mean_excess", "uncertainty")
                stats_are_none = [row[name] is None for name in stat_fields]
                if (exceed is None) != all(stats_are_none) or (
                    any(stats_are_none) and not all(stats_are_none)
                ):
                    raise ValueError(
                        f"{target}: exceed, rate, mean_excess and uncertainty "
                        "must be all None or all present"
                    )
                if exceed is not None:
                    if count <= 0:
                        raise ValueError(
                            f"{target}: present stats require a positive count"
                        )
                    for name in stat_fields:
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
    uncertainty`` and the count / rate invariants.

    Rows are aggregated across all windows, in element order and then region
    order, by summing ``count``.  When the summed ``count`` is zero,
    ``exceed`` is ``0`` and ``rate``, ``mean_excess`` and ``uncertainty``
    are ``None``.  When the summed ``count`` is positive but any
    contributing row (a row with positive ``count``) has an ``exceed`` or
    any of the three stats of ``None``, ``exceed`` and the three stats are
    all ``None``.  Otherwise ``exceed`` is the summed exceedances, ``rate``
    is the summed ``exceed`` over the summed ``count``, ``mean_excess`` is
    ``sum(mean_excess * count) / sum(count)`` over the windows and
    ``uncertainty`` is ``sqrt(sum((uncertainty * count) ** 2)) /
    sum(count)`` over the windows.

    The returned mapping uses the key order ``schema, elements, threshold,
    windows, regions, data``; ``schema`` is
    ``climate-grid/delta-exceed-summary-v1`` and ``elements``, ``threshold``,
    ``windows`` and ``regions`` echo the exceedance members.  Each ``data``
    row uses the key order ``element, region, count, exceed, rate,
    mean_excess, uncertainty``; ``count`` is an int, ``exceed`` is an int or
    ``None`` and every output float is ``round(x, 12)`` with negative zero
    normalized to ``0.0``.  Inputs are not modified.

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
            contributing = [row for row in rows if row["count"] > 0]

            count = sum(row["count"] for row in rows)
            if count == 0:
                exceed = 0
                rate = None
                mean_excess = None
                uncertainty = None
            elif any(
                row["exceed"] is None
                or row["rate"] is None
                or row["mean_excess"] is None
                or row["uncertainty"] is None
                for row in contributing
            ):
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
        "schema": _DELTA_EXCEED_SUMMARY_SCHEMA,
        "elements": list(elements),
        "threshold": threshold,
        "windows": windows,
        "regions": list(regions),
        "data": result_data,
    }


_EXCEED_BATCH_SCHEMA = "climate-grid/exceed-batch-v1"


def batch_exceed(delta, regions, windows, thresholds, *, min_count: int = 1) -> dict:
    """Summarize exceedance stats for a batch of thresholds.

    ``delta``, ``regions``, ``windows`` and ``min_count`` follow
    :func:`aggregate_value_delta_exceedance` exactly.  ``thresholds`` must
    be a non-empty, strictly increasing list of finite non-bool int/float
    items; a non-list or a wrong item type raises ``TypeError`` while an
    empty list, a non-finite item or a non-increasing sequence raises
    ``ValueError``.

    For every threshold ``t`` (in ``thresholds`` order) the per-threshold
    summary is exactly
    ``aggregate_value_delta_exceedance_summary(aggregate_value_delta_exceedance(delta, regions, windows, t, min_count=min_count))``.

    The returned mapping uses the key order ``schema, elements, thresholds,
    windows, regions, data``; ``schema`` is ``climate-grid/exceed-batch-v1``,
    ``elements`` follows the delta element order, ``thresholds`` and
    ``windows`` echo the arguments as-is and ``regions`` lists the region
    names in input order.  ``data`` is a flat list of rows in
    threshold-then-element-then-region order; each row uses the key order
    ``threshold, element, region, count, exceed, rate, mean_excess,
    uncertainty`` where ``threshold`` is ``t`` and the remaining values are
    copied from the corresponding summary row.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    _validate_axis(thresholds, "thresholds")

    elements: list[str] = []
    region_names: list[str] = []
    result_data = []
    for threshold in thresholds:
        summary = aggregate_value_delta_exceedance_summary(
            aggregate_value_delta_exceedance(
                delta, regions, windows, threshold, min_count=min_count
            )
        )
        if not elements:
            elements = summary["elements"]
            region_names = summary["regions"]
        for row in summary["data"]:
            result_data.append(
                {
                    "threshold": threshold,
                    "element": row["element"],
                    "region": row["region"],
                    "count": row["count"],
                    "exceed": row["exceed"],
                    "rate": row["rate"],
                    "mean_excess": row["mean_excess"],
                    "uncertainty": row["uncertainty"],
                }
            )

    return {
        "schema": _EXCEED_BATCH_SCHEMA,
        "elements": elements,
        "thresholds": thresholds,
        "windows": windows,
        "regions": region_names,
        "data": result_data,
    }


_EXCEED_COMPARE_SCHEMA = "climate-grid/exceed-compare-v1"
_COMPARE_ITEM_KEYS = ("name", "delta")


def compare_exceed(
    items, regions, windows, thresholds, *, min_count: int = 1
) -> dict:
    """Compare exceedance batches across multiple scenario deltas.

    ``items`` must be a list of at least two mappings, each with exactly the
    keys ``name, delta`` in that order.  ``name`` is a unique non-empty str
    labeling the scenario and ``delta`` is a complete
    :func:`value_delta_multi` result (schema
    ``climate-grid/ensemble-delta-multi-v1``); every item's delta is validated
    against that contract and all deltas must share equal ``elements``,
    ``times``, ``lats`` and ``lons`` arrays in the same order.  ``regions``,
    ``windows``, ``thresholds`` and ``min_count`` follow :func:`batch_exceed`
    exactly.

    For every item, :func:`batch_exceed` is called with the item's delta and
    the remaining arguments; the per-scenario rows are concatenated in item
    (scenario) order, each already ordered threshold-then-element-then-region.

    The returned mapping uses the key order ``schema, scenarios, elements,
    thresholds, windows, regions, data``; ``schema`` is
    ``climate-grid/exceed-compare-v1``, ``scenarios`` lists the item names in
    item order, ``elements`` is taken from the first item's batch result,
    ``thresholds`` and ``windows`` echo the arguments as-is and ``regions``
    lists the region names in input order.  ``data`` is a flat list of rows
    in scenario-then-threshold-then-element-then-region order; each row uses
    the key order ``scenario, threshold, element, region, count, exceed,
    rate, mean_excess, uncertainty`` where ``scenario`` is the item name and
    the remaining values are copied from the corresponding
    :func:`batch_exceed` row.  Inputs are not modified.

    Raises ``TypeError`` for wrong ``items`` / item-member / argument types
    and ``ValueError`` for any other contract violation.
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if len(items) < 2:
        raise ValueError("items must contain at least two items")

    scenario_names: list[str] = []
    validated_deltas: list = []
    seen_names: set[str] = set()
    for index, item in enumerate(items):
        where = f"items[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{where} must be a dict")
        if tuple(item.keys()) != _COMPARE_ITEM_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys name, delta in order"
            )

        name = item["name"]
        if not isinstance(name, str):
            raise TypeError(f"{where}.name must be a str")
        if name == "":
            raise ValueError(f"{where}.name must be non-empty")
        if name in seen_names:
            raise ValueError(f"duplicate scenario name: {name!r}")
        seen_names.add(name)

        elements, times, lats, lons, _quantiles, _data = _validate_ensemble_multi(
            item["delta"],
            schema=_ENSEMBLE_DELTA_MULTI_SCHEMA,
            where=f"{where}.delta",
        )
        if not validated_deltas:
            first_elements = elements
            first_times = times
            first_lats = lats
            first_lons = lons
        else:
            if elements != first_elements:
                raise ValueError(
                    "all items must have equal elements in the same order"
                )
            if times != first_times:
                raise ValueError(
                    "all items must have equal times in the same order"
                )
            if lats != first_lats:
                raise ValueError(
                    "all items must have equal lats in the same order"
                )
            if lons != first_lons:
                raise ValueError(
                    "all items must have equal lons in the same order"
                )

        scenario_names.append(name)
        validated_deltas.append(item["delta"])

    elements_out: list[str] = []
    region_names: list[str] = []
    result_data = []
    for name, delta in zip(scenario_names, validated_deltas):
        batch = batch_exceed(
            delta, regions, windows, thresholds, min_count=min_count
        )
        if not elements_out:
            elements_out = batch["elements"]
            region_names = batch["regions"]
        for row in batch["data"]:
            result_data.append(
                {
                    "scenario": name,
                    "threshold": row["threshold"],
                    "element": row["element"],
                    "region": row["region"],
                    "count": row["count"],
                    "exceed": row["exceed"],
                    "rate": row["rate"],
                    "mean_excess": row["mean_excess"],
                    "uncertainty": row["uncertainty"],
                }
            )

    return {
        "schema": _EXCEED_COMPARE_SCHEMA,
        "scenarios": scenario_names,
        "elements": elements_out,
        "thresholds": thresholds,
        "windows": windows,
        "regions": region_names,
        "data": result_data,
    }


_QSHIFT_SCHEMA = "climate-grid/qshift-v1"


def quantile_shift(items, regions, windows, quantiles, *, min_count: int = 1) -> dict:
    """Compare per-window region quantiles of scenario deltas vs a reference.

    ``items`` follows :func:`compare_exceed` exactly: a list of at least two
    mappings, each with exactly the keys ``name, delta`` in that order, where
    ``name`` is a unique non-empty str and ``delta`` is a complete
    :func:`value_delta_multi` result (schema
    ``climate-grid/ensemble-delta-multi-v1``); every item's delta is validated
    against that contract and all deltas must share equal ``elements``,
    ``times``, ``lats`` and ``lons`` arrays in the same order.  ``regions``,
    ``windows``, ``quantiles`` and ``min_count`` follow
    :func:`aggregate_value_delta_quantile` exactly.

    The first item is the reference.  For every subsequent scenario (in item
    order), window (in window order), region (in region order) and element
    (in element order), the closed calendar interval is swept across the
    region's cells; only cells that are non-``missing`` in both the scenario
    and the reference contribute a sample ``d = v1 - v0`` with combined
    uncertainty ``du = sqrt(u1 ** 2 + u0 ** 2)``.  ``count`` is the sample
    count ``n``.  When ``n`` is below ``min_count``, ``quantiles`` is a list
    of ``None`` of the same length as the argument and ``uncertainty`` is
    ``None``.  Otherwise the samples are sorted ascending and each quantile
    ``q`` is computed as ``h = (n - 1) * q``, ``a = floor(h)``,
    ``b = ceil(h)``, ``d[a] + (h - a) * (d[b] - d[a])``; ``uncertainty`` is
    ``sqrt(sum(du ** 2)) / n`` over the sampled combined uncertainties.

    The returned mapping uses the key order ``schema, reference, scenarios,
    quantiles, windows, regions, data``; ``schema`` is
    ``climate-grid/qshift-v1``, ``reference`` is the first item's name,
    ``scenarios`` lists the remaining item names in item order,
    ``quantiles`` and ``windows`` echo the arguments as-is and ``regions``
    lists the region names in input order.  ``data`` is a flat list of rows
    in scenario-then-window-then-region-then-element order; each row uses
    the key order ``scenario, window, region, element, count, quantiles,
    uncertainty``.  ``count`` is an int and every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs are
    not modified.

    Raises ``TypeError`` for wrong ``items`` / item-member / argument types
    and ``ValueError`` for any other contract violation.
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if len(items) < 2:
        raise ValueError("items must contain at least two items")

    scenario_names: list[str] = []
    validated_datas: list = []
    seen_names: set[str] = set()
    for index, item in enumerate(items):
        where = f"items[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{where} must be a dict")
        if tuple(item.keys()) != _COMPARE_ITEM_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys name, delta in order"
            )

        name = item["name"]
        if not isinstance(name, str):
            raise TypeError(f"{where}.name must be a str")
        if name == "":
            raise ValueError(f"{where}.name must be non-empty")
        if name in seen_names:
            raise ValueError(f"duplicate scenario name: {name!r}")
        seen_names.add(name)

        elements, times, lats, lons, _quantiles, data = _validate_ensemble_multi(
            item["delta"],
            schema=_ENSEMBLE_DELTA_MULTI_SCHEMA,
            where=f"{where}.delta",
        )
        if not validated_datas:
            first_elements = elements
            first_times = times
            first_lats = lats
            first_lons = lons
        else:
            if elements != first_elements:
                raise ValueError(
                    "all items must have equal elements in the same order"
                )
            if times != first_times:
                raise ValueError(
                    "all items must have equal times in the same order"
                )
            if lats != first_lats:
                raise ValueError(
                    "all items must have equal lats in the same order"
                )
            if lons != first_lons:
                raise ValueError(
                    "all items must have equal lons in the same order"
                )

        scenario_names.append(name)
        validated_datas.append(data)

    validated_regions = _validate_regions(regions, len(first_lats), len(first_lons))
    validated_windows = _validate_windows(windows, first_times)
    validated_quantiles = _regional_validate_quantiles(quantiles)

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    reference_data = validated_datas[0]
    result_data = []
    for s_index in range(1, len(items)):
        scenario_data = validated_datas[s_index]
        for w_name, _start, _end, t_start, t_end in validated_windows:
            for r_name, cells in validated_regions:
                for element in first_elements:
                    reference_series = reference_data[element]
                    scenario_series = scenario_data[element]
                    reference_status = reference_series["status"]
                    scenario_status = scenario_series["status"]
                    reference_values = reference_series["values"]
                    scenario_values = scenario_series["values"]
                    reference_uncertainty = reference_series["uncertainty"]
                    scenario_uncertainty = scenario_series["uncertainty"]

                    deltas: list[float] = []
                    combined: list[float] = []
                    for t in range(t_start, t_end + 1):
                        for i, j in cells:
                            if (
                                scenario_status[t][i][j] == "missing"
                                or reference_status[t][i][j] == "missing"
                            ):
                                continue
                            deltas.append(
                                scenario_values[t][i][j]
                                - reference_values[t][i][j]
                            )
                            combined.append(
                                math.sqrt(
                                    scenario_uncertainty[t][i][j] ** 2
                                    + reference_uncertainty[t][i][j] ** 2
                                )
                            )

                    count = len(deltas)
                    if count < min_count:
                        quantile_values = [None for _q in validated_quantiles]
                        uncertainty = None
                    else:
                        deltas.sort()
                        quantile_values = []
                        for q in validated_quantiles:
                            h = (count - 1) * q
                            a = math.floor(h)
                            b = math.ceil(h)
                            quantile_values.append(
                                _round_output(
                                    deltas[a] + (h - a) * (deltas[b] - deltas[a])
                                )
                            )
                        uncertainty = _round_output(
                            math.sqrt(sum(du ** 2 for du in combined)) / count
                        )

                    result_data.append(
                        {
                            "scenario": scenario_names[s_index],
                            "window": w_name,
                            "region": r_name,
                            "element": element,
                            "count": count,
                            "quantiles": quantile_values,
                            "uncertainty": uncertainty,
                        }
                    )

    return {
        "schema": _QSHIFT_SCHEMA,
        "reference": scenario_names[0],
        "scenarios": scenario_names[1:],
        "quantiles": quantiles,
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


_QSHIFT_SUMMARY_SCHEMA = "climate-grid/qshift-summary-v1"
_QSHIFT_KEYS = (
    "schema",
    "reference",
    "scenarios",
    "quantiles",
    "windows",
    "regions",
    "data",
)
_QSHIFT_ROW_KEYS = (
    "scenario",
    "window",
    "region",
    "element",
    "count",
    "quantiles",
    "uncertainty",
)


def _validate_qshift(
    shift: Any, *, where: str = "shift"
) -> tuple[str, list[str], list[str], list, list, list[str], list]:
    if not isinstance(shift, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(shift.keys()) != _QSHIFT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, reference, scenarios, "
            "quantiles, windows, regions, data in order"
        )

    schema = shift["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _QSHIFT_SCHEMA:
        raise ValueError(f"{where}.schema must be {_QSHIFT_SCHEMA!r}")

    reference = shift["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    scenarios = shift["scenarios"]
    if not isinstance(scenarios, list):
        raise TypeError(f"{where}.scenarios must be a list")
    if len(scenarios) == 0:
        raise ValueError(f"{where}.scenarios must be non-empty")
    seen_scenarios: set[str] = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, str):
            raise TypeError(f"{where}.scenarios[{index}] must be a str")
        if scenario == "":
            raise ValueError(f"{where}.scenarios[{index}] must be non-empty")
        if scenario in seen_scenarios:
            raise ValueError(f"duplicate {where}.scenarios entry: {scenario!r}")
        seen_scenarios.add(scenario)
    if reference in seen_scenarios:
        raise ValueError(f"{where}.reference must not appear in {where}.scenarios")

    quantiles = _regional_validate_quantiles(shift["quantiles"])

    windows = _validate_coverage_windows(
        shift["windows"], prefix=f"{where}.windows"
    )

    regions = shift["regions"]
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

    data = shift["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    if len(data) == 0:
        raise ValueError(f"{where}.data must be non-empty")

    n_quantiles = len(quantiles)
    for row_index, row in enumerate(data):
        target = f"{where}.data[{row_index}]"
        if not isinstance(row, dict):
            raise TypeError(f"{target} must be a dict")
        if tuple(row.keys()) != _QSHIFT_ROW_KEYS:
            raise ValueError(
                f"{target} must have exactly the keys scenario, window, region, "
                "element, count, quantiles, uncertainty in order"
            )

        for name in ("scenario", "window", "region", "element"):
            if not isinstance(row[name], str):
                raise TypeError(f"{target}.{name} must be a str")
            if row[name] == "":
                raise ValueError(f"{target}.{name} must be non-empty")

        count = row["count"]
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError(f"{target}.count must be a non-bool int")
        if count < 0:
            raise ValueError(f"{target}.count must be non-negative")

        row_quantiles = row["quantiles"]
        if not isinstance(row_quantiles, list):
            raise TypeError(f"{target}.quantiles must be a list")
        if len(row_quantiles) != n_quantiles:
            raise ValueError(
                f"{target}.quantiles must have {n_quantiles} items "
                "(one per quantile)"
            )
        none_flags = []
        for q_index, value in enumerate(row_quantiles):
            _validate_number(
                value, f"{target}.quantiles[{q_index}]", nullable=True
            )
            none_flags.append(value is None)
        if any(none_flags) and not all(none_flags):
            raise ValueError(
                f"{target}.quantiles must be all None or all numbers"
            )

        uncertainty = row["uncertainty"]
        _validate_number(uncertainty, f"{target}.uncertainty", nullable=True)
        if (uncertainty is None) != all(none_flags):
            raise ValueError(
                f"{target}: quantiles and uncertainty must be None together"
            )
        if uncertainty is not None and uncertainty < 0:
            raise ValueError(f"{target}.uncertainty must be non-negative")
        if count == 0 and uncertainty is not None:
            raise ValueError(
                f"{target}: a zero count requires None quantiles and uncertainty"
            )

    # The element order is taken from the first scenario/window/region block.
    first_scenario = scenarios[0]
    first_window = windows[0]["name"]
    first_region = regions[0]
    elements: list[str] = []
    for row in data:
        if (
            row["scenario"] != first_scenario
            or row["window"] != first_window
            or row["region"] != first_region
        ):
            break
        if row["element"] in elements:
            raise ValueError(
                f"duplicate element {row['element']!r} in the first "
                f"{where}.data block"
            )
        elements.append(row["element"])

    n_elements = len(elements)
    n_windows = len(windows)
    n_regions = len(regions)
    expected_rows = len(scenarios) * n_windows * n_regions * n_elements
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows "
            "(one per scenario/window/region/element combination, in "
            "scenario-then-window-then-region-then-element order)"
        )

    row_index = 0
    for scenario in scenarios:
        for window in windows:
            for region in regions:
                for element in elements:
                    target = f"{where}.data[{row_index}]"
                    row = data[row_index]
                    if row["scenario"] != scenario:
                        raise ValueError(
                            f"{target}.scenario must be {scenario!r} for its "
                            "scenario-then-window-then-region-then-element "
                            "position"
                        )
                    if row["window"] != window["name"]:
                        raise ValueError(
                            f"{target}.window must be {window['name']!r} for "
                            "its scenario-then-window-then-region-then-element "
                            "position"
                        )
                    if row["region"] != region:
                        raise ValueError(
                            f"{target}.region must be {region!r} for its "
                            "scenario-then-window-then-region-then-element "
                            "position"
                        )
                    if row["element"] != element:
                        raise ValueError(
                            f"{target}.element must be {element!r} for its "
                            "scenario-then-window-then-region-then-element "
                            "position"
                        )
                    row_index += 1

    return reference, scenarios, elements, quantiles, windows, regions, data


def quantile_shift_summary(shift, *, min_count: int = 1) -> dict:
    """Summarize a quantile-shift result across all windows and regions.

    ``shift`` must be a complete :func:`quantile_shift` result (schema
    ``climate-grid/qshift-v1``) with exactly the keys ``schema, reference,
    scenarios, quantiles, windows, regions, data`` in that order; every
    member is validated against that contract, including the flat
    scenario-then-window-then-region-then-element ``data`` row order, each
    row's key order ``scenario, window, region, element, count, quantiles,
    uncertainty`` and the count / quantile invariants (a row's ``quantiles``
    are all ``None`` or all finite non-bool numbers, ``uncertainty`` is
    ``None`` exactly when the quantiles are, and a zero ``count`` requires
    the ``None`` form).  The element order is taken from the first
    scenario/window/region block of ``data``.  ``min_count`` must be a
    non-bool positive int.

    Rows are aggregated per scenario (in scenario order) and element (in
    element order) across all windows and regions: ``count`` is the sum of
    the corresponding row counts.  When the summed ``count`` is below
    ``min_count``, or any contributing row (a row with positive ``count``)
    has ``None`` quantiles or a ``None`` uncertainty, the row's
    ``quantiles`` is a list of ``None`` of the same length as the input
    quantiles and ``uncertainty`` is ``None``.  Otherwise each quantile
    value is the count-weighted mean ``sum(n * q) / sum(n)`` over the
    contributing rows and ``uncertainty`` is
    ``sqrt(sum((n * u) ** 2)) / sum(n)``.

    The returned mapping uses the key order ``schema, reference, scenarios,
    elements, quantiles, windows, regions, data``; ``schema`` is
    ``climate-grid/qshift-summary-v1`` and every member except ``elements``
    echoes the shift input as-is.  ``data`` follows the
    scenario-then-element order; each row uses the key order ``scenario,
    element, count, quantiles, uncertainty``.  ``count`` is an int and every
    output float is ``round(x, 12)`` with negative zero normalized to
    ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    (
        reference,
        scenarios,
        elements,
        quantiles,
        windows,
        regions,
        data,
    ) = _validate_qshift(shift)

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    n_quantiles = len(quantiles)
    n_elements = len(elements)
    n_regions = len(regions)
    n_windows = len(windows)

    result_data = []
    for s_index, scenario in enumerate(scenarios):
        for e_index, element in enumerate(elements):
            rows = [
                data[
                    ((s_index * n_windows + w_index) * n_regions + r_index)
                    * n_elements
                    + e_index
                ]
                for w_index in range(n_windows)
                for r_index in range(n_regions)
            ]

            count = sum(row["count"] for row in rows)
            contributing = [row for row in rows if row["count"] > 0]

            if count < min_count or any(
                row["uncertainty"] is None
                or any(value is None for value in row["quantiles"])
                for row in contributing
            ):
                quantile_values = [None for _q in quantiles]
                uncertainty = None
            else:
                quantile_values = [
                    _round_output(
                        sum(
                            row["count"] * row["quantiles"][q_index]
                            for row in contributing
                        )
                        / count
                    )
                    for q_index in range(n_quantiles)
                ]
                uncertainty = _round_output(
                    math.sqrt(
                        sum(
                            (row["count"] * row["uncertainty"]) ** 2
                            for row in contributing
                        )
                    )
                    / count
                )

            result_data.append(
                {
                    "scenario": scenario,
                    "element": element,
                    "count": count,
                    "quantiles": quantile_values,
                    "uncertainty": uncertainty,
                }
            )

    return {
        "schema": _QSHIFT_SUMMARY_SCHEMA,
        "reference": reference,
        "scenarios": list(scenarios),
        "elements": elements,
        "quantiles": shift["quantiles"],
        "windows": windows,
        "regions": list(regions),
        "data": result_data,
    }


_QSHIFT_RANK_SCHEMA = "climate-grid/qshift-rank-v1"
_QSHIFT_SUMMARY_KEYS = (
    "schema",
    "reference",
    "scenarios",
    "elements",
    "quantiles",
    "windows",
    "regions",
    "data",
)
_QSHIFT_SUMMARY_ROW_KEYS = (
    "scenario",
    "element",
    "count",
    "quantiles",
    "uncertainty",
)


def _validate_qshift_summary(
    summary: Any, *, where: str = "summary"
) -> tuple[str, list[str], list[str], list, list]:
    if not isinstance(summary, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(summary.keys()) != _QSHIFT_SUMMARY_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, reference, scenarios, "
            "elements, quantiles, windows, regions, data in order"
        )

    schema = summary["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _QSHIFT_SUMMARY_SCHEMA:
        raise ValueError(f"{where}.schema must be {_QSHIFT_SUMMARY_SCHEMA!r}")

    reference = summary["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    scenarios = summary["scenarios"]
    if not isinstance(scenarios, list):
        raise TypeError(f"{where}.scenarios must be a list")
    if len(scenarios) == 0:
        raise ValueError(f"{where}.scenarios must be non-empty")
    seen_scenarios: set[str] = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, str):
            raise TypeError(f"{where}.scenarios[{index}] must be a str")
        if scenario == "":
            raise ValueError(f"{where}.scenarios[{index}] must be non-empty")
        if scenario in seen_scenarios:
            raise ValueError(f"duplicate {where}.scenarios entry: {scenario!r}")
        seen_scenarios.add(scenario)
    if reference in seen_scenarios:
        raise ValueError(f"{where}.reference must not appear in {where}.scenarios")

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

    quantiles = _regional_validate_quantiles(summary["quantiles"])

    _validate_coverage_windows(summary["windows"], prefix=f"{where}.windows")

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

    n_quantiles = len(quantiles)
    n_elements = len(elements)
    expected_rows = len(scenarios) * n_elements
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows "
            "(one per scenario/element combination, in "
            "scenario-then-element order)"
        )

    row_index = 0
    for scenario in scenarios:
        for element in elements:
            target = f"{where}.data[{row_index}]"
            row = data[row_index]
            if not isinstance(row, dict):
                raise TypeError(f"{target} must be a dict")
            if tuple(row.keys()) != _QSHIFT_SUMMARY_ROW_KEYS:
                raise ValueError(
                    f"{target} must have exactly the keys scenario, element, "
                    "count, quantiles, uncertainty in order"
                )

            if not isinstance(row["scenario"], str):
                raise TypeError(f"{target}.scenario must be a str")
            if row["scenario"] != scenario:
                raise ValueError(
                    f"{target}.scenario must be {scenario!r} for its "
                    "scenario-then-element position"
                )
            if not isinstance(row["element"], str):
                raise TypeError(f"{target}.element must be a str")
            if row["element"] != element:
                raise ValueError(
                    f"{target}.element must be {element!r} for its "
                    "scenario-then-element position"
                )

            count = row["count"]
            if not isinstance(count, int) or isinstance(count, bool):
                raise TypeError(f"{target}.count must be a non-bool int")
            if count < 0:
                raise ValueError(f"{target}.count must be non-negative")

            row_quantiles = row["quantiles"]
            if not isinstance(row_quantiles, list):
                raise TypeError(f"{target}.quantiles must be a list")
            if len(row_quantiles) != n_quantiles:
                raise ValueError(
                    f"{target}.quantiles must have {n_quantiles} items "
                    "(one per quantile)"
                )
            none_flags = []
            for q_index, value in enumerate(row_quantiles):
                _validate_number(
                    value, f"{target}.quantiles[{q_index}]", nullable=True
                )
                none_flags.append(value is None)
            if any(none_flags) and not all(none_flags):
                raise ValueError(
                    f"{target}.quantiles must be all None or all numbers"
                )

            uncertainty = row["uncertainty"]
            _validate_number(uncertainty, f"{target}.uncertainty", nullable=True)
            if (uncertainty is None) != all(none_flags):
                raise ValueError(
                    f"{target}: quantiles and uncertainty must be None together"
                )
            if uncertainty is not None and uncertainty < 0:
                raise ValueError(f"{target}.uncertainty must be non-negative")
            if count == 0 and uncertainty is not None:
                raise ValueError(
                    f"{target}: a zero count requires None quantiles and "
                    "uncertainty"
                )

            row_index += 1

    return reference, scenarios, elements, quantiles, data


def rank_quantile_shift(summary, quantile, *, min_count: int = 1) -> dict:
    """Rank scenarios per element by one quantile of a qshift summary.

    ``summary`` must be a complete :func:`quantile_shift_summary` result
    (schema ``climate-grid/qshift-summary-v1``) with exactly the keys
    ``schema, reference, scenarios, elements, quantiles, windows, regions,
    data`` in that order; every member is validated against that contract,
    including the flat scenario-then-element ``data`` row order, each row's
    key order ``scenario, element, count, quantiles, uncertainty`` and the
    count / quantile invariants (a row's ``quantiles`` are all ``None`` or
    all finite non-bool numbers, ``uncertainty`` is ``None`` exactly when
    the quantiles are, and a zero ``count`` requires the ``None`` form).
    ``quantile`` must be a finite non-bool number present in
    ``summary.quantiles``.  ``min_count`` must be a non-bool positive int.

    Rows are ranked per element (in element order) across the scenarios:
    a row whose ``count`` is below ``min_count``, or whose selected
    quantile or ``uncertainty`` is ``None``, gets ``difference``,
    ``uncertainty`` and ``rank`` all ``None``.  Otherwise ``difference``
    is the row's selected quantile value and the valid rows are ordered by
    descending ``difference`` with ties keeping the original ``scenarios``
    order; ``rank`` runs consecutively from 1.  Invalid rows follow the
    valid ones in their original scenario order.

    The returned mapping uses the key order ``schema, reference, scenarios,
    elements, quantile, data``; ``schema`` is
    ``climate-grid/qshift-rank-v1`` and ``reference``, ``scenarios``,
    ``elements`` and ``quantile`` echo the input.  ``data`` is a flat list
    of rows in element-then-rank order; each row uses the key order
    ``element, scenario, count, difference, uncertainty, rank``.
    ``count`` is an int, ``rank`` is an int or ``None`` and every output
    float is ``round(x, 12)`` with negative zero normalized to ``0.0``.
    Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    reference, scenarios, elements, quantiles, data = _validate_qshift_summary(
        summary
    )

    if not isinstance(quantile, (int, float)) or isinstance(quantile, bool):
        raise TypeError("quantile must be a finite non-bool int or float")
    if not math.isfinite(quantile):
        raise ValueError("quantile must be finite")
    if quantile not in quantiles:
        raise ValueError("quantile must be one of summary.quantiles")

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    q_index = quantiles.index(quantile)
    n_elements = len(elements)

    result_data = []
    for e_index, element in enumerate(elements):
        rows = [
            data[s_index * n_elements + e_index]
            for s_index in range(len(scenarios))
        ]
        valid = []
        invalid = []
        for row in rows:
            value = row["quantiles"][q_index]
            if (
                row["count"] < min_count
                or value is None
                or row["uncertainty"] is None
            ):
                invalid.append(row)
            else:
                valid.append(row)
        valid.sort(
            key=lambda row: row["quantiles"][q_index], reverse=True
        )
        for rank, row in enumerate(valid, start=1):
            result_data.append(
                {
                    "element": element,
                    "scenario": row["scenario"],
                    "count": row["count"],
                    "difference": _round_output(row["quantiles"][q_index]),
                    "uncertainty": _round_output(row["uncertainty"]),
                    "rank": rank,
                }
            )
        for row in invalid:
            result_data.append(
                {
                    "element": element,
                    "scenario": row["scenario"],
                    "count": row["count"],
                    "difference": None,
                    "uncertainty": None,
                    "rank": None,
                }
            )

    return {
        "schema": _QSHIFT_RANK_SCHEMA,
        "reference": reference,
        "scenarios": list(scenarios),
        "elements": list(elements),
        "quantile": quantile,
        "data": result_data,
    }


_QSHIFT_RANK_STABILITY_SCHEMA = "climate-grid/qshift-rank-stability-v1"


def rank_stability(summary, *, min_count: int = 1) -> dict:
    """Aggregate per-quantile scenario ranks into rank stability stats.

    ``summary`` must be a complete :func:`quantile_shift_summary` result
    (schema ``climate-grid/qshift-summary-v1``) with exactly the keys
    ``schema, reference, scenarios, elements, quantiles, windows, regions,
    data`` in that order; every member is validated against that contract,
    including the flat scenario-then-element ``data`` row order, each row's
    key order ``scenario, element, count, quantiles, uncertainty`` and the
    count / quantile invariants (a row's ``quantiles`` are all ``None`` or
    all finite non-bool numbers, ``uncertainty`` is ``None`` exactly when
    the quantiles are, and a zero ``count`` requires the ``None`` form).
    ``min_count`` must be a non-bool positive int.

    For every quantile (in quantile order) and element (in element order)
    the scenarios are ranked: a scenario is valid when its row's ``count``
    is at least ``min_count`` and neither the selected quantile value nor
    the row's ``uncertainty`` is ``None``.  Valid scenarios are ordered by
    descending quantile value with ties keeping the original ``scenarios``
    order; ranks run consecutively from 1.

    Ranks are then aggregated per element (in element order) and scenario
    (in scenario order): ``rank_count`` is the number of quantiles for
    which the scenario was valid.  When ``rank_count`` is zero,
    ``mean_rank``, ``best_rank``, ``worst_rank`` and ``uncertainty`` are
    all ``None``; otherwise they are the mean, minimum and maximum of the
    scenario's ranks and ``sqrt(sum(u ** 2)) / rank_count`` where ``u`` is
    the summary row's ``uncertainty``, counted once per valid quantile.

    The returned mapping uses the key order ``schema, reference, scenarios,
    elements, quantiles, data``; ``schema`` is
    ``climate-grid/qshift-rank-stability-v1`` and ``reference``,
    ``scenarios``, ``elements`` and ``quantiles`` echo the summary.
    ``data`` is a flat list of rows in element-then-scenario order; each
    row uses the key order ``element, scenario, rank_count, mean_rank,
    best_rank, worst_rank, uncertainty``.  ``rank_count``, ``best_rank``
    and ``worst_rank`` are ints (or ``None``) and every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs are
    not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    reference, scenarios, elements, quantiles, data = _validate_qshift_summary(
        summary
    )

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    n_elements = len(elements)
    n_quantiles = len(quantiles)
    n_scenarios = len(scenarios)

    result_data = []
    for e_index, element in enumerate(elements):
        rows = [
            data[s_index * n_elements + e_index]
            for s_index in range(n_scenarios)
        ]

        # ranks[q_index][s_index] is the scenario's rank for that quantile,
        # or None when the scenario is not valid for it.
        ranks = [
            [None for _s in range(n_scenarios)] for _q in range(n_quantiles)
        ]
        for q_index in range(n_quantiles):
            valid = []
            for s_index, row in enumerate(rows):
                value = row["quantiles"][q_index]
                if (
                    row["count"] >= min_count
                    and value is not None
                    and row["uncertainty"] is not None
                ):
                    valid.append((value, s_index))
            # The sort is stable, so ties keep the original scenarios order.
            valid.sort(key=lambda item: item[0], reverse=True)
            for rank, (_value, s_index) in enumerate(valid, start=1):
                ranks[q_index][s_index] = rank

        for s_index, scenario in enumerate(scenarios):
            scenario_ranks = [
                ranks[q_index][s_index]
                for q_index in range(n_quantiles)
                if ranks[q_index][s_index] is not None
            ]
            rank_count = len(scenario_ranks)
            if rank_count == 0:
                mean_rank = None
                best_rank = None
                worst_rank = None
                uncertainty = None
            else:
                mean_rank = _round_output(sum(scenario_ranks) / rank_count)
                best_rank = min(scenario_ranks)
                worst_rank = max(scenario_ranks)
                u = rows[s_index]["uncertainty"]
                uncertainty = _round_output(
                    math.sqrt(rank_count * u ** 2) / rank_count
                )
            result_data.append(
                {
                    "element": element,
                    "scenario": scenario,
                    "rank_count": rank_count,
                    "mean_rank": mean_rank,
                    "best_rank": best_rank,
                    "worst_rank": worst_rank,
                    "uncertainty": uncertainty,
                }
            )

    return {
        "schema": _QSHIFT_RANK_STABILITY_SCHEMA,
        "reference": reference,
        "scenarios": list(scenarios),
        "elements": list(elements),
        "quantiles": summary["quantiles"],
        "data": result_data,
    }


_COMPOSITE_RANK_SCHEMA = "climate-grid/crank-v1"
_QSHIFT_RANK_STABILITY_KEYS = (
    "schema",
    "reference",
    "scenarios",
    "elements",
    "quantiles",
    "data",
)
_QSHIFT_RANK_STABILITY_ROW_KEYS = (
    "element",
    "scenario",
    "rank_count",
    "mean_rank",
    "best_rank",
    "worst_rank",
    "uncertainty",
)


def _validate_rank_stability(
    stability: Any, *, where: str = "stability"
) -> tuple[list[str], list[str], list]:
    if not isinstance(stability, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(stability.keys()) != _QSHIFT_RANK_STABILITY_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, reference, scenarios, "
            "elements, quantiles, data in order"
        )

    schema = stability["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _QSHIFT_RANK_STABILITY_SCHEMA:
        raise ValueError(
            f"{where}.schema must be {_QSHIFT_RANK_STABILITY_SCHEMA!r}"
        )

    reference = stability["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    scenarios = stability["scenarios"]
    if not isinstance(scenarios, list):
        raise TypeError(f"{where}.scenarios must be a list")
    if len(scenarios) == 0:
        raise ValueError(f"{where}.scenarios must be non-empty")
    seen_scenarios: set[str] = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, str):
            raise TypeError(f"{where}.scenarios[{index}] must be a str")
        if scenario == "":
            raise ValueError(f"{where}.scenarios[{index}] must be non-empty")
        if scenario in seen_scenarios:
            raise ValueError(f"duplicate {where}.scenarios entry: {scenario!r}")
        seen_scenarios.add(scenario)
    if reference in seen_scenarios:
        raise ValueError(f"{where}.reference must not appear in {where}.scenarios")

    elements = stability["elements"]
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

    quantiles = _regional_validate_quantiles(stability["quantiles"])

    data = stability["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")

    n_quantiles = len(quantiles)
    n_scenarios = len(scenarios)
    expected_rows = len(elements) * n_scenarios
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows "
            "(one per element/scenario combination, in "
            "element-then-scenario order)"
        )

    row_index = 0
    for element in elements:
        for scenario in scenarios:
            target = f"{where}.data[{row_index}]"
            row = data[row_index]
            if not isinstance(row, dict):
                raise TypeError(f"{target} must be a dict")
            if tuple(row.keys()) != _QSHIFT_RANK_STABILITY_ROW_KEYS:
                raise ValueError(
                    f"{target} must have exactly the keys element, scenario, "
                    "rank_count, mean_rank, best_rank, worst_rank, uncertainty "
                    "in order"
                )

            if not isinstance(row["element"], str):
                raise TypeError(f"{target}.element must be a str")
            if row["element"] != element:
                raise ValueError(
                    f"{target}.element must be {element!r} for its "
                    "element-then-scenario position"
                )
            if not isinstance(row["scenario"], str):
                raise TypeError(f"{target}.scenario must be a str")
            if row["scenario"] != scenario:
                raise ValueError(
                    f"{target}.scenario must be {scenario!r} for its "
                    "element-then-scenario position"
                )

            rank_count = row["rank_count"]
            if not isinstance(rank_count, int) or isinstance(rank_count, bool):
                raise TypeError(f"{target}.rank_count must be a non-bool int")
            if rank_count < 0:
                raise ValueError(f"{target}.rank_count must be non-negative")
            if rank_count > n_quantiles:
                raise ValueError(
                    f"{target}.rank_count must not exceed the number of quantiles"
                )

            stat_fields = ("mean_rank", "best_rank", "worst_rank", "uncertainty")
            _validate_number(
                row["mean_rank"], f"{target}.mean_rank", nullable=True
            )
            for name in ("best_rank", "worst_rank"):
                value = row[name]
                if value is None:
                    continue
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"{target}.{name} must be a non-bool int")
                if value < 1:
                    raise ValueError(f"{target}.{name} must be positive")
                if value > n_scenarios:
                    raise ValueError(
                        f"{target}.{name} must not exceed the number of "
                        "scenarios"
                    )
            _validate_number(
                row["uncertainty"], f"{target}.uncertainty", nullable=True
            )

            none_flags = [row[name] is None for name in stat_fields]
            if any(none_flags) and not all(none_flags):
                raise ValueError(
                    f"{target}: mean_rank, best_rank, worst_rank and "
                    "uncertainty must be all None or all present"
                )
            if (rank_count == 0) != all(none_flags):
                raise ValueError(
                    f"{target}: mean_rank, best_rank, worst_rank and "
                    "uncertainty must be None exactly when rank_count is zero"
                )
            if rank_count > 0:
                if row["best_rank"] > row["worst_rank"]:
                    raise ValueError(
                        f"{target}.best_rank must not exceed {target}.worst_rank"
                    )
                if not (row["best_rank"] <= row["mean_rank"] <= row["worst_rank"]):
                    raise ValueError(
                        f"{target}.mean_rank must lie between best_rank and "
                        "worst_rank"
                    )
                if row["uncertainty"] < 0:
                    raise ValueError(
                        f"{target}.uncertainty must be non-negative"
                    )

            row_index += 1

    return scenarios, elements, data


def _validate_weights(weights: Any, n_elements: int, *, where: str = "weights"):
    if not isinstance(weights, list):
        raise TypeError(f"{where} must be a list")
    if len(weights) != n_elements:
        raise ValueError(
            f"{where} must have one item per stability element "
            f"({n_elements} expected)"
        )
    for index, weight in enumerate(weights):
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise TypeError(
                f"{where}[{index}] must be a finite non-bool int or float"
            )
        if not math.isfinite(weight):
            raise ValueError(f"{where}[{index}] must be finite")
        if weight < 0:
            raise ValueError(f"{where}[{index}] must be non-negative")
    return weights


def _composite_rows(
    scenarios: list[str],
    elements: list[str],
    data: list,
    weights: list,
    min_elements: int,
) -> list[dict]:
    """Per-scenario composite stats in scenario order, without ranks."""
    n_scenarios = len(scenarios)
    n_elements = len(elements)

    rows = []
    for s_index, scenario in enumerate(scenarios):
        included = [
            (weights[e_index], data[e_index * n_scenarios + s_index])
            for e_index in range(n_elements)
            if weights[e_index] > 0
            and data[e_index * n_scenarios + s_index]["rank_count"] > 0
        ]
        covered = len(included)
        if covered < min_elements:
            rows.append(
                {
                    "scenario": scenario,
                    "covered": covered,
                    "mean": None,
                    "best": None,
                    "worst": None,
                    "u": None,
                }
            )
            continue
        # Divide the weights by the largest positive one and fsum-normalize
        # them into coefficients a (their fsum is 1); this keeps every
        # weight * rank / uncertainty product bounded even with extreme
        # weights.  The combined uncertainty is M * hypot(a * u / M) with M
        # the largest included uncertainty, so no intermediate term is
        # squared unscaled and the products cannot overflow.
        max_weight = max(weight for weight, _row in included)
        coeffs = [weight / max_weight for weight, _row in included]
        coeff_total = math.fsum(coeffs)
        coeffs = [coeff / coeff_total for coeff in coeffs]

        uncertainties = [row["uncertainty"] for _weight, row in included]
        max_u = max(uncertainties)
        if max_u == 0:
            combined_u = 0.0
        else:
            combined_u = max_u * math.hypot(
                *[
                    coeff * uncertainty / max_u
                    for coeff, uncertainty in zip(coeffs, uncertainties)
                ]
            )

        rows.append(
            {
                "scenario": scenario,
                "covered": covered,
                "mean": _round_output(
                    math.fsum(
                        coeff * row["mean_rank"]
                        for coeff, (_weight, row) in zip(coeffs, included)
                    )
                ),
                "best": min(row["best_rank"] for _weight, row in included),
                "worst": max(row["worst_rank"] for _weight, row in included),
                "u": _round_output(combined_u),
            }
        )
    return rows


def _rank_composite(
    scenarios: list[str],
    elements: list[str],
    data: list,
    weights: list,
    min_elements: int,
) -> list[dict]:
    """Composite rows sorted into rank order with ranks assigned."""
    rows = _composite_rows(scenarios, elements, data, weights, min_elements)

    valid = []
    invalid = []
    for row in rows:
        ranked = {**row, "rank": None}
        if row["mean"] is None:
            invalid.append(ranked)
        else:
            valid.append(ranked)

    # The sort is stable, so ties keep the original scenarios order.
    valid.sort(key=lambda row: row["mean"])
    for rank, row in enumerate(valid, start=1):
        row["rank"] = rank

    return valid + invalid


def composite_rank(stability, weights, *, min_elements: int = 1) -> dict:
    """Combine per-element rank stability stats into a composite ranking.

    ``stability`` must be a complete :func:`rank_stability` result (schema
    ``climate-grid/qshift-rank-stability-v1``) with exactly the keys
    ``schema, reference, scenarios, elements, quantiles, data`` in that
    order; every member is validated against that contract, including the
    flat element-then-scenario ``data`` row order, each row's key order
    ``element, scenario, rank_count, mean_rank, best_rank, worst_rank,
    uncertainty`` and the rank invariants (``rank_count`` is a non-bool int
    between zero and the number of quantiles; a zero ``rank_count``
    requires the four stats to be ``None``, otherwise ``mean_rank`` and
    ``uncertainty`` are finite non-bool numbers with ``uncertainty``
    non-negative and ``best_rank``/``worst_rank`` are positive non-bool
    ints bracketing ``mean_rank``).  ``weights`` is a list with one item
    per ``stability.elements`` entry, in element order; each item must be
    a finite non-bool non-negative number (zero weights are allowed and
    simply do not participate).  ``min_elements`` must be a non-bool
    positive int.

    For every scenario (in scenario order) the rows with a positive weight
    and a positive ``rank_count`` are included; ``covered`` is the number
    of included rows.  When ``covered`` is below ``min_elements``,
    ``mean``, ``best``, ``worst``, ``u`` and ``rank`` are all ``None``.
    Otherwise, with ``W`` the sum of the included weights, ``mean`` is
    ``Σ(w × mean_rank) / W``, ``best`` is the minimum ``best_rank``,
    ``worst`` is the maximum ``worst_rank`` and ``u`` is
    ``√Σ(w × uncertainty)² / W`` over the included rows.  The weights are
    internally divided by the largest positive included weight and
    fsum-normalized into coefficients ``a`` (so ``Σa = 1``), and the
    combined uncertainty is computed as ``M · hypot(aᵢuᵢ/M)`` with ``M``
    the largest included uncertainty (``0.0`` when ``M`` is zero), so
    extreme weights and uncertainties cannot overflow the intermediate
    products.

    The scenarios are then ranked by ascending ``mean`` with ties keeping
    the original ``scenarios`` order; ``rank`` runs consecutively from 1.
    Scenarios without a composite (the invalid ones) follow the ranked
    ones in their original scenario order.

    The returned mapping uses the key order ``schema, scenarios, data``;
    ``schema`` is ``climate-grid/crank-v1`` and ``scenarios`` echoes the
    stability scenarios.  ``data`` is a flat list of rows in rank order
    (invalid rows last, in scenario order); each row uses the key order
    ``scenario, covered, mean, best, worst, u, rank``.  ``covered`` is an
    int, ``best``, ``worst`` and ``rank`` are ints or ``None`` and every
    output float is ``round(x, 12)`` with negative zero normalized to
    ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    scenarios, elements, data = _validate_rank_stability(stability)
    weights = _validate_weights(weights, len(elements))

    if not isinstance(min_elements, int) or isinstance(min_elements, bool):
        raise TypeError("min_elements must be a non-bool int")
    if min_elements < 1:
        raise ValueError("min_elements must be positive")

    ranked_rows = _rank_composite(
        scenarios, elements, data, weights, min_elements
    )

    return {
        "schema": _COMPOSITE_RANK_SCHEMA,
        "scenarios": list(scenarios),
        "data": ranked_rows,
    }


_COMPOSITE_RANK_SENSITIVITY_SCHEMA = "climate-grid/crank-sensitivity-v1"


def _validate_rank_configs(
    configs: Any, n_elements: int, *, minimum: int = 1
) -> list[str]:
    if not isinstance(configs, list):
        raise TypeError("configs must be a list")
    if len(configs) < minimum:
        if minimum == 1:
            raise ValueError("configs must be non-empty")
        raise ValueError(f"configs must contain at least {minimum} entries")

    names: list[str] = []
    seen_names: set[str] = set()
    for index, config in enumerate(configs):
        where = f"configs[{index}]"
        if not isinstance(config, dict):
            raise TypeError(f"{where} must be a dict")
        if list(config.keys()) != ["name", "weights"]:
            raise ValueError(
                f"{where} must have exactly the keys name, weights in order"
            )

        name = config["name"]
        if not isinstance(name, str):
            raise TypeError(f"{where}.name must be a str")
        if name == "":
            raise ValueError(f"{where}.name must be non-empty")
        if name in seen_names:
            raise ValueError(f"duplicate configuration name: {name!r}")
        seen_names.add(name)

        _validate_weights(config["weights"], n_elements, where=f"{where}.weights")
        names.append(name)

    return names


def rank_sensitivity(stability, configs, *, min_elements: int = 1) -> dict:
    """Measure how sensitive composite scenario ranks are to weight choices.

    ``stability`` is validated exactly as in :func:`composite_rank` (a
    complete :func:`rank_stability` result, schema
    ``climate-grid/qshift-rank-stability-v1``).  ``configs`` is a non-empty
    list of dicts; each dict must have exactly the keys ``name, weights`` in
    that order.  ``name`` is a unique non-empty str and ``weights`` follows
    the :func:`composite_rank` weights contract (one finite non-bool
    non-negative number per stability element, in element order).
    ``min_elements`` must be a non-bool positive int.

    A composite ranking is computed for every configuration exactly as in
    :func:`composite_rank`.  Then, for every scenario (in scenario order),
    the configurations whose composite ``rank`` is not ``None`` are
    collected; ``count`` is their number.  When ``count`` is zero,
    ``mean_rank``, ``best_rank``, ``worst_rank`` and ``uncertainty`` are all
    ``None``; otherwise they are respectively the mean of the ranks, the
    minimum rank, the maximum rank and
    ``sqrt(fsum((rank - mean_rank) ** 2) / count)``, the population standard
    deviation of the ranks across configurations.

    The returned mapping uses the key order ``schema, configurations,
    scenarios, data``; ``schema`` is ``climate-grid/crank-sensitivity-v1``,
    ``configurations`` lists the configuration names in input order and
    ``scenarios`` echoes the stability scenarios.  ``data`` follows the
    scenario order; each row uses the key order ``scenario, count,
    mean_rank, best_rank, worst_rank, uncertainty``.  ``count``,
    ``best_rank`` and ``worst_rank`` are ints (the last two ``None`` for a
    zero count) and every output float is ``round(x, 12)`` with negative
    zero normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    scenarios, elements, data = _validate_rank_stability(stability)
    names = _validate_rank_configs(configs, len(elements))

    if not isinstance(min_elements, int) or isinstance(min_elements, bool):
        raise TypeError("min_elements must be a non-bool int")
    if min_elements < 1:
        raise ValueError("min_elements must be positive")

    ranks_by_config: dict[str, dict[str, int]] = {}
    for config in configs:
        composite_rows = _rank_composite(
            scenarios, elements, data, config["weights"], min_elements
        )
        ranks_by_config[config["name"]] = {
            row["scenario"]: row["rank"] for row in composite_rows
        }

    result_data = []
    for scenario in scenarios:
        ranks = [
            ranks_by_config[name][scenario]
            for name in names
            if ranks_by_config[name][scenario] is not None
        ]
        count = len(ranks)
        if count == 0:
            result_data.append(
                {
                    "scenario": scenario,
                    "count": 0,
                    "mean_rank": None,
                    "best_rank": None,
                    "worst_rank": None,
                    "uncertainty": None,
                }
            )
            continue
        mean_rank = math.fsum(ranks) / count
        uncertainty = math.sqrt(
            math.fsum((rank - mean_rank) ** 2 for rank in ranks) / count
        )
        result_data.append(
            {
                "scenario": scenario,
                "count": count,
                "mean_rank": _round_output(mean_rank),
                "best_rank": min(ranks),
                "worst_rank": max(ranks),
                "uncertainty": _round_output(uncertainty),
            }
        )

    return {
        "schema": _COMPOSITE_RANK_SENSITIVITY_SCHEMA,
        "configurations": names,
        "scenarios": list(scenarios),
        "data": result_data,
    }

_COMPOSITE_RANK_ROBUSTNESS_SCHEMA = "climate-grid/robust-v1"


def rank_robustness(stability, configs, *, min_elements: int = 1) -> dict:
    """Measure how robust composite scenario ranks are under perturbations.

    ``stability`` is validated exactly as in :func:`composite_rank` (a
    complete :func:`rank_stability` result, schema
    ``climate-grid/qshift-rank-stability-v1``).  ``configs`` is a list of at
    least two dicts validated exactly as in :func:`rank_sensitivity`; each
    dict must have exactly the keys ``name, weights`` in that order,
    ``name`` is a unique non-empty str and ``weights`` follows the
    :func:`composite_rank` weights contract.  The first configuration is
    the baseline and every other configuration is a perturbation.
    ``min_elements`` must be a non-bool positive int.

    A composite ranking is computed for every configuration exactly as in
    :func:`composite_rank`.  Then, for every scenario (in scenario order),
    the perturbation configurations whose composite ``rank`` is not
    ``None`` *and* whose baseline rank is not ``None`` are collected; the
    rank difference for each is ``d = rank - baseline_rank`` and ``count``
    is the number of differences.  When ``count`` is zero, ``range`` is
    ``[None, None]`` and ``stable`` and ``rms`` are ``None``; otherwise
    ``range`` is ``[min(d), max(d)]``, ``stable`` is the proportion of
    differences equal to zero and ``rms`` is
    ``sqrt(fsum(d ** 2) / count)``.

    The returned mapping uses the key order ``schema, baseline, configs,
    data``; ``schema`` is ``climate-grid/robust-v1``, ``baseline`` is the
    first configuration name and ``configs`` lists the remaining
    configuration names in input order.  ``data`` follows the scenario
    order; each row uses the key order ``scenario, baseline_rank, count,
    range, stable, rms``.  ``baseline_rank``, ``count`` and the ``range``
    endpoints are ints or ``None`` and every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs are
    not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    scenarios, elements, data = _validate_rank_stability(stability)
    names = _validate_rank_configs(configs, len(elements), minimum=2)

    if not isinstance(min_elements, int) or isinstance(min_elements, bool):
        raise TypeError("min_elements must be a non-bool int")
    if min_elements < 1:
        raise ValueError("min_elements must be positive")

    ranks_by_config: dict[str, dict[str, int | None]] = {}
    for config in configs:
        composite_rows = _rank_composite(
            scenarios, elements, data, config["weights"], min_elements
        )
        ranks_by_config[config["name"]] = {
            row["scenario"]: row["rank"] for row in composite_rows
        }

    baseline_name = names[0]
    perturbation_names = names[1:]

    result_data = []
    for scenario in scenarios:
        baseline_rank = ranks_by_config[baseline_name][scenario]
        differences = [
            ranks_by_config[name][scenario] - baseline_rank
            for name in perturbation_names
            if ranks_by_config[name][scenario] is not None
            and baseline_rank is not None
        ]
        count = len(differences)
        if count == 0:
            result_data.append(
                {
                    "scenario": scenario,
                    "baseline_rank": baseline_rank,
                    "count": 0,
                    "range": [None, None],
                    "stable": None,
                    "rms": None,
                }
            )
            continue
        stable = sum(1 for difference in differences if difference == 0) / count
        rms = math.sqrt(
            math.fsum(difference ** 2 for difference in differences) / count
        )
        result_data.append(
            {
                "scenario": scenario,
                "baseline_rank": baseline_rank,
                "count": count,
                "range": [min(differences), max(differences)],
                "stable": _round_output(stable),
                "rms": _round_output(rms),
            }
        )

    return {
        "schema": _COMPOSITE_RANK_ROBUSTNESS_SCHEMA,
        "baseline": baseline_name,
        "configs": perturbation_names,
        "data": result_data,
    }


_RANK_TRANSITIONS_SCHEMA = "climate-grid/rank-transitions-v1"


def rank_transitions(stability, configs, *, min_elements: int = 1) -> dict:
    """Tabulate composite rank transitions from a baseline to perturbations.

    ``stability`` is validated exactly as in :func:`composite_rank` (a
    complete :func:`rank_stability` result, schema
    ``climate-grid/qshift-rank-stability-v1``).  ``configs`` is a list of at
    least two dicts validated exactly as in :func:`rank_robustness`; each
    dict must have exactly the keys ``name, weights`` in that order,
    ``name`` is a unique non-empty str and ``weights`` follows the
    :func:`composite_rank` weights contract.  The first configuration is
    the baseline and every other configuration is a perturbation.
    ``min_elements`` must be a non-bool positive int.

    A composite ranking is computed for every configuration exactly as in
    :func:`composite_rank`.  Then, for every perturbation configuration (in
    configuration order) and every scenario (in scenario order), the
    baseline rank and the perturbation rank are recorded; ``delta`` is
    ``rank - baseline_rank`` when both ranks are not ``None`` and ``None``
    otherwise.  The detail rows with both ranks present are counted by
    their ``(baseline_rank, rank)`` pair and the groups are sorted by the
    pair in ascending order.

    The returned mapping uses the key order ``schema, baseline, configs,
    scenarios, data, transitions``; ``schema`` is
    ``climate-grid/rank-transitions-v1``, ``baseline`` is the first
    configuration name, ``configs`` lists the remaining configuration names
    in input order and ``scenarios`` echoes the stability scenarios.
    ``data`` is a flat list of rows in configuration-then-scenario order;
    each row uses the key order ``config, scenario, baseline_rank, rank,
    delta`` and the ranks and ``delta`` are non-bool ints or ``None``.
    ``transitions`` is a list of rows sorted by ascending ``(baseline_rank,
    rank)``; each row uses the key order ``baseline_rank, rank, count`` and
    all three fields are non-bool positive ints.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    scenarios, elements, data = _validate_rank_stability(stability)
    names = _validate_rank_configs(configs, len(elements), minimum=2)

    if not isinstance(min_elements, int) or isinstance(min_elements, bool):
        raise TypeError("min_elements must be a non-bool int")
    if min_elements < 1:
        raise ValueError("min_elements must be positive")

    ranks_by_config: dict[str, dict[str, int | None]] = {}
    for config in configs:
        composite_rows = _rank_composite(
            scenarios, elements, data, config["weights"], min_elements
        )
        ranks_by_config[config["name"]] = {
            row["scenario"]: row["rank"] for row in composite_rows
        }

    baseline_name = names[0]
    perturbation_names = names[1:]

    result_data = []
    transition_counts: dict[tuple[int, int], int] = {}
    for name in perturbation_names:
        for scenario in scenarios:
            baseline_rank = ranks_by_config[baseline_name][scenario]
            rank = ranks_by_config[name][scenario]
            if baseline_rank is None or rank is None:
                delta = None
            else:
                delta = rank - baseline_rank
                pair = (baseline_rank, rank)
                transition_counts[pair] = transition_counts.get(pair, 0) + 1
            result_data.append(
                {
                    "config": name,
                    "scenario": scenario,
                    "baseline_rank": baseline_rank,
                    "rank": rank,
                    "delta": delta,
                }
            )

    transitions = [
        {"baseline_rank": baseline_rank, "rank": rank, "count": count}
        for (baseline_rank, rank), count in sorted(transition_counts.items())
    ]

    return {
        "schema": _RANK_TRANSITIONS_SCHEMA,
        "baseline": baseline_name,
        "configs": perturbation_names,
        "scenarios": list(scenarios),
        "data": result_data,
        "transitions": transitions,
    }


_RANK_TRANSITION_SUMMARY_SCHEMA = "climate-grid/rank-transition-summary-v1"
_RANK_TRANSITIONS_KEYS = (
    "schema",
    "baseline",
    "configs",
    "scenarios",
    "data",
    "transitions",
)
_RANK_TRANSITIONS_ROW_KEYS = ("config", "scenario", "baseline_rank", "rank", "delta")
_RANK_TRANSITIONS_GROUP_KEYS = ("baseline_rank", "rank", "count")


def _validate_rank_transitions(result: Any, *, where: str = "result") -> tuple:
    if not isinstance(result, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(result.keys()) != _RANK_TRANSITIONS_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, baseline, configs, "
            "scenarios, data, transitions in order"
        )

    schema = result["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _RANK_TRANSITIONS_SCHEMA:
        raise ValueError(f"{where}.schema must be {_RANK_TRANSITIONS_SCHEMA!r}")

    baseline = result["baseline"]
    if not isinstance(baseline, str):
        raise TypeError(f"{where}.baseline must be a str")
    if baseline == "":
        raise ValueError(f"{where}.baseline must be non-empty")

    configs = result["configs"]
    if not isinstance(configs, list):
        raise TypeError(f"{where}.configs must be a list")
    if len(configs) == 0:
        raise ValueError(f"{where}.configs must be non-empty")
    seen_configs: set[str] = set()
    for index, config in enumerate(configs):
        if not isinstance(config, str):
            raise TypeError(f"{where}.configs[{index}] must be a str")
        if config == "":
            raise ValueError(f"{where}.configs[{index}] must be non-empty")
        if config in seen_configs:
            raise ValueError(f"duplicate {where}.configs entry: {config!r}")
        seen_configs.add(config)

    scenarios = result["scenarios"]
    if not isinstance(scenarios, list):
        raise TypeError(f"{where}.scenarios must be a list")
    if len(scenarios) == 0:
        raise ValueError(f"{where}.scenarios must be non-empty")
    seen_scenarios: set[str] = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, str):
            raise TypeError(f"{where}.scenarios[{index}] must be a str")
        if scenario == "":
            raise ValueError(f"{where}.scenarios[{index}] must be non-empty")
        if scenario in seen_scenarios:
            raise ValueError(f"duplicate {where}.scenarios entry: {scenario!r}")
        seen_scenarios.add(scenario)

    data = result["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    expected_rows = len(configs) * len(scenarios)
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows "
            "(one per config/scenario combination, in "
            "config-then-scenario order)"
        )

    row_index = 0
    valid_total = 0
    for config in configs:
        for scenario in scenarios:
            row_where = f"{where}.data[{row_index}]"
            row = data[row_index]
            if not isinstance(row, dict):
                raise TypeError(f"{row_where} must be a dict")
            if tuple(row.keys()) != _RANK_TRANSITIONS_ROW_KEYS:
                raise ValueError(
                    f"{row_where} must have exactly the keys config, scenario, "
                    "baseline_rank, rank, delta in order"
                )

            if not isinstance(row["config"], str):
                raise TypeError(f"{row_where}.config must be a str")
            if row["config"] != config:
                raise ValueError(
                    f"{row_where}.config must be {config!r} for its "
                    "config-then-scenario position"
                )
            if not isinstance(row["scenario"], str):
                raise TypeError(f"{row_where}.scenario must be a str")
            if row["scenario"] != scenario:
                raise ValueError(
                    f"{row_where}.scenario must be {scenario!r} for its "
                    "config-then-scenario position"
                )

            for rank_name in ("baseline_rank", "rank"):
                rank_value = row[rank_name]
                if rank_value is not None and (
                    not isinstance(rank_value, int) or isinstance(rank_value, bool)
                ):
                    raise TypeError(
                        f"{row_where}.{rank_name} must be a non-bool int or None"
                    )

            baseline_rank = row["baseline_rank"]
            rank = row["rank"]
            delta = row["delta"]
            if delta is not None and (
                not isinstance(delta, int) or isinstance(delta, bool)
            ):
                raise TypeError(
                    f"{row_where}.delta must be a non-bool int or None"
                )
            if baseline_rank is None or rank is None:
                if delta is not None:
                    raise ValueError(
                        f"{row_where}.delta must be None when either rank is None"
                    )
            else:
                valid_total += 1
                if delta != rank - baseline_rank:
                    raise ValueError(
                        f"{row_where}.delta must equal rank - baseline_rank"
                    )

            row_index += 1

    transitions = result["transitions"]
    if not isinstance(transitions, list):
        raise TypeError(f"{where}.transitions must be a list")
    previous_pair: tuple[int, int] | None = None
    transition_total = 0
    for index, group in enumerate(transitions):
        group_where = f"{where}.transitions[{index}]"
        if not isinstance(group, dict):
            raise TypeError(f"{group_where} must be a dict")
        if tuple(group.keys()) != _RANK_TRANSITIONS_GROUP_KEYS:
            raise ValueError(
                f"{group_where} must have exactly the keys baseline_rank, rank, "
                "count in order"
            )
        for field in ("baseline_rank", "rank", "count"):
            value = group[field]
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{group_where}.{field} must be a non-bool int")
            if value < 1:
                raise ValueError(f"{group_where}.{field} must be positive")
        pair = (group["baseline_rank"], group["rank"])
        if previous_pair is not None and pair <= previous_pair:
            raise ValueError(
                f"{where}.transitions must be sorted by ascending "
                "(baseline_rank, rank) with no duplicate pairs"
            )
        previous_pair = pair
        transition_total += group["count"]

    if transition_total != valid_total:
        raise ValueError(
            f"{where}.transitions counts must sum to the number of data rows "
            "with both ranks present"
        )

    return baseline, configs, scenarios, data


def rank_transition_summary(result) -> dict:
    """Summarize composite rank transitions per perturbation configuration.

    ``result`` must be a complete :func:`rank_transitions` result (schema
    ``climate-grid/rank-transitions-v1``) with exactly the keys ``schema,
    baseline, configs, scenarios, data, transitions`` in that order; every
    member is validated against that contract, including the flat
    config-then-scenario ``data`` row order, each row's key order ``config,
    scenario, baseline_rank, rank, delta``, the rank/delta consistency
    (``delta`` is ``rank - baseline_rank`` when both ranks are not ``None``
    and ``None`` otherwise) and the ``transitions`` groups (key order
    ``baseline_rank, rank, count``; non-bool positive ints; sorted by
    ascending ``(baseline_rank, rank)``; counts summing to the number of
    data rows with both ranks present).

    For every perturbation configuration (in configuration order),
    ``total`` is the number of scenarios, ``valid`` is the number of data
    rows whose baseline rank and perturbation rank are both not ``None``
    and ``stable`` is the number of those rows with equal ranks.  When
    ``valid`` is zero, ``stable_rate`` is ``None``; otherwise it is
    ``stable / valid``.  The rows with both ranks present are counted by
    their ``(baseline_rank, rank)`` pair and the groups are sorted by the
    pair in ascending order; each group's ``rate`` is ``count / valid``.

    The returned mapping uses the key order ``schema, baseline, configs,
    scenarios, data, transitions``; ``schema`` is
    ``climate-grid/rank-transition-summary-v1`` and ``baseline``,
    ``configs`` and ``scenarios`` echo the input arrays.  ``data`` follows
    the configuration order; each row uses the key order ``config, total,
    valid, stable, stable_rate``.  ``transitions`` is a flat list in
    configuration-then-pair order; each row uses the key order ``config,
    baseline_rank, rank, count, rate``.  The counts and ranks are non-bool
    ints, the rates are floats or ``None`` and every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs are
    not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    baseline, configs, scenarios, data = _validate_rank_transitions(result)

    n_scenarios = len(scenarios)
    summary_data = []
    summary_transitions = []
    for config_index, config in enumerate(configs):
        rows = data[
            config_index * n_scenarios : (config_index + 1) * n_scenarios
        ]
        valid_rows = [
            row
            for row in rows
            if row["baseline_rank"] is not None and row["rank"] is not None
        ]
        valid = len(valid_rows)
        stable = sum(
            1 for row in valid_rows if row["baseline_rank"] == row["rank"]
        )
        stable_rate = None if valid == 0 else _round_output(stable / valid)
        summary_data.append(
            {
                "config": config,
                "total": n_scenarios,
                "valid": valid,
                "stable": stable,
                "stable_rate": stable_rate,
            }
        )

        pair_counts: dict[tuple[int, int], int] = {}
        for row in valid_rows:
            pair = (row["baseline_rank"], row["rank"])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        for (baseline_rank, rank), count in sorted(pair_counts.items()):
            summary_transitions.append(
                {
                    "config": config,
                    "baseline_rank": baseline_rank,
                    "rank": rank,
                    "count": count,
                    "rate": _round_output(count / valid),
                }
            )

    return {
        "schema": _RANK_TRANSITION_SUMMARY_SCHEMA,
        "baseline": baseline,
        "configs": list(configs),
        "scenarios": list(scenarios),
        "data": summary_data,
        "transitions": summary_transitions,
    }


_RANK_COMPARE_SCHEMA = "climate-grid/rank-compare-v1"
_RANK_TRANSITION_SUMMARY_KEYS = (
    "schema",
    "baseline",
    "configs",
    "scenarios",
    "data",
    "transitions",
)
_RANK_TRANSITION_SUMMARY_ROW_KEYS = (
    "config",
    "total",
    "valid",
    "stable",
    "stable_rate",
)
_RANK_TRANSITION_SUMMARY_GROUP_KEYS = (
    "config",
    "baseline_rank",
    "rank",
    "count",
    "rate",
)


def _validate_rank_transition_summary(
    summary: Any, *, where: str = "summary", minimum: int = 1
) -> tuple:
    if not isinstance(summary, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(summary.keys()) != _RANK_TRANSITION_SUMMARY_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, baseline, configs, "
            "scenarios, data, transitions in order"
        )

    schema = summary["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _RANK_TRANSITION_SUMMARY_SCHEMA:
        raise ValueError(
            f"{where}.schema must be {_RANK_TRANSITION_SUMMARY_SCHEMA!r}"
        )

    baseline = summary["baseline"]
    if not isinstance(baseline, str):
        raise TypeError(f"{where}.baseline must be a str")
    if baseline == "":
        raise ValueError(f"{where}.baseline must be non-empty")

    configs = summary["configs"]
    if not isinstance(configs, list):
        raise TypeError(f"{where}.configs must be a list")
    if len(configs) < minimum:
        if minimum == 1:
            raise ValueError(f"{where}.configs must be non-empty")
        raise ValueError(
            f"{where}.configs must contain at least {minimum} entries"
        )
    seen_configs: set[str] = set()
    for index, config in enumerate(configs):
        if not isinstance(config, str):
            raise TypeError(f"{where}.configs[{index}] must be a str")
        if config == "":
            raise ValueError(f"{where}.configs[{index}] must be non-empty")
        if config in seen_configs:
            raise ValueError(f"duplicate {where}.configs entry: {config!r}")
        seen_configs.add(config)

    scenarios = summary["scenarios"]
    if not isinstance(scenarios, list):
        raise TypeError(f"{where}.scenarios must be a list")
    if len(scenarios) == 0:
        raise ValueError(f"{where}.scenarios must be non-empty")
    seen_scenarios: set[str] = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, str):
            raise TypeError(f"{where}.scenarios[{index}] must be a str")
        if scenario == "":
            raise ValueError(f"{where}.scenarios[{index}] must be non-empty")
        if scenario in seen_scenarios:
            raise ValueError(f"duplicate {where}.scenarios entry: {scenario!r}")
        seen_scenarios.add(scenario)

    data = summary["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    if len(data) != len(configs):
        raise ValueError(
            f"{where}.data must have {len(configs)} rows "
            "(one per config, in config order)"
        )

    n_scenarios = len(scenarios)
    valids: dict[str, int] = {}
    stable_rates: dict[str, float | None] = {}
    for index, config in enumerate(configs):
        row_where = f"{where}.data[{index}]"
        row = data[index]
        if not isinstance(row, dict):
            raise TypeError(f"{row_where} must be a dict")
        if tuple(row.keys()) != _RANK_TRANSITION_SUMMARY_ROW_KEYS:
            raise ValueError(
                f"{row_where} must have exactly the keys config, total, "
                "valid, stable, stable_rate in order"
            )

        if not isinstance(row["config"], str):
            raise TypeError(f"{row_where}.config must be a str")
        if row["config"] != config:
            raise ValueError(
                f"{row_where}.config must be {config!r} for its config position"
            )

        for count_name in ("total", "valid", "stable"):
            count = row[count_name]
            if not isinstance(count, int) or isinstance(count, bool):
                raise TypeError(
                    f"{row_where}.{count_name} must be a non-bool int"
                )
            if count < 0:
                raise ValueError(
                    f"{row_where}.{count_name} must be non-negative"
                )
        if row["total"] != n_scenarios:
            raise ValueError(
                f"{row_where}.total must equal the number of scenarios"
            )
        if row["valid"] > row["total"]:
            raise ValueError(f"{row_where}.valid must not exceed total")
        if row["stable"] > row["valid"]:
            raise ValueError(f"{row_where}.stable must not exceed valid")

        stable_rate = row["stable_rate"]
        if row["valid"] == 0:
            if stable_rate is not None:
                raise ValueError(
                    f"{row_where}.stable_rate must be None when valid is zero"
                )
        else:
            _validate_number(
                stable_rate, f"{row_where}.stable_rate", nullable=False
            )
            if stable_rate != _round_output(row["stable"] / row["valid"]):
                raise ValueError(
                    f"{row_where}.stable_rate must equal stable / valid"
                )

        valids[config] = row["valid"]
        stable_rates[config] = stable_rate

    transitions = summary["transitions"]
    if not isinstance(transitions, list):
        raise TypeError(f"{where}.transitions must be a list")
    config_positions = {config: index for index, config in enumerate(configs)}
    transition_totals = {config: 0 for config in configs}
    previous_key: tuple[int, int, int] | None = None
    for index, group in enumerate(transitions):
        group_where = f"{where}.transitions[{index}]"
        if not isinstance(group, dict):
            raise TypeError(f"{group_where} must be a dict")
        if tuple(group.keys()) != _RANK_TRANSITION_SUMMARY_GROUP_KEYS:
            raise ValueError(
                f"{group_where} must have exactly the keys config, "
                "baseline_rank, rank, count, rate in order"
            )

        config = group["config"]
        if not isinstance(config, str):
            raise TypeError(f"{group_where}.config must be a str")
        if config not in config_positions:
            raise ValueError(
                f"{group_where}.config must be one of {where}.configs"
            )

        for field in ("baseline_rank", "rank", "count"):
            value = group[field]
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{group_where}.{field} must be a non-bool int")
            if value < 1:
                raise ValueError(f"{group_where}.{field} must be positive")

        key = (config_positions[config], group["baseline_rank"], group["rank"])
        if previous_key is not None and key <= previous_key:
            raise ValueError(
                f"{where}.transitions must be sorted by config then ascending "
                "(baseline_rank, rank) with no duplicate pairs"
            )
        previous_key = key

        valid = valids[config]
        if valid == 0:
            raise ValueError(
                f"{group_where}: transition rows require a positive valid "
                f"for config {config!r}"
            )
        rate = group["rate"]
        _validate_number(rate, f"{group_where}.rate", nullable=False)
        if rate != _round_output(group["count"] / valid):
            raise ValueError(f"{group_where}.rate must equal count / valid")

        transition_totals[config] += group["count"]

    for config in configs:
        if transition_totals[config] != valids[config]:
            raise ValueError(
                f"{where}.transitions counts must sum to each config's valid"
            )

    return configs, n_scenarios, valids, stable_rates, transitions


def compare_rank(summary) -> dict:
    """Compare per-config rank transition stats against a reference config.

    ``summary`` must be a complete :func:`rank_transition_summary` result
    (schema ``climate-grid/rank-transition-summary-v1``) with exactly the
    keys ``schema, baseline, configs, scenarios, data, transitions`` in
    that order and at least two configs; every member is validated against
    that contract, including the per-config ``data`` rows (key order
    ``config, total, valid, stable, stable_rate``; ``total`` equal to the
    number of scenarios; ``stable_rate`` equal to ``stable / valid`` or
    ``None`` when ``valid`` is zero) and the ``transitions`` groups (key
    order ``config, baseline_rank, rank, count, rate``; sorted by config
    then ascending ``(baseline_rank, rank)``; ``rate`` equal to
    ``count / valid``; counts summing to each config's ``valid``).  The
    first configuration is the reference.

    For every configuration (in configuration order), ``valid`` echoes the
    summary row and ``mean_abs`` is ``Σ(count × |rank − baseline_rank|) /
    valid`` over the configuration's transition groups; ``normalized`` is
    ``mean_abs / max(n_scenarios − 1, 1)``.  When ``valid`` is zero,
    ``mean_abs`` and ``normalized`` are both ``None``.  ``stable_delta``
    is ``stable_rate − reference.stable_rate`` and ``normalized_delta`` is
    ``normalized − reference.normalized``; each is ``None`` when either
    operand is ``None``.

    The returned mapping uses the key order ``schema, reference, data``;
    ``schema`` is ``climate-grid/rank-compare-v1`` and ``reference`` is
    the first configuration name.  ``data`` follows the configuration
    order; each row uses the key order ``config, valid, mean_abs,
    normalized, stable_delta, normalized_delta``.  ``valid`` is an int and
    every output float is ``round(x, 12)`` with negative zero normalized
    to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    configs, n_scenarios, valids, stable_rates, transitions = (
        _validate_rank_transition_summary(summary, minimum=2)
    )

    reference = configs[0]
    mean_abs_by_config: dict[str, float | None] = {}
    normalized_by_config: dict[str, float | None] = {}
    for config in configs:
        valid = valids[config]
        if valid == 0:
            mean_abs_by_config[config] = None
            normalized_by_config[config] = None
            continue
        weighted = sum(
            group["count"] * abs(group["rank"] - group["baseline_rank"])
            for group in transitions
            if group["config"] == config
        )
        mean_abs = weighted / valid
        mean_abs_by_config[config] = mean_abs
        normalized_by_config[config] = mean_abs / max(n_scenarios - 1, 1)

    reference_stable_rate = stable_rates[reference]
    reference_normalized = normalized_by_config[reference]

    compare_data = []
    for config in configs:
        stable_rate = stable_rates[config]
        normalized = normalized_by_config[config]
        if stable_rate is None or reference_stable_rate is None:
            stable_delta = None
        else:
            stable_delta = _round_output(stable_rate - reference_stable_rate)
        if normalized is None or reference_normalized is None:
            normalized_delta = None
        else:
            normalized_delta = _round_output(normalized - reference_normalized)
        mean_abs = mean_abs_by_config[config]
        compare_data.append(
            {
                "config": config,
                "valid": valids[config],
                "mean_abs": None if mean_abs is None else _round_output(mean_abs),
                "normalized": (
                    None if normalized is None else _round_output(normalized)
                ),
                "stable_delta": stable_delta,
                "normalized_delta": normalized_delta,
            }
        )

    return {
        "schema": _RANK_COMPARE_SCHEMA,
        "reference": reference,
        "data": compare_data,
    }


_RANK_ATTRIBUTE_SCHEMA = "climate-grid/ra-v1"
_RANK_COMPARE_KEYS = ("schema", "reference", "data")
_RANK_COMPARE_ROW_KEYS = (
    "config",
    "valid",
    "mean_abs",
    "normalized",
    "stable_delta",
    "normalized_delta",
)


def _validate_rank_compare(
    compare: Any, *, where: str = "compare"
) -> tuple[str, list[str], list]:
    if not isinstance(compare, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(compare.keys()) != _RANK_COMPARE_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, reference, data in order"
        )

    schema = compare["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _RANK_COMPARE_SCHEMA:
        raise ValueError(f"{where}.schema must be {_RANK_COMPARE_SCHEMA!r}")

    reference = compare["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    data = compare["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    if len(data) < 2:
        raise ValueError(
            f"{where}.data must contain at least 2 rows (one per config)"
        )

    configs: list[str] = []
    reference_valid: int | None = None
    for index, row in enumerate(data):
        row_where = f"{where}.data[{index}]"
        if not isinstance(row, dict):
            raise TypeError(f"{row_where} must be a dict")
        if tuple(row.keys()) != _RANK_COMPARE_ROW_KEYS:
            raise ValueError(
                f"{row_where} must have exactly the keys config, valid, "
                "mean_abs, normalized, stable_delta, normalized_delta in order"
            )

        config = row["config"]
        if not isinstance(config, str):
            raise TypeError(f"{row_where}.config must be a str")
        if config == "":
            raise ValueError(f"{row_where}.config must be non-empty")
        if config in configs:
            raise ValueError(f"duplicate {where}.data config: {config!r}")

        valid = row["valid"]
        if not isinstance(valid, int) or isinstance(valid, bool):
            raise TypeError(f"{row_where}.valid must be a non-bool int")
        if valid < 0:
            raise ValueError(f"{row_where}.valid must be non-negative")

        for name in ("mean_abs", "normalized"):
            value = row[name]
            if valid == 0:
                if value is not None:
                    raise ValueError(
                        f"{row_where}.{name} must be None when valid is zero"
                    )
            else:
                if value is None:
                    raise ValueError(
                        f"{row_where}.{name} must not be None when valid is "
                        "positive"
                    )
                _validate_number(value, f"{row_where}.{name}", nullable=False)

        if reference_valid is None:
            reference_valid = valid
        for name in ("stable_delta", "normalized_delta"):
            value = row[name]
            if valid == 0 or reference_valid == 0:
                if value is not None:
                    raise ValueError(
                        f"{row_where}.{name} must be None when its config or "
                        "the reference has a zero valid"
                    )
            else:
                _validate_number(value, f"{row_where}.{name}", nullable=False)

        configs.append(config)

    if configs[0] != reference:
        raise ValueError(
            f"{where}.reference must be the first {where}.data config"
        )

    return reference, configs, data


def _validate_drivers(drivers: Any, n_configs: int) -> list:
    if not isinstance(drivers, dict):
        raise TypeError("drivers must be a dict")
    if len(drivers) == 0:
        raise ValueError("drivers must be non-empty")

    validated = []
    for name, values in drivers.items():
        if not isinstance(name, str):
            raise TypeError("drivers keys must be str")
        if name == "":
            raise ValueError("drivers keys must be non-empty")
        where = f"drivers[{name!r}]"
        if not isinstance(values, list):
            raise TypeError(f"{where} must be a list")
        if len(values) != n_configs:
            raise ValueError(
                f"{where} must have {n_configs} items (one per compare config)"
            )
        for index, value in enumerate(values):
            target = f"{where}[{index}]"
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{target} must be a number or None")
            if not math.isfinite(value):
                raise ValueError(f"{target} must be finite")
            if value < -1.0 or value > 1.0:
                raise ValueError(f"{target} must be between -1 and 1")
        validated.append((name, values))
    return validated


def rank_attribute(compare, drivers) -> dict:
    """Attribute per-config rank compare deltas to scenario drivers.

    ``compare`` must be a complete :func:`compare_rank` result (schema
    ``climate-grid/rank-compare-v1``) with exactly the keys ``schema,
    reference, data`` in that order; every member is validated against that
    contract, including the per-config ``data`` rows (key order ``config,
    valid, mean_abs, normalized, stable_delta, normalized_delta``;
    ``mean_abs`` and ``normalized`` are ``None`` exactly when ``valid`` is
    zero (and required finite numbers when ``valid`` is positive); each
    delta exactly ``None`` when its config or the reference has a zero
    ``valid``) and ``reference`` naming the first config.  ``drivers`` is a
    non-empty dict (insertion order is kept) whose keys are non-empty str
    and whose values are lists with one item per ``compare.data`` config;
    each item is ``None`` or a finite non-bool number in ``[-1, 1]``.

    For every driver (in driver order) the retained pairs are the config
    positions where both the driver value ``x`` and the config's
    ``normalized_delta`` ``y`` are not ``None``; ``count`` is their number
    ``n``.  When ``n < 2`` or ``Sxx = Σ(x − x̄)²`` is zero, ``contribution``
    and ``uncertainty`` are all ``None`` (one slot per config).  Otherwise,
    with ``b = Σ(x − x̄)(y − ȳ) / Sxx``, residuals ``e = y − ȳ − b(x − x̄)``
    and ``r = √(Σe² / n)``, each retained position gets
    ``contribution = b(x − x̄)`` and
    ``uncertainty = r√(1/n + (x − x̄)²/Sxx)`` while dropped positions get
    ``None``.

    The returned mapping uses the key order ``schema, reference, configs,
    data``; ``schema`` is ``climate-grid/ra-v1``, ``reference`` echoes the
    compare reference and ``configs`` lists the config names in
    ``compare.data`` order.  ``data`` follows the driver order; each row
    uses the key order ``driver, count, contribution, uncertainty``.
    ``count`` is an int and every output float is ``round(x, 12)`` with
    negative zero normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    reference, configs, compare_data = _validate_rank_compare(compare)
    validated_drivers = _validate_drivers(drivers, len(configs))

    result_data = _rank_attribute_rows(
        compare_data, validated_drivers, len(configs)
    )

    return {
        "schema": _RANK_ATTRIBUTE_SCHEMA,
        "reference": reference,
        "configs": list(configs),
        "data": result_data,
    }


_RANK_ATTRIBUTE_BATCH_SCHEMA = "climate-grid/ra-batch-v1"
_RANK_ATTRIBUTE_ITEM_KEYS = ("element", "compare")


def _rank_attribute_rows(
    compare_data: list, validated_drivers: list, n_configs: int
) -> list:
    normalized_deltas = [row["normalized_delta"] for row in compare_data]

    result_data = []
    for name, values in validated_drivers:
        retained = [
            (index, x, y)
            for index, (x, y) in enumerate(zip(values, normalized_deltas))
            if x is not None and y is not None
        ]
        n = len(retained)

        contribution = [None] * n_configs
        uncertainty = [None] * n_configs
        if n >= 2:
            x_mean = sum(x for _index, x, _y in retained) / n
            y_mean = sum(y for _index, _x, y in retained) / n
            sxx = sum((x - x_mean) ** 2 for _index, x, _y in retained)
            if sxx != 0.0:
                slope = (
                    sum(
                        (x - x_mean) * (y - y_mean)
                        for _index, x, y in retained
                    )
                    / sxx
                )
                residual_sq = sum(
                    (y - y_mean - slope * (x - x_mean)) ** 2
                    for _index, x, y in retained
                )
                r = math.sqrt(residual_sq / n)
                for index, x, _y in retained:
                    dx = x - x_mean
                    contribution[index] = _round_output(slope * dx)
                    uncertainty[index] = _round_output(
                        r * math.sqrt(1.0 / n + dx * dx / sxx)
                    )

        result_data.append(
            {
                "driver": name,
                "count": n,
                "contribution": contribution,
                "uncertainty": uncertainty,
            }
        )

    return result_data


def batch_rank_attribute(items, drivers) -> dict:
    """Attribute per-config rank compare deltas for several elements at once.

    ``items`` must be a non-empty list of mappings, each with exactly the
    keys ``element, compare`` in that order.  ``element`` is a unique
    non-empty str and ``compare`` is a complete :func:`compare_rank` result
    (schema ``climate-grid/rank-compare-v1``), validated exactly as for
    :func:`rank_attribute`.  Every item's compare must share the same
    ``reference`` and ``configs`` sequence in the same order; both are taken
    from the first item.  ``drivers`` follows :func:`rank_attribute`
    exactly: a non-empty dict (insertion order is kept) whose keys are
    non-empty str and whose values are lists with one item per compare
    config, each ``None`` or a finite non-bool number in ``[-1, 1]``.

    The same driver mapping is attributed against every item's compare
    using the identical regression logic as :func:`rank_attribute`
    (retained config positions where both the driver value and the
    config's ``normalized_delta`` are present; ``contribution`` and
    ``uncertainty`` ``None`` with fewer than two retained positions or
    zero driver variance).

    The returned mapping uses the key order ``schema, reference, configs,
    elements, drivers, data``; ``schema`` is ``climate-grid/ra-batch-v1``,
    ``reference`` and ``configs`` are taken from the first item,
    ``elements`` lists the item element names in item order and
    ``drivers`` lists the driver mapping keys in insertion order.  ``data``
    is a flat list of rows in element-then-driver order; each row uses the
    key order ``element, driver, count, contribution, uncertainty``.
    ``count`` is an int and ``contribution`` and ``uncertainty`` are lists
    in config order, each a finite ``round(x, 12)`` float (negative zero
    normalized to ``0.0``) or ``None`` exactly as in
    :func:`rank_attribute`.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if len(items) == 0:
        raise ValueError("items must be non-empty")

    elements: list[str] = []
    validated_compares: list[tuple[str, list[str], list]] = []
    seen_elements: set[str] = set()
    for index, item in enumerate(items):
        where = f"items[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{where} must be a dict")
        if tuple(item.keys()) != _RANK_ATTRIBUTE_ITEM_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys element, compare in order"
            )

        element = item["element"]
        if not isinstance(element, str):
            raise TypeError(f"{where}.element must be a str")
        if element == "":
            raise ValueError(f"{where}.element must be non-empty")
        if element in seen_elements:
            raise ValueError(f"duplicate element: {element!r}")
        seen_elements.add(element)

        reference, configs, compare_data = _validate_rank_compare(
            item["compare"], where=f"{where}.compare"
        )
        if not validated_compares:
            first_reference = reference
            first_configs = configs
        else:
            if reference != first_reference:
                raise ValueError(
                    "all items must share the same reference, taken from the "
                    "first item"
                )
            if configs != first_configs:
                raise ValueError(
                    "all items must share the same configs in the same order, "
                    "taken from the first item"
                )

        elements.append(element)
        validated_compares.append((reference, configs, compare_data))

    validated_drivers = _validate_drivers(drivers, len(first_configs))
    driver_names = [name for name, _values in validated_drivers]

    result_data = []
    for element, (_reference, _configs, compare_data) in zip(
        elements, validated_compares
    ):
        rows = _rank_attribute_rows(
            compare_data, validated_drivers, len(first_configs)
        )
        for row in rows:
            result_data.append(
                {
                    "element": element,
                    "driver": row["driver"],
                    "count": row["count"],
                    "contribution": row["contribution"],
                    "uncertainty": row["uncertainty"],
                }
            )

    return {
        "schema": _RANK_ATTRIBUTE_BATCH_SCHEMA,
        "reference": first_reference,
        "configs": list(first_configs),
        "elements": elements,
        "drivers": driver_names,
        "data": result_data,
    }


_RANK_ATTRIBUTE_COMPARE_SCHEMA = "climate-grid/ra-compare-v1"
_RANK_ATTRIBUTE_BATCH_KEYS = (
    "schema",
    "reference",
    "configs",
    "elements",
    "drivers",
    "data",
)
_RANK_ATTRIBUTE_BATCH_ROW_KEYS = (
    "element",
    "driver",
    "count",
    "contribution",
    "uncertainty",
)
_RANK_ATTRIBUTE_COMPARE_ITEM_KEYS = ("name", "attribution")


def _validate_string_list(values: Any, name: str, *, minimum: int = 1) -> list[str]:
    if not isinstance(values, list):
        raise TypeError(f"{name} must be a list")
    if len(values) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} items")
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"{name}[{index}] must be a str")
        if value == "":
            raise ValueError(f"{name}[{index}] must be non-empty")
        if value in seen:
            raise ValueError(f"duplicate {name} entry: {value!r}")
        seen.add(value)
    return values


def _validate_rank_attribute_batch(
    attribution: Any, *, where: str = "attribution"
) -> tuple[str, list[str], list[str], list[str], dict]:
    if not isinstance(attribution, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(attribution.keys()) != _RANK_ATTRIBUTE_BATCH_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, reference, configs, "
            "elements, drivers, data in order"
        )

    schema = attribution["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _RANK_ATTRIBUTE_BATCH_SCHEMA:
        raise ValueError(f"{where}.schema must be {_RANK_ATTRIBUTE_BATCH_SCHEMA!r}")

    reference = attribution["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    configs = _validate_string_list(
        attribution["configs"], f"{where}.configs", minimum=2
    )
    elements = _validate_string_list(attribution["elements"], f"{where}.elements")
    drivers = _validate_string_list(attribution["drivers"], f"{where}.drivers")

    data = attribution["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    expected_rows = len(elements) * len(drivers)
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "element/driver combination, in element-then-driver order)"
        )

    n_configs = len(configs)
    rows: dict[tuple[str, str], dict] = {}
    row_index = 0
    for element in elements:
        for driver in drivers:
            row_where = f"{where}.data[{row_index}]"
            row = data[row_index]
            if not isinstance(row, dict):
                raise TypeError(f"{row_where} must be a dict")
            if tuple(row.keys()) != _RANK_ATTRIBUTE_BATCH_ROW_KEYS:
                raise ValueError(
                    f"{row_where} must have exactly the keys element, driver, "
                    "count, contribution, uncertainty in order"
                )

            row_element = row["element"]
            if not isinstance(row_element, str):
                raise TypeError(f"{row_where}.element must be a str")
            if row_element != element:
                raise ValueError(
                    f"{row_where}.element must be {element!r} for its "
                    "element-then-driver position"
                )
            row_driver = row["driver"]
            if not isinstance(row_driver, str):
                raise TypeError(f"{row_where}.driver must be a str")
            if row_driver != driver:
                raise ValueError(
                    f"{row_where}.driver must be {driver!r} for its "
                    "element-then-driver position"
                )

            count = row["count"]
            if not isinstance(count, int) or isinstance(count, bool):
                raise TypeError(f"{row_where}.count must be a non-bool int")
            if count < 0 or count > n_configs:
                raise ValueError(
                    f"{row_where}.count must be between 0 and {n_configs}"
                )

            none_patterns = []
            for member in ("contribution", "uncertainty"):
                values = row[member]
                member_where = f"{row_where}.{member}"
                if not isinstance(values, list):
                    raise TypeError(f"{member_where} must be a list")
                if len(values) != n_configs:
                    raise ValueError(
                        f"{member_where} must have {n_configs} items "
                        "(one per config)"
                    )
                for index, value in enumerate(values):
                    target = f"{member_where}[{index}]"
                    _validate_number(value, target, nullable=True)
                    if member == "uncertainty" and value is not None and value < 0:
                        raise ValueError(f"{target} must be non-negative")
                none_patterns.append([value is None for value in values])

            if none_patterns[0] != none_patterns[1]:
                raise ValueError(
                    f"{row_where}: contribution and uncertainty must be None "
                    "at the same config positions"
                )
            present = sum(not is_none for is_none in none_patterns[0])
            if present != 0 and present != count:
                raise ValueError(
                    f"{row_where}: contribution and uncertainty must be "
                    "present at exactly count config positions or none"
                )
            if count < 2 and present != 0:
                raise ValueError(
                    f"{row_where}: contribution and uncertainty must be all "
                    "None when count is below 2"
                )

            rows[(element, driver)] = row
            row_index += 1

    return reference, configs, elements, drivers, rows


def compare_batch_rank_attribute(items) -> dict:
    """Compare batch rank attributions of several scenarios to a baseline.

    ``items`` must be a list of at least two mappings, each with exactly the
    keys ``name, attribution`` in that order.  ``name`` is a unique
    non-empty str and ``attribution`` is a complete
    :func:`batch_rank_attribute` result (schema ``climate-grid/ra-batch-v1``)
    with exactly the keys ``schema, reference, configs, elements, drivers,
    data`` in that order; every member is validated against that contract,
    including the flat element-then-driver ``data`` row order, each row's
    key order ``element, driver, count, contribution, uncertainty`` and the
    per-config ``contribution``/``uncertainty`` lists (``None`` at the same
    positions in both, present at exactly ``count`` positions or none, all
    ``None`` when ``count`` is below 2 and non-``None`` uncertainties
    non-negative).  Every item's attribution must
    share the same ``reference``, ``configs``, ``elements`` and ``drivers``
    in the same order; all four are taken from the first item.

    The first item is the baseline.  For every other item (in item order),
    every element (in element order) and every driver (in driver order),
    the scenario row and the baseline row are paired config by config:
    positions where either side has a ``None`` contribution or uncertainty
    are skipped; otherwise ``d = c - c0`` and ``du = sqrt(u ** 2 + u0 ** 2)``
    with ``(c0, u0)`` the baseline pair.  ``count`` is the number ``n`` of
    retained pairs; when ``n`` is zero, ``contribution`` and ``uncertainty``
    are both ``None``, otherwise they are ``Σd / n`` and
    ``sqrt(Σdu²) / n``.

    The returned mapping uses the key order ``schema, reference, scenarios,
    elements, drivers, data``; ``schema`` is ``climate-grid/ra-compare-v1``,
    ``reference`` is the first item's name, ``scenarios`` lists the
    remaining item names in item order and ``elements``/``drivers`` are
    taken from the first item's attribution.  ``data`` is a flat list of
    rows in scenario-then-element-then-driver order; each row uses the key
    order ``scenario, element, driver, count, contribution, uncertainty``.
    ``count`` is an int and every output float is ``round(x, 12)`` with
    negative zero normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if len(items) < 2:
        raise ValueError("items must contain at least 2 items")

    names: list[str] = []
    validated_rows: list[dict] = []
    seen_names: set[str] = set()
    for index, item in enumerate(items):
        where = f"items[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{where} must be a dict")
        if tuple(item.keys()) != _RANK_ATTRIBUTE_COMPARE_ITEM_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys name, attribution in order"
            )

        name = item["name"]
        if not isinstance(name, str):
            raise TypeError(f"{where}.name must be a str")
        if name == "":
            raise ValueError(f"{where}.name must be non-empty")
        if name in seen_names:
            raise ValueError(f"duplicate item name: {name!r}")
        seen_names.add(name)

        reference, configs, elements, drivers, rows = (
            _validate_rank_attribute_batch(
                item["attribution"], where=f"{where}.attribution"
            )
        )
        if not validated_rows:
            first_reference = reference
            first_configs = configs
            first_elements = elements
            first_drivers = drivers
        else:
            if reference != first_reference:
                raise ValueError(
                    "all items must share the same attribution reference, "
                    "taken from the first item"
                )
            if configs != first_configs:
                raise ValueError(
                    "all items must share the same attribution configs in the "
                    "same order, taken from the first item"
                )
            if elements != first_elements:
                raise ValueError(
                    "all items must share the same attribution elements in the "
                    "same order, taken from the first item"
                )
            if drivers != first_drivers:
                raise ValueError(
                    "all items must share the same attribution drivers in the "
                    "same order, taken from the first item"
                )

        names.append(name)
        validated_rows.append(rows)

    n_configs = len(first_configs)
    baseline_rows = validated_rows[0]

    result_data = []
    for scenario_index in range(1, len(items)):
        scenario_rows = validated_rows[scenario_index]
        scenario_name = names[scenario_index]
        for element in first_elements:
            for driver in first_drivers:
                key = (element, driver)
                baseline_row = baseline_rows[key]
                scenario_row = scenario_rows[key]

                deltas = []
                delta_uncertainties = []
                for config_index in range(n_configs):
                    c0 = baseline_row["contribution"][config_index]
                    u0 = baseline_row["uncertainty"][config_index]
                    c = scenario_row["contribution"][config_index]
                    u = scenario_row["uncertainty"][config_index]
                    if c0 is None or u0 is None or c is None or u is None:
                        continue
                    deltas.append(c - c0)
                    delta_uncertainties.append(math.sqrt(u * u + u0 * u0))

                n = len(deltas)
                if n == 0:
                    contribution = None
                    uncertainty = None
                else:
                    contribution = _round_output(sum(deltas) / n)
                    uncertainty = _round_output(
                        math.sqrt(sum(du * du for du in delta_uncertainties)) / n
                    )

                result_data.append(
                    {
                        "scenario": scenario_name,
                        "element": element,
                        "driver": driver,
                        "count": n,
                        "contribution": contribution,
                        "uncertainty": uncertainty,
                    }
                )

    return {
        "schema": _RANK_ATTRIBUTE_COMPARE_SCHEMA,
        "reference": names[0],
        "scenarios": names[1:],
        "elements": list(first_elements),
        "drivers": list(first_drivers),
        "data": result_data,
    }


_RANK_ATTRIBUTE_GROUP_SUMMARY_SCHEMA = "climate-grid/ra-group-summary-v1"
_RANK_ATTRIBUTE_COMPARE_KEYS = (
    "schema",
    "reference",
    "scenarios",
    "elements",
    "drivers",
    "data",
)
_RANK_ATTRIBUTE_COMPARE_ROW_KEYS = (
    "scenario",
    "element",
    "driver",
    "count",
    "contribution",
    "uncertainty",
)
_RANK_ATTRIBUTE_GROUP_ITEM_KEYS = ("name", "comparison")


def _validate_rank_attribute_compare(
    comparison: Any, *, where: str = "comparison"
) -> tuple[str, list[str], list[str], list[str], dict]:
    if not isinstance(comparison, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(comparison.keys()) != _RANK_ATTRIBUTE_COMPARE_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, reference, scenarios, "
            "elements, drivers, data in order"
        )

    schema = comparison["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _RANK_ATTRIBUTE_COMPARE_SCHEMA:
        raise ValueError(f"{where}.schema must be {_RANK_ATTRIBUTE_COMPARE_SCHEMA!r}")

    reference = comparison["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    scenarios = _validate_string_list(comparison["scenarios"], f"{where}.scenarios")
    if reference in scenarios:
        raise ValueError(
            f"{where}.scenarios must not contain the reference {reference!r}"
        )
    elements = _validate_string_list(comparison["elements"], f"{where}.elements")
    drivers = _validate_string_list(comparison["drivers"], f"{where}.drivers")

    data = comparison["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    expected_rows = len(scenarios) * len(elements) * len(drivers)
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "scenario/element/driver combination, in "
            "scenario-then-element-then-driver order)"
        )

    rows: dict[tuple[str, str, str], dict] = {}
    row_index = 0
    for scenario in scenarios:
        for element in elements:
            for driver in drivers:
                row_where = f"{where}.data[{row_index}]"
                row = data[row_index]
                if not isinstance(row, dict):
                    raise TypeError(f"{row_where} must be a dict")
                if tuple(row.keys()) != _RANK_ATTRIBUTE_COMPARE_ROW_KEYS:
                    raise ValueError(
                        f"{row_where} must have exactly the keys scenario, "
                        "element, driver, count, contribution, uncertainty "
                        "in order"
                    )

                row_scenario = row["scenario"]
                if not isinstance(row_scenario, str):
                    raise TypeError(f"{row_where}.scenario must be a str")
                if row_scenario != scenario:
                    raise ValueError(
                        f"{row_where}.scenario must be {scenario!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_element = row["element"]
                if not isinstance(row_element, str):
                    raise TypeError(f"{row_where}.element must be a str")
                if row_element != element:
                    raise ValueError(
                        f"{row_where}.element must be {element!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_driver = row["driver"]
                if not isinstance(row_driver, str):
                    raise TypeError(f"{row_where}.driver must be a str")
                if row_driver != driver:
                    raise ValueError(
                        f"{row_where}.driver must be {driver!r} for its "
                        "scenario-then-element-then-driver position"
                    )

                count = row["count"]
                if not isinstance(count, int) or isinstance(count, bool):
                    raise TypeError(f"{row_where}.count must be a non-bool int")
                if count < 0:
                    raise ValueError(f"{row_where}.count must be non-negative")

                contribution = row["contribution"]
                uncertainty = row["uncertainty"]
                _validate_number(
                    contribution, f"{row_where}.contribution", nullable=True
                )
                _validate_number(
                    uncertainty, f"{row_where}.uncertainty", nullable=True
                )
                if (contribution is None) != (uncertainty is None):
                    raise ValueError(
                        f"{row_where}: contribution and uncertainty must be "
                        "both None or both present"
                    )
                if (count == 0) != (contribution is None):
                    raise ValueError(
                        f"{row_where}: contribution and uncertainty must be "
                        "None exactly when count is zero"
                    )
                if uncertainty is not None and uncertainty < 0:
                    raise ValueError(
                        f"{row_where}.uncertainty must be non-negative"
                    )

                rows[(scenario, element, driver)] = row
                row_index += 1

    return reference, scenarios, elements, drivers, rows


def summarize_rank_attribute_groups(items) -> dict:
    """Pool grouped rank-attribution comparisons per scenario/element/driver.

    ``items`` must be a list of at least two mappings, each with exactly the
    keys ``name, comparison`` in that order.  ``name`` is a unique non-empty
    str and ``comparison`` is a complete :func:`compare_batch_rank_attribute`
    result (schema ``climate-grid/ra-compare-v1``) with exactly the keys
    ``schema, reference, scenarios, elements, drivers, data`` in that order;
    every member is validated against that contract, including the flat
    scenario-then-element-then-driver ``data`` row order, each row's key
    order ``scenario, element, driver, count, contribution, uncertainty``
    and the row invariants (``count`` a non-negative non-bool int;
    ``contribution`` and ``uncertainty`` both ``None`` exactly when
    ``count`` is zero and both finite numbers otherwise, with
    ``uncertainty`` non-negative).  The items' ``reference`` values may
    differ, but every item's comparison must share the same ``scenarios``,
    ``elements`` and ``drivers`` in the same order; all three are taken
    from the first item.

    For every scenario (in scenario order), element (in element order) and
    driver (in driver order), the rows with ``count > 0`` across all items
    are pooled as ``(n, c, u)`` triples.  ``group_count`` is the number of
    retained rows and ``count`` is ``N = Σn``.  When ``N`` is zero,
    ``contribution`` and ``uncertainty`` are both ``None``; otherwise they
    are the count-weighted pool ``C = Σ(n c) / N`` and
    ``U = √(Σ(n u)²) / N``.

    The returned mapping uses the key order ``schema, groups, scenarios,
    elements, drivers, data``; ``schema`` is
    ``climate-grid/ra-group-summary-v1``, ``groups`` lists the item names
    in item order and ``scenarios``/``elements``/``drivers`` are taken from
    the first item's comparison.  ``data`` is a flat list of rows in
    scenario-then-element-then-driver order; each row uses the key order
    ``scenario, element, driver, group_count, count, contribution,
    uncertainty``.  ``group_count`` and ``count`` are ints and every output
    float is ``round(x, 12)`` with negative zero normalized to ``0.0``.
    Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if len(items) < 2:
        raise ValueError("items must contain at least 2 items")

    names: list[str] = []
    validated_rows: list[dict] = []
    seen_names: set[str] = set()
    for index, item in enumerate(items):
        where = f"items[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{where} must be a dict")
        if tuple(item.keys()) != _RANK_ATTRIBUTE_GROUP_ITEM_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys name, comparison in order"
            )

        name = item["name"]
        if not isinstance(name, str):
            raise TypeError(f"{where}.name must be a str")
        if name == "":
            raise ValueError(f"{where}.name must be non-empty")
        if name in seen_names:
            raise ValueError(f"duplicate item name: {name!r}")
        seen_names.add(name)

        _reference, scenarios, elements, drivers, rows = (
            _validate_rank_attribute_compare(
                item["comparison"], where=f"{where}.comparison"
            )
        )
        if not validated_rows:
            first_scenarios = scenarios
            first_elements = elements
            first_drivers = drivers
        else:
            if scenarios != first_scenarios:
                raise ValueError(
                    "all items must share the same comparison scenarios in "
                    "the same order, taken from the first item"
                )
            if elements != first_elements:
                raise ValueError(
                    "all items must share the same comparison elements in "
                    "the same order, taken from the first item"
                )
            if drivers != first_drivers:
                raise ValueError(
                    "all items must share the same comparison drivers in "
                    "the same order, taken from the first item"
                )

        names.append(name)
        validated_rows.append(rows)

    result_data = []
    for scenario in first_scenarios:
        for element in first_elements:
            for driver in first_drivers:
                key = (scenario, element, driver)
                retained = [
                    rows[key] for rows in validated_rows if rows[key]["count"] > 0
                ]
                group_count = len(retained)
                total = sum(row["count"] for row in retained)
                if total == 0:
                    contribution = None
                    uncertainty = None
                else:
                    contribution = _round_output(
                        sum(
                            row["count"] * row["contribution"] for row in retained
                        )
                        / total
                    )
                    uncertainty = _round_output(
                        math.sqrt(
                            sum(
                                (row["count"] * row["uncertainty"]) ** 2
                                for row in retained
                            )
                        )
                        / total
                    )

                result_data.append(
                    {
                        "scenario": scenario,
                        "element": element,
                        "driver": driver,
                        "group_count": group_count,
                        "count": total,
                        "contribution": contribution,
                        "uncertainty": uncertainty,
                    }
                )

    return {
        "schema": _RANK_ATTRIBUTE_GROUP_SUMMARY_SCHEMA,
        "groups": names,
        "scenarios": list(first_scenarios),
        "elements": list(first_elements),
        "drivers": list(first_drivers),
        "data": result_data,
    }


_RANK_ATTRIBUTE_LAYER_SCHEMA = "climate-grid/ra-layer-v1"
_RANK_ATTRIBUTE_LAYER_ITEM_KEYS = ("period", "region", "summary")
_RANK_ATTRIBUTE_GROUP_SUMMARY_KEYS = (
    "schema",
    "groups",
    "scenarios",
    "elements",
    "drivers",
    "data",
)
_RANK_ATTRIBUTE_GROUP_SUMMARY_ROW_KEYS = (
    "scenario",
    "element",
    "driver",
    "group_count",
    "count",
    "contribution",
    "uncertainty",
)
_RANK_ATTRIBUTE_LAYER_ROW_KEYS = (
    "scenario",
    "element",
    "driver",
    "difference",
    "uncertainty",
)


def _validate_rank_attribute_group_summary(
    summary: Any, *, where: str
) -> tuple[list[str], list[str], list[str], dict]:
    if not isinstance(summary, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(summary.keys()) != _RANK_ATTRIBUTE_GROUP_SUMMARY_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, groups, scenarios, "
            "elements, drivers, data in order"
        )

    schema = summary["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _RANK_ATTRIBUTE_GROUP_SUMMARY_SCHEMA:
        raise ValueError(
            f"{where}.schema must be {_RANK_ATTRIBUTE_GROUP_SUMMARY_SCHEMA!r}"
        )

    groups = _validate_string_list(
        summary["groups"], f"{where}.groups", minimum=2
    )
    scenarios = _validate_string_list(
        summary["scenarios"], f"{where}.scenarios"
    )
    elements = _validate_string_list(
        summary["elements"], f"{where}.elements"
    )
    drivers = _validate_string_list(summary["drivers"], f"{where}.drivers")

    data = summary["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    expected_rows = len(scenarios) * len(elements) * len(drivers)
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "scenario/element/driver combination, in "
            "scenario-then-element-then-driver order)"
        )

    n_groups = len(groups)
    rows: dict[tuple[str, str, str], dict] = {}
    row_index = 0
    for scenario in scenarios:
        for element in elements:
            for driver in drivers:
                row_where = f"{where}.data[{row_index}]"
                row = data[row_index]
                if not isinstance(row, dict):
                    raise TypeError(f"{row_where} must be a dict")
                if tuple(row.keys()) != _RANK_ATTRIBUTE_GROUP_SUMMARY_ROW_KEYS:
                    raise ValueError(
                        f"{row_where} must have exactly the keys scenario, "
                        "element, driver, group_count, count, contribution, "
                        "uncertainty in order"
                    )

                row_scenario = row["scenario"]
                if not isinstance(row_scenario, str):
                    raise TypeError(f"{row_where}.scenario must be a str")
                if row_scenario != scenario:
                    raise ValueError(
                        f"{row_where}.scenario must be {scenario!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_element = row["element"]
                if not isinstance(row_element, str):
                    raise TypeError(f"{row_where}.element must be a str")
                if row_element != element:
                    raise ValueError(
                        f"{row_where}.element must be {element!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_driver = row["driver"]
                if not isinstance(row_driver, str):
                    raise TypeError(f"{row_where}.driver must be a str")
                if row_driver != driver:
                    raise ValueError(
                        f"{row_where}.driver must be {driver!r} for its "
                        "scenario-then-element-then-driver position"
                    )

                for count_name in ("group_count", "count"):
                    count = row[count_name]
                    if not isinstance(count, int) or isinstance(count, bool):
                        raise TypeError(
                            f"{row_where}.{count_name} must be a non-bool int"
                        )
                    if count < 0:
                        raise ValueError(
                            f"{row_where}.{count_name} must be non-negative"
                        )
                if row["group_count"] > n_groups:
                    raise ValueError(
                        f"{row_where}.group_count must not exceed the number of "
                        f"groups ({n_groups})"
                    )
                if (row["group_count"] == 0) != (row["count"] == 0):
                    raise ValueError(
                        f"{row_where}: group_count and count must be zero at "
                        "the same time"
                    )

                contribution = row["contribution"]
                uncertainty = row["uncertainty"]
                _validate_number(
                    contribution, f"{row_where}.contribution", nullable=True
                )
                _validate_number(
                    uncertainty, f"{row_where}.uncertainty", nullable=True
                )
                if (contribution is None) != (uncertainty is None):
                    raise ValueError(
                        f"{row_where}: contribution and uncertainty must be "
                        "both None or both present"
                    )
                if (row["count"] == 0) != (contribution is None):
                    raise ValueError(
                        f"{row_where}: contribution and uncertainty must be "
                        "None exactly when count is zero"
                    )
                if uncertainty is not None and uncertainty < 0:
                    raise ValueError(
                        f"{row_where}.uncertainty must be non-negative"
                    )

                rows[(scenario, element, driver)] = row
                row_index += 1

    return scenarios, elements, drivers, rows


def compare_attribute_layers(items) -> dict:
    """Compare pooled rank-attribution layers across period/region pairs.

    ``items`` must be a list of at least two mappings, each with exactly the
    keys ``period, region, summary`` in that order.  ``period`` and
    ``region`` are non-empty str and every ``(period, region)`` pair must be
    unique.  ``summary`` is a complete :func:`summarize_rank_attribute_groups`
    result (schema ``climate-grid/ra-group-summary-v1``) with exactly the
    keys ``schema, groups, scenarios, elements, drivers, data`` in that
    order; every member is validated against that contract, including the
    flat scenario-then-element-then-driver ``data`` row order, each row's key
    order ``scenario, element, driver, group_count, count, contribution,
    uncertainty`` and the row invariants (``group_count`` and ``count``
    non-negative non-bool ints, ``group_count`` no larger than the number of
    groups; ``contribution`` and ``uncertainty`` both ``None`` exactly when
    ``count`` is zero and both finite numbers otherwise, with
    ``uncertainty`` non-negative).  Every item's summary must share the same
    ``scenarios``, ``elements`` and ``drivers`` in the same order; all three
    are taken from the first item.

    The first item is the reference layer.  For every other item (in item
    order), scenario (in scenario order), element (in element order) and
    driver (in driver order), the current row's ``contribution`` ``c`` and
    ``uncertainty`` ``u`` are paired with the reference row's ``c0`` and
    ``u0``: when all four values are not ``None``, ``difference`` is
    ``c - c0`` and ``uncertainty`` is ``sqrt(u ** 2 + u0 ** 2)``; otherwise
    both are ``None``.

    The returned mapping uses the key order ``schema, reference, layers,
    scenarios, elements, drivers, data``; ``schema`` is
    ``climate-grid/ra-layer-v1``, ``reference`` is the first item's
    ``{"period": ..., "region": ...}`` dict and ``layers`` lists the same
    dicts for the remaining items in item order.  ``scenarios``,
    ``elements`` and ``drivers`` echo the first item's summary axes.
    ``data`` is a flat list of rows in layer-then-scenario-then-element-
    then-driver order; each row uses the key order ``scenario, element,
    driver, difference, uncertainty``.  Every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs are
    not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if len(items) < 2:
        raise ValueError("items must contain at least 2 items")

    layer_refs: list[dict] = []
    validated_rows: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, item in enumerate(items):
        where = f"items[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{where} must be a dict")
        if tuple(item.keys()) != _RANK_ATTRIBUTE_LAYER_ITEM_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys period, region, "
                "summary in order"
            )

        period = item["period"]
        region = item["region"]
        if not isinstance(period, str):
            raise TypeError(f"{where}.period must be a str")
        if period == "":
            raise ValueError(f"{where}.period must be non-empty")
        if not isinstance(region, str):
            raise TypeError(f"{where}.region must be a str")
        if region == "":
            raise ValueError(f"{where}.region must be non-empty")
        if (period, region) in seen_pairs:
            raise ValueError(
                f"duplicate period/region pair: {(period, region)!r}"
            )
        seen_pairs.add((period, region))

        scenarios, elements, drivers, rows = (
            _validate_rank_attribute_group_summary(
                item["summary"], where=f"{where}.summary"
            )
        )
        if not validated_rows:
            first_scenarios = scenarios
            first_elements = elements
            first_drivers = drivers
        else:
            if scenarios != first_scenarios:
                raise ValueError(
                    "all items must share the same summary scenarios in the "
                    "same order, taken from the first item"
                )
            if elements != first_elements:
                raise ValueError(
                    "all items must share the same summary elements in the "
                    "same order, taken from the first item"
                )
            if drivers != first_drivers:
                raise ValueError(
                    "all items must share the same summary drivers in the "
                    "same order, taken from the first item"
                )

        layer_refs.append({"period": period, "region": region})
        validated_rows.append(rows)

    baseline_rows = validated_rows[0]

    result_data = []
    for layer_index in range(1, len(items)):
        current_rows = validated_rows[layer_index]
        for scenario in first_scenarios:
            for element in first_elements:
                for driver in first_drivers:
                    key = (scenario, element, driver)
                    baseline_row = baseline_rows[key]
                    current_row = current_rows[key]
                    c0 = baseline_row["contribution"]
                    u0 = baseline_row["uncertainty"]
                    c = current_row["contribution"]
                    u = current_row["uncertainty"]
                    if c0 is None or u0 is None or c is None or u is None:
                        difference = None
                        uncertainty = None
                    else:
                        difference = _round_output(c - c0)
                        uncertainty = _round_output(
                            math.sqrt(u * u + u0 * u0)
                        )

                    result_data.append(
                        {
                            "scenario": scenario,
                            "element": element,
                            "driver": driver,
                            "difference": difference,
                            "uncertainty": uncertainty,
                        }
                    )

    return {
        "schema": _RANK_ATTRIBUTE_LAYER_SCHEMA,
        "reference": layer_refs[0],
        "layers": layer_refs[1:],
        "scenarios": list(first_scenarios),
        "elements": list(first_elements),
        "drivers": list(first_drivers),
        "data": result_data,
    }


_RANK_ATTRIBUTE_LAYER_SUMMARY_SCHEMA = "climate-grid/ra-layer-summary-v1"
_RANK_ATTRIBUTE_LAYER_KEYS = (
    "schema",
    "reference",
    "layers",
    "scenarios",
    "elements",
    "drivers",
    "data",
)
_RANK_ATTRIBUTE_LAYER_REF_KEYS = ("period", "region")
_RANK_ATTRIBUTE_LAYER_SUMMARY_ROW_KEYS = (
    "scenario",
    "element",
    "driver",
    "count",
    "mean",
    "min",
    "max",
    "uncertainty",
)


def _validate_rank_attribute_layer_ref(ref: Any, where: str) -> dict:
    if not isinstance(ref, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(ref.keys()) != _RANK_ATTRIBUTE_LAYER_REF_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys period, region in order"
        )
    period = ref["period"]
    if not isinstance(period, str):
        raise TypeError(f"{where}.period must be a str")
    if period == "":
        raise ValueError(f"{where}.period must be non-empty")
    region = ref["region"]
    if not isinstance(region, str):
        raise TypeError(f"{where}.region must be a str")
    if region == "":
        raise ValueError(f"{where}.region must be non-empty")
    return ref


def _validate_rank_attribute_layer(
    result: Any, *, where: str = "result"
) -> tuple[dict, list, list[str], list[str], list[str], list]:
    if not isinstance(result, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(result.keys()) != _RANK_ATTRIBUTE_LAYER_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, reference, layers, "
            "scenarios, elements, drivers, data in order"
        )

    schema = result["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _RANK_ATTRIBUTE_LAYER_SCHEMA:
        raise ValueError(
            f"{where}.schema must be {_RANK_ATTRIBUTE_LAYER_SCHEMA!r}"
        )

    reference = _validate_rank_attribute_layer_ref(
        result["reference"], f"{where}.reference"
    )
    layers = result["layers"]
    if not isinstance(layers, list):
        raise TypeError(f"{where}.layers must be a list")
    if len(layers) == 0:
        raise ValueError(f"{where}.layers must be non-empty")
    seen_pairs = {(reference["period"], reference["region"])}
    for index, layer in enumerate(layers):
        _validate_rank_attribute_layer_ref(layer, f"{where}.layers[{index}]")
        pair = (layer["period"], layer["region"])
        if pair in seen_pairs:
            raise ValueError(
                f"duplicate period/region pair: {pair!r}"
            )
        seen_pairs.add(pair)

    scenarios = _validate_string_list(
        result["scenarios"], f"{where}.scenarios"
    )
    elements = _validate_string_list(
        result["elements"], f"{where}.elements"
    )
    drivers = _validate_string_list(result["drivers"], f"{where}.drivers")

    data = result["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    expected_rows = (
        len(layers) * len(scenarios) * len(elements) * len(drivers)
    )
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "layer/scenario/element/driver combination, in "
            "layer-then-scenario-then-element-then-driver order)"
        )

    row_index = 0
    for _layer_index in range(len(layers)):
        for scenario in scenarios:
            for element in elements:
                for driver in drivers:
                    row_where = f"{where}.data[{row_index}]"
                    row = data[row_index]
                    if not isinstance(row, dict):
                        raise TypeError(f"{row_where} must be a dict")
                    if tuple(row.keys()) != _RANK_ATTRIBUTE_LAYER_ROW_KEYS:
                        raise ValueError(
                            f"{row_where} must have exactly the keys scenario, "
                            "element, driver, difference, uncertainty in order"
                        )

                    row_scenario = row["scenario"]
                    if not isinstance(row_scenario, str):
                        raise TypeError(f"{row_where}.scenario must be a str")
                    if row_scenario != scenario:
                        raise ValueError(
                            f"{row_where}.scenario must be {scenario!r} for its "
                            "layer-then-scenario-then-element-then-driver "
                            "position"
                        )
                    row_element = row["element"]
                    if not isinstance(row_element, str):
                        raise TypeError(f"{row_where}.element must be a str")
                    if row_element != element:
                        raise ValueError(
                            f"{row_where}.element must be {element!r} for its "
                            "layer-then-scenario-then-element-then-driver "
                            "position"
                        )
                    row_driver = row["driver"]
                    if not isinstance(row_driver, str):
                        raise TypeError(f"{row_where}.driver must be a str")
                    if row_driver != driver:
                        raise ValueError(
                            f"{row_where}.driver must be {driver!r} for its "
                            "layer-then-scenario-then-element-then-driver "
                            "position"
                        )

                    difference = row["difference"]
                    uncertainty = row["uncertainty"]
                    _validate_number(
                        difference, f"{row_where}.difference", nullable=True
                    )
                    _validate_number(
                        uncertainty, f"{row_where}.uncertainty", nullable=True
                    )
                    if (difference is None) != (uncertainty is None):
                        raise ValueError(
                            f"{row_where}: difference and uncertainty must be "
                            "both None or both present"
                        )
                    if uncertainty is not None and uncertainty < 0:
                        raise ValueError(
                            f"{row_where}.uncertainty must be non-negative"
                        )

                    row_index += 1

    return reference, layers, scenarios, elements, drivers, data


def aggregate_attribute_layers(result, *, min_layers: int = 1) -> dict:
    """Aggregate layer-comparison differences per scenario/element/driver.

    ``result`` must be a complete :func:`compare_attribute_layers` result
    (schema ``climate-grid/ra-layer-v1``) with exactly the keys ``schema,
    reference, layers, scenarios, elements, drivers, data`` in that order;
    every member is validated against that contract, including the
    ``reference`` and ``layers`` dicts (exactly the keys ``period, region``
    in that order, both non-empty str, with the ``(period, region)`` pairs
    unique across the reference and all layers), the flat
    layer-then-scenario-then-element-then-driver ``data`` row order, each
    row's key order ``scenario, element, driver, difference, uncertainty``
    and the row invariants (``difference`` and ``uncertainty`` both ``None``
    or both finite numbers, with ``uncertainty`` non-negative).
    ``min_layers`` must be a non-bool positive int.

    For every scenario (in scenario order), element (in element order) and
    driver (in driver order), the ``(difference, uncertainty)`` pairs whose
    members are both not ``None`` are collected in layer order and ``count``
    is their number ``n``.  When ``n`` is below ``min_layers``, ``mean``,
    ``min``, ``max`` and ``uncertainty`` are all ``None``; otherwise they
    are ``sum(d) / n``, the smallest ``d``, the largest ``d`` and
    ``sqrt(sum(u ** 2)) / n`` respectively.

    The returned mapping uses the key order ``schema, reference, layers,
    scenarios, elements, drivers, data``; ``schema`` is
    ``climate-grid/ra-layer-summary-v1`` and ``reference``, ``layers``,
    ``scenarios``, ``elements`` and ``drivers`` echo the input metadata
    as-is.  ``data`` is a flat list in scenario-then-element-then-driver
    order; each row uses the key order ``scenario, element, driver, count,
    mean, min, max, uncertainty``.  ``count`` is a non-bool int and every
    output float is ``round(x, 12)`` with negative zero normalized to
    ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    (
        reference,
        layers,
        scenarios,
        elements,
        drivers,
        data,
    ) = _validate_rank_attribute_layer(result)

    if not isinstance(min_layers, int) or isinstance(min_layers, bool):
        raise TypeError("min_layers must be a non-bool int")
    if min_layers < 1:
        raise ValueError("min_layers must be positive")

    n_layers = len(layers)
    n_scenarios = len(scenarios)
    n_elements = len(elements)
    n_drivers = len(drivers)

    result_data = []
    for s_index, scenario in enumerate(scenarios):
        for e_index, element in enumerate(elements):
            for d_index, driver in enumerate(drivers):
                pairs = []
                for layer_index in range(n_layers):
                    row = data[
                        (
                            (layer_index * n_scenarios + s_index) * n_elements
                            + e_index
                        )
                        * n_drivers
                        + d_index
                    ]
                    difference = row["difference"]
                    uncertainty = row["uncertainty"]
                    if difference is not None and uncertainty is not None:
                        pairs.append((difference, uncertainty))

                count = len(pairs)
                if count < min_layers:
                    mean = None
                    minimum = None
                    maximum = None
                    combined = None
                else:
                    mean = _round_output(
                        sum(d for d, _u in pairs) / count
                    )
                    minimum = _round_output(min(d for d, _u in pairs))
                    maximum = _round_output(max(d for d, _u in pairs))
                    combined = _round_output(
                        math.sqrt(sum(u * u for _d, u in pairs)) / count
                    )

                result_data.append(
                    {
                        "scenario": scenario,
                        "element": element,
                        "driver": driver,
                        "count": count,
                        "mean": mean,
                        "min": minimum,
                        "max": maximum,
                        "uncertainty": combined,
                    }
                )

    return {
        "schema": _RANK_ATTRIBUTE_LAYER_SUMMARY_SCHEMA,
        "reference": reference,
        "layers": list(layers),
        "scenarios": list(scenarios),
        "elements": list(elements),
        "drivers": list(drivers),
        "data": result_data,
    }


_RANK_ATTRIBUTE_SCENARIO_SCHEMA = "climate-grid/ra-scenario-v1"
_RANK_ATTRIBUTE_SCENARIO_ROW_KEYS = (
    "scenario",
    "element",
    "driver",
    "count",
    "difference",
    "uncertainty",
)


def _validate_rank_attribute_layer_summary(
    summary: Any, *, where: str = "summary"
) -> tuple[dict, list, list[str], list[str], list[str], list]:
    if not isinstance(summary, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(summary.keys()) != _RANK_ATTRIBUTE_LAYER_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, reference, layers, "
            "scenarios, elements, drivers, data in order"
        )

    schema = summary["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _RANK_ATTRIBUTE_LAYER_SUMMARY_SCHEMA:
        raise ValueError(
            f"{where}.schema must be {_RANK_ATTRIBUTE_LAYER_SUMMARY_SCHEMA!r}"
        )

    reference = _validate_rank_attribute_layer_ref(
        summary["reference"], f"{where}.reference"
    )
    layers = summary["layers"]
    if not isinstance(layers, list):
        raise TypeError(f"{where}.layers must be a list")
    if len(layers) == 0:
        raise ValueError(f"{where}.layers must be non-empty")
    seen_pairs = {(reference["period"], reference["region"])}
    for index, layer in enumerate(layers):
        _validate_rank_attribute_layer_ref(layer, f"{where}.layers[{index}]")
        pair = (layer["period"], layer["region"])
        if pair in seen_pairs:
            raise ValueError(
                f"duplicate period/region pair: {pair!r}"
            )
        seen_pairs.add(pair)

    scenarios = _validate_string_list(
        summary["scenarios"], f"{where}.scenarios"
    )
    elements = _validate_string_list(
        summary["elements"], f"{where}.elements"
    )
    drivers = _validate_string_list(summary["drivers"], f"{where}.drivers")

    data = summary["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    expected_rows = len(scenarios) * len(elements) * len(drivers)
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "scenario/element/driver combination, in "
            "scenario-then-element-then-driver order)"
        )

    n_layers = len(layers)
    row_index = 0
    for scenario in scenarios:
        for element in elements:
            for driver in drivers:
                row_where = f"{where}.data[{row_index}]"
                row = data[row_index]
                if not isinstance(row, dict):
                    raise TypeError(f"{row_where} must be a dict")
                if tuple(row.keys()) != _RANK_ATTRIBUTE_LAYER_SUMMARY_ROW_KEYS:
                    raise ValueError(
                        f"{row_where} must have exactly the keys scenario, "
                        "element, driver, count, mean, min, max, uncertainty "
                        "in order"
                    )

                row_scenario = row["scenario"]
                if not isinstance(row_scenario, str):
                    raise TypeError(f"{row_where}.scenario must be a str")
                if row_scenario != scenario:
                    raise ValueError(
                        f"{row_where}.scenario must be {scenario!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_element = row["element"]
                if not isinstance(row_element, str):
                    raise TypeError(f"{row_where}.element must be a str")
                if row_element != element:
                    raise ValueError(
                        f"{row_where}.element must be {element!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_driver = row["driver"]
                if not isinstance(row_driver, str):
                    raise TypeError(f"{row_where}.driver must be a str")
                if row_driver != driver:
                    raise ValueError(
                        f"{row_where}.driver must be {driver!r} for its "
                        "scenario-then-element-then-driver position"
                    )

                count = row["count"]
                if not isinstance(count, int) or isinstance(count, bool):
                    raise TypeError(f"{row_where}.count must be a non-bool int")
                if count < 0 or count > n_layers:
                    raise ValueError(
                        f"{row_where}.count must be between 0 and the "
                        "number of layers"
                    )

                mean = row["mean"]
                minimum = row["min"]
                maximum = row["max"]
                uncertainty = row["uncertainty"]
                _validate_number(mean, f"{row_where}.mean", nullable=True)
                _validate_number(minimum, f"{row_where}.min", nullable=True)
                _validate_number(maximum, f"{row_where}.max", nullable=True)
                _validate_number(
                    uncertainty, f"{row_where}.uncertainty", nullable=True
                )
                present = [mean is not None, minimum is not None,
                           maximum is not None, uncertainty is not None]
                if any(present) and not all(present):
                    raise ValueError(
                        f"{row_where}: mean, min, max and uncertainty must "
                        "be all None or all present"
                    )
                if all(present):
                    if count == 0:
                        raise ValueError(
                            f"{row_where}: mean, min, max and uncertainty "
                            "must be None when count is 0"
                        )
                    if not minimum <= mean <= maximum:
                        raise ValueError(
                            f"{row_where} must satisfy min <= mean <= max"
                        )
                if uncertainty is not None and uncertainty < 0:
                    raise ValueError(
                        f"{row_where}.uncertainty must be non-negative"
                    )

                row_index += 1

    return reference, layers, scenarios, elements, drivers, data


def compare_layer_scenarios(summary, *, min_layers: int = 1) -> dict:
    """Compare per-scenario layer aggregates against the first scenario.

    ``summary`` must be a complete :func:`aggregate_attribute_layers` result
    (schema ``climate-grid/ra-layer-summary-v1``) with exactly the keys
    ``schema, reference, layers, scenarios, elements, drivers, data`` in that
    order; every member is validated against that contract, including the
    ``reference`` and ``layers`` dicts (exactly the keys ``period, region``
    in that order, both non-empty str, with the ``(period, region)`` pairs
    unique across the reference and all layers), the flat
    scenario-then-element-then-driver ``data`` row order, each row's key
    order ``scenario, element, driver, count, mean, min, max, uncertainty``
    and the row invariants (``count`` a non-bool int between 0 and the
    number of layers; ``mean``, ``min``, ``max`` and ``uncertainty`` all
    ``None`` or all finite numbers, all ``None`` whenever ``count`` is 0,
    with ``count`` positive, ``min <= mean <= max`` and ``uncertainty``
    non-negative when present).
    ``scenarios`` must contain at least 2 items and ``min_layers`` must be
    a non-bool positive int.

    The first scenario is the reference.  For every remaining scenario (in
    scenario order), element (in element order) and driver (in driver
    order), its row is paired with the reference scenario's row for the
    same element and driver and ``count`` is the smaller of the two row
    counts.  When either row's ``count`` is below ``min_layers`` or either
    row's ``mean``/``uncertainty`` is ``None``, ``difference`` and
    ``uncertainty`` are both ``None``; otherwise they are
    ``mean - reference_mean`` and ``sqrt(u0 ** 2 + u1 ** 2)`` respectively.

    The returned mapping uses the key order ``schema, reference, scenarios,
    elements, drivers, data``; ``schema`` is ``climate-grid/ra-scenario-v1``,
    ``reference`` is the first scenario and ``scenarios`` lists the
    remaining scenarios, with ``elements`` and ``drivers`` echoing the
    input axes in their original order.  ``data`` is a flat list in
    scenario-then-element-then-driver order; each row uses the key order
    ``scenario, element, driver, count, difference, uncertainty``.
    ``count`` is an int and every output float is ``round(x, 12)`` with
    negative zero normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    (
        _reference,
        _layers,
        scenarios,
        elements,
        drivers,
        data,
    ) = _validate_rank_attribute_layer_summary(summary)

    if not isinstance(min_layers, int) or isinstance(min_layers, bool):
        raise TypeError("min_layers must be a non-bool int")
    if min_layers < 1:
        raise ValueError("min_layers must be positive")
    if len(scenarios) < 2:
        raise ValueError("summary.scenarios must contain at least 2 items")

    n_elements = len(elements)
    n_drivers = len(drivers)

    result_data = []
    for s_index, scenario in enumerate(scenarios[1:], start=1):
        for e_index, element in enumerate(elements):
            for d_index, driver in enumerate(drivers):
                reference_row = data[(0 * n_elements + e_index) * n_drivers + d_index]
                row = data[(s_index * n_elements + e_index) * n_drivers + d_index]

                count = min(reference_row["count"], row["count"])
                reference_mean = reference_row["mean"]
                mean = row["mean"]
                reference_uncertainty = reference_row["uncertainty"]
                uncertainty = row["uncertainty"]
                if (
                    reference_row["count"] < min_layers
                    or row["count"] < min_layers
                    or reference_mean is None
                    or mean is None
                    or reference_uncertainty is None
                    or uncertainty is None
                ):
                    difference = None
                    combined = None
                else:
                    difference = _round_output(mean - reference_mean)
                    combined = _round_output(
                        math.sqrt(
                            reference_uncertainty * reference_uncertainty
                            + uncertainty * uncertainty
                        )
                    )

                result_data.append(
                    {
                        "scenario": scenario,
                        "element": element,
                        "driver": driver,
                        "count": count,
                        "difference": difference,
                        "uncertainty": combined,
                    }
                )

    return {
        "schema": _RANK_ATTRIBUTE_SCENARIO_SCHEMA,
        "reference": scenarios[0],
        "scenarios": list(scenarios[1:]),
        "elements": list(elements),
        "drivers": list(drivers),
        "data": result_data,
    }


_RANK_ATTRIBUTE_SCENARIO_KEYS = (
    "schema",
    "reference",
    "scenarios",
    "elements",
    "drivers",
    "data",
)


def _validate_rank_attribute_scenario(
    comparison: Any, *, where: str = "comparison"
) -> tuple[str, list[str], list[str], list[str], list]:
    if not isinstance(comparison, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(comparison.keys()) != _RANK_ATTRIBUTE_SCENARIO_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, reference, "
            "scenarios, elements, drivers, data in order"
        )

    schema = comparison["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _RANK_ATTRIBUTE_SCENARIO_SCHEMA:
        raise ValueError(
            f"{where}.schema must be {_RANK_ATTRIBUTE_SCENARIO_SCHEMA!r}"
        )

    reference = comparison["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    scenarios = _validate_string_list(
        comparison["scenarios"], f"{where}.scenarios"
    )
    elements = _validate_string_list(
        comparison["elements"], f"{where}.elements"
    )
    drivers = _validate_string_list(
        comparison["drivers"], f"{where}.drivers"
    )

    data = comparison["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    expected_rows = len(scenarios) * len(elements) * len(drivers)
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "scenario/element/driver combination, in "
            "scenario-then-element-then-driver order)"
        )

    row_index = 0
    for scenario in scenarios:
        for element in elements:
            for driver in drivers:
                row_where = f"{where}.data[{row_index}]"
                row = data[row_index]
                if not isinstance(row, dict):
                    raise TypeError(f"{row_where} must be a dict")
                if tuple(row.keys()) != _RANK_ATTRIBUTE_SCENARIO_ROW_KEYS:
                    raise ValueError(
                        f"{row_where} must have exactly the keys scenario, "
                        "element, driver, count, difference, uncertainty "
                        "in order"
                    )

                row_scenario = row["scenario"]
                if not isinstance(row_scenario, str):
                    raise TypeError(f"{row_where}.scenario must be a str")
                if row_scenario != scenario:
                    raise ValueError(
                        f"{row_where}.scenario must be {scenario!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_element = row["element"]
                if not isinstance(row_element, str):
                    raise TypeError(f"{row_where}.element must be a str")
                if row_element != element:
                    raise ValueError(
                        f"{row_where}.element must be {element!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_driver = row["driver"]
                if not isinstance(row_driver, str):
                    raise TypeError(f"{row_where}.driver must be a str")
                if row_driver != driver:
                    raise ValueError(
                        f"{row_where}.driver must be {driver!r} for its "
                        "scenario-then-element-then-driver position"
                    )

                count = row["count"]
                if not isinstance(count, int) or isinstance(count, bool):
                    raise TypeError(f"{row_where}.count must be a non-bool int")
                if count < 0:
                    raise ValueError(
                        f"{row_where}.count must be non-negative"
                    )

                difference = row["difference"]
                uncertainty = row["uncertainty"]
                _validate_number(
                    difference, f"{row_where}.difference", nullable=True
                )
                _validate_number(
                    uncertainty, f"{row_where}.uncertainty", nullable=True
                )
                if (difference is None) != (uncertainty is None):
                    raise ValueError(
                        f"{row_where}: difference and uncertainty must be "
                        "both None or both present"
                    )
                if uncertainty is not None and uncertainty < 0:
                    raise ValueError(
                        f"{row_where}.uncertainty must be non-negative"
                    )

                row_index += 1

    return reference, scenarios, elements, drivers, data


_LAYER_SUM_SCHEMA = "climate-grid/rs-v1"
_LAYER_SUM_ITEM_KEYS = ("period", "region", "comparison")


def layer_sum(items, *, min_count: int = 1) -> dict:
    """Aggregate scenario-comparison differences across period/region layers.

    ``items`` must be a non-empty list of mappings, each with exactly the
    keys ``period, region, comparison`` in that order.  ``period`` and
    ``region`` are non-empty str and every ``(period, region)`` pair must
    be unique.  ``comparison`` is a complete :func:`compare_layer_scenarios`
    result (schema ``climate-grid/ra-scenario-v1``) with exactly the keys
    ``schema, reference, scenarios, elements, drivers, data`` in that
    order; every member is validated against that contract, including the
    non-empty str ``reference``, the flat
    scenario-then-element-then-driver ``data`` row order, each row's key
    order ``scenario, element, driver, count, difference, uncertainty``
    and the row invariants (``count`` a non-negative non-bool int;
    ``difference`` and ``uncertainty`` both ``None`` or both finite
    numbers, with ``uncertainty`` non-negative).  Every item's comparison
    must share the same ``reference`` and the same ``scenarios``,
    ``elements`` and ``drivers`` in the same order; all four are taken
    from the first item.  ``min_count`` must be a non-bool positive int.

    For every scenario (in scenario order), element (in element order) and
    driver (in driver order), the ``(difference, uncertainty)`` pairs
    whose members are both not ``None`` are collected in item order and
    ``count`` is their number ``n``.  When ``n`` is below ``min_count``,
    ``mean``, ``min``, ``max`` and ``uncertainty`` are all ``None``;
    otherwise they are ``sum(d) / n``, the smallest ``d``, the largest
    ``d`` and ``sqrt(sum(u ** 2)) / n`` respectively.

    The returned mapping uses the key order ``schema, layers, reference,
    scenarios, elements, drivers, data``; ``schema`` is
    ``climate-grid/rs-v1``, ``layers`` lists the ``{"period": ...,
    "region": ...}`` dicts in item order and ``reference``,
    ``scenarios``, ``elements`` and ``drivers`` echo the first item's
    comparison axes.  ``data`` is a flat list in
    scenario-then-element-then-driver order; each row uses the key order
    ``scenario, element, driver, count, mean, min, max, uncertainty``.
    ``count`` is a non-bool int and every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs
    are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if len(items) == 0:
        raise ValueError("items must be non-empty")

    layers: list[dict] = []
    validated: list[list] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, item in enumerate(items):
        where = f"items[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{where} must be a dict")
        if tuple(item.keys()) != _LAYER_SUM_ITEM_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys period, region, "
                "comparison in order"
            )

        period = item["period"]
        region = item["region"]
        if not isinstance(period, str):
            raise TypeError(f"{where}.period must be a str")
        if period == "":
            raise ValueError(f"{where}.period must be non-empty")
        if not isinstance(region, str):
            raise TypeError(f"{where}.region must be a str")
        if region == "":
            raise ValueError(f"{where}.region must be non-empty")
        if (period, region) in seen_pairs:
            raise ValueError(
                f"duplicate period/region pair: {(period, region)!r}"
            )
        seen_pairs.add((period, region))

        reference, scenarios, elements, drivers, data = (
            _validate_rank_attribute_scenario(
                item["comparison"], where=f"{where}.comparison"
            )
        )
        if not validated:
            first_reference = reference
            first_scenarios = scenarios
            first_elements = elements
            first_drivers = drivers
        else:
            if reference != first_reference:
                raise ValueError(
                    "all items must share the same comparison reference, "
                    "taken from the first item"
                )
            if scenarios != first_scenarios:
                raise ValueError(
                    "all items must share the same comparison scenarios in "
                    "the same order, taken from the first item"
                )
            if elements != first_elements:
                raise ValueError(
                    "all items must share the same comparison elements in "
                    "the same order, taken from the first item"
                )
            if drivers != first_drivers:
                raise ValueError(
                    "all items must share the same comparison drivers in "
                    "the same order, taken from the first item"
                )

        layers.append({"period": period, "region": region})
        validated.append(data)

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    n_elements = len(first_elements)
    n_drivers = len(first_drivers)

    result_data = []
    for s_index, scenario in enumerate(first_scenarios):
        for e_index, element in enumerate(first_elements):
            for d_index, driver in enumerate(first_drivers):
                row_offset = (
                    (s_index * n_elements + e_index) * n_drivers + d_index
                )
                pairs = []
                for data in validated:
                    row = data[row_offset]
                    difference = row["difference"]
                    uncertainty = row["uncertainty"]
                    if difference is not None and uncertainty is not None:
                        pairs.append((difference, uncertainty))

                count = len(pairs)
                if count < min_count:
                    mean = None
                    minimum = None
                    maximum = None
                    combined = None
                else:
                    mean = _round_output(
                        sum(d for d, _u in pairs) / count
                    )
                    minimum = _round_output(min(d for d, _u in pairs))
                    maximum = _round_output(max(d for d, _u in pairs))
                    combined = _round_output(
                        math.sqrt(sum(u * u for _d, u in pairs)) / count
                    )

                result_data.append(
                    {
                        "scenario": scenario,
                        "element": element,
                        "driver": driver,
                        "count": count,
                        "mean": mean,
                        "min": minimum,
                        "max": maximum,
                        "uncertainty": combined,
                    }
                )

    return {
        "schema": _LAYER_SUM_SCHEMA,
        "layers": layers,
        "reference": first_reference,
        "scenarios": list(first_scenarios),
        "elements": list(first_elements),
        "drivers": list(first_drivers),
        "data": result_data,
    }


_LAYER_SUM_RESULT_KEYS = (
    "schema",
    "layers",
    "reference",
    "scenarios",
    "elements",
    "drivers",
    "data",
)
_LAYER_SUM_LAYER_KEYS = ("period", "region")
_LAYER_SUM_RESULT_ROW_KEYS = (
    "scenario",
    "element",
    "driver",
    "count",
    "mean",
    "min",
    "max",
    "uncertainty",
)
_LAYER_TREND_SCHEMA = "climate-grid/lt-v1"


def _validate_layer_sum_result(
    summary: Any, *, where: str = "summary"
) -> tuple[str, list[str], list[str], list[str], list]:
    if not isinstance(summary, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(summary.keys()) != _LAYER_SUM_RESULT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, layers, "
            "reference, scenarios, elements, drivers, data in order"
        )

    schema = summary["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _LAYER_SUM_SCHEMA:
        raise ValueError(f"{where}.schema must be {_LAYER_SUM_SCHEMA!r}")

    layers = summary["layers"]
    if not isinstance(layers, list):
        raise TypeError(f"{where}.layers must be a list")
    if len(layers) == 0:
        raise ValueError(f"{where}.layers must be non-empty")
    seen_pairs: set[tuple[str, str]] = set()
    for index, layer in enumerate(layers):
        layer_where = f"{where}.layers[{index}]"
        if not isinstance(layer, dict):
            raise TypeError(f"{layer_where} must be a dict")
        if tuple(layer.keys()) != _LAYER_SUM_LAYER_KEYS:
            raise ValueError(
                f"{layer_where} must have exactly the keys period, region "
                "in order"
            )
        period = layer["period"]
        region = layer["region"]
        if not isinstance(period, str):
            raise TypeError(f"{layer_where}.period must be a str")
        if period == "":
            raise ValueError(f"{layer_where}.period must be non-empty")
        if not isinstance(region, str):
            raise TypeError(f"{layer_where}.region must be a str")
        if region == "":
            raise ValueError(f"{layer_where}.region must be non-empty")
        if (period, region) in seen_pairs:
            raise ValueError(
                f"duplicate period/region pair in {where}: "
                f"{(period, region)!r}"
            )
        seen_pairs.add((period, region))

    reference = summary["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    scenarios = _validate_string_list(
        summary["scenarios"], f"{where}.scenarios"
    )
    elements = _validate_string_list(
        summary["elements"], f"{where}.elements"
    )
    drivers = _validate_string_list(summary["drivers"], f"{where}.drivers")

    data = summary["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    expected_rows = len(scenarios) * len(elements) * len(drivers)
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "scenario/element/driver combination, in "
            "scenario-then-element-then-driver order)"
        )

    row_index = 0
    for scenario in scenarios:
        for element in elements:
            for driver in drivers:
                row_where = f"{where}.data[{row_index}]"
                row = data[row_index]
                if not isinstance(row, dict):
                    raise TypeError(f"{row_where} must be a dict")
                if tuple(row.keys()) != _LAYER_SUM_RESULT_ROW_KEYS:
                    raise ValueError(
                        f"{row_where} must have exactly the keys scenario, "
                        "element, driver, count, mean, min, max, uncertainty "
                        "in order"
                    )

                row_scenario = row["scenario"]
                if not isinstance(row_scenario, str):
                    raise TypeError(f"{row_where}.scenario must be a str")
                if row_scenario != scenario:
                    raise ValueError(
                        f"{row_where}.scenario must be {scenario!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_element = row["element"]
                if not isinstance(row_element, str):
                    raise TypeError(f"{row_where}.element must be a str")
                if row_element != element:
                    raise ValueError(
                        f"{row_where}.element must be {element!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_driver = row["driver"]
                if not isinstance(row_driver, str):
                    raise TypeError(f"{row_where}.driver must be a str")
                if row_driver != driver:
                    raise ValueError(
                        f"{row_where}.driver must be {driver!r} for its "
                        "scenario-then-element-then-driver position"
                    )

                count = row["count"]
                if not isinstance(count, int) or isinstance(count, bool):
                    raise TypeError(f"{row_where}.count must be a non-bool int")
                if count < 0:
                    raise ValueError(f"{row_where}.count must be non-negative")

                mean = row["mean"]
                minimum = row["min"]
                maximum = row["max"]
                uncertainty = row["uncertainty"]
                _validate_number(mean, f"{row_where}.mean", nullable=True)
                _validate_number(minimum, f"{row_where}.min", nullable=True)
                _validate_number(maximum, f"{row_where}.max", nullable=True)
                _validate_number(
                    uncertainty, f"{row_where}.uncertainty", nullable=True
                )
                if (
                    (mean is None)
                    != (minimum is None)
                    or (mean is None) != (maximum is None)
                    or (mean is None) != (uncertainty is None)
                ):
                    raise ValueError(
                        f"{row_where}: mean, min, max and uncertainty must be "
                        "all None or all present"
                    )
                if count == 0 and mean is not None:
                    raise ValueError(
                        f"{row_where}: mean, min, max and uncertainty must be "
                        "all None when count is 0"
                    )
                if uncertainty is not None and uncertainty < 0:
                    raise ValueError(
                        f"{row_where}.uncertainty must be non-negative"
                    )

                row_index += 1

    return reference, scenarios, elements, drivers, data


def layer_trend(summaries, years, *, min_points: int = 2) -> dict:
    """Fit linear trends to :func:`layer_sum` summaries across years.

    ``summaries`` must be a non-empty list of complete :func:`layer_sum`
    results (schema ``climate-grid/rs-v1``); every member is validated
    against that contract, including the key order
    ``schema, layers, reference, scenarios, elements, drivers, data``,
    the ``period, region`` layer entries, the flat
    scenario-then-element-then-driver ``data`` row order, each row's key
    order ``scenario, element, driver, count, mean, min, max,
    uncertainty`` and the row invariants (``count`` a non-negative
    non-bool int; ``mean``, ``min``, ``max`` and ``uncertainty`` all
    ``None`` or all finite numbers, all ``None`` whenever ``count`` is
    0, with ``uncertainty`` non-negative).
    Every summary must share the same ``reference`` and the same
    ``scenarios``, ``elements`` and ``drivers`` in the same order; all
    four are taken from the first summary.  ``years`` must be a list
    with one strictly increasing non-bool int per summary.
    ``min_points`` must be a non-bool int no smaller than ``2``.

    For every scenario (in scenario order), element (in element order)
    and driver (in driver order), the rows whose ``mean`` and
    ``uncertainty`` are both not ``None`` are collected in summary
    order and paired with the corresponding year; their number is
    ``n``, using year as ``x``, mean as ``y`` and uncertainty as ``u``.
    When ``n`` is below ``min_points``, ``slope`` and ``uncertainty``
    are both ``None``; otherwise, with ``x̄ = Σx/n`` and ``ȳ = Σy/n``,
    ``S = Σ(x−x̄)²``, ``slope`` is ``Σ(x−x̄)(y−ȳ)/S`` and
    ``uncertainty`` is ``√Σ((x−x̄)u)²/S``.

    The returned mapping uses the key order
    ``schema, years, reference, scenarios, elements, drivers, data``;
    ``schema`` is ``climate-grid/lt-v1`` and ``years`` echoes the input
    years.  ``data`` is a flat list in scenario-then-element-then-driver
    order; each row uses the key order
    ``scenario, element, driver, count, slope, uncertainty``.
    ``count`` is a non-bool int and every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs
    are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    if not isinstance(summaries, list):
        raise TypeError("summaries must be a list")
    if len(summaries) == 0:
        raise ValueError("summaries must be non-empty")

    validated: list[list] = []
    for index, summary in enumerate(summaries):
        reference, scenarios, elements, drivers, data = (
            _validate_layer_sum_result(summary, where=f"summaries[{index}]")
        )
        if not validated:
            first_reference = reference
            first_scenarios = scenarios
            first_elements = elements
            first_drivers = drivers
        else:
            if reference != first_reference:
                raise ValueError(
                    "all summaries must share the same reference, taken "
                    "from the first summary"
                )
            if scenarios != first_scenarios:
                raise ValueError(
                    "all summaries must share the same scenarios in the "
                    "same order, taken from the first summary"
                )
            if elements != first_elements:
                raise ValueError(
                    "all summaries must share the same elements in the "
                    "same order, taken from the first summary"
                )
            if drivers != first_drivers:
                raise ValueError(
                    "all summaries must share the same drivers in the same "
                    "order, taken from the first summary"
                )
        validated.append(data)

    if not isinstance(years, list):
        raise TypeError("years must be a list")
    if len(years) != len(summaries):
        raise ValueError("years must have one entry per summary")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"years[{index}] must be a non-bool int")
    for index in range(1, len(years)):
        if years[index] <= years[index - 1]:
            raise ValueError("years must be strictly increasing")

    if not isinstance(min_points, int) or isinstance(min_points, bool):
        raise TypeError("min_points must be a non-bool int")
    if min_points < 2:
        raise ValueError("min_points must be at least 2")

    n_elements = len(first_elements)
    n_drivers = len(first_drivers)

    result_data = []
    for s_index, scenario in enumerate(first_scenarios):
        for e_index, element in enumerate(first_elements):
            for d_index, driver in enumerate(first_drivers):
                row_offset = (
                    (s_index * n_elements + e_index) * n_drivers + d_index
                )
                points = []
                for summary_index, data in enumerate(validated):
                    row = data[row_offset]
                    mean = row["mean"]
                    uncertainty = row["uncertainty"]
                    if mean is not None and uncertainty is not None:
                        points.append(
                            (years[summary_index], mean, uncertainty)
                        )

                count = len(points)
                if count < min_points:
                    slope = None
                    trend_uncertainty = None
                else:
                    x_bar = sum(x for x, _y, _u in points) / count
                    y_bar = sum(y for _x, y, _u in points) / count
                    s_xx = sum((x - x_bar) * (x - x_bar) for x, _y, _u in points)
                    slope = _round_output(
                        sum(
                            (x - x_bar) * (y - y_bar)
                            for x, y, _u in points
                        )
                        / s_xx
                    )
                    trend_uncertainty = _round_output(
                        math.sqrt(
                            sum(
                                (x - x_bar) * u * (x - x_bar) * u
                                for x, _y, u in points
                            )
                        )
                        / s_xx
                    )

                result_data.append(
                    {
                        "scenario": scenario,
                        "element": element,
                        "driver": driver,
                        "count": count,
                        "slope": slope,
                        "uncertainty": trend_uncertainty,
                    }
                )

    return {
        "schema": _LAYER_TREND_SCHEMA,
        "years": list(years),
        "reference": first_reference,
        "scenarios": list(first_scenarios),
        "elements": list(first_elements),
        "drivers": list(first_drivers),
        "data": result_data,
    }


_LAYER_TREND_RESULT_KEYS = (
    "schema",
    "years",
    "reference",
    "scenarios",
    "elements",
    "drivers",
    "data",
)
_LAYER_TREND_ROW_KEYS = (
    "scenario",
    "element",
    "driver",
    "count",
    "slope",
    "uncertainty",
)
_LAYER_TREND_COMPARE_SCHEMA = "climate-grid/ltc-v1"


def _validate_layer_trend_result(
    trend: Any, *, where: str = "trend"
) -> tuple[list, str, list[str], list[str], list[str], list]:
    if not isinstance(trend, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(trend.keys()) != _LAYER_TREND_RESULT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, years, reference, "
            "scenarios, elements, drivers, data in order"
        )

    schema = trend["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _LAYER_TREND_SCHEMA:
        raise ValueError(f"{where}.schema must be {_LAYER_TREND_SCHEMA!r}")

    years = trend["years"]
    if not isinstance(years, list):
        raise TypeError(f"{where}.years must be a list")
    if len(years) == 0:
        raise ValueError(f"{where}.years must be non-empty")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"{where}.years[{index}] must be a non-bool int")
    for index in range(1, len(years)):
        if years[index] <= years[index - 1]:
            raise ValueError(f"{where}.years must be strictly increasing")

    reference = trend["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    scenarios = _validate_string_list(
        trend["scenarios"], f"{where}.scenarios"
    )
    elements = _validate_string_list(
        trend["elements"], f"{where}.elements"
    )
    drivers = _validate_string_list(trend["drivers"], f"{where}.drivers")

    data = trend["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    expected_rows = len(scenarios) * len(elements) * len(drivers)
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "scenario/element/driver combination, in "
            "scenario-then-element-then-driver order)"
        )

    n_years = len(years)
    row_index = 0
    for scenario in scenarios:
        for element in elements:
            for driver in drivers:
                row_where = f"{where}.data[{row_index}]"
                row = data[row_index]
                if not isinstance(row, dict):
                    raise TypeError(f"{row_where} must be a dict")
                if tuple(row.keys()) != _LAYER_TREND_ROW_KEYS:
                    raise ValueError(
                        f"{row_where} must have exactly the keys scenario, "
                        "element, driver, count, slope, uncertainty "
                        "in order"
                    )

                row_scenario = row["scenario"]
                if not isinstance(row_scenario, str):
                    raise TypeError(f"{row_where}.scenario must be a str")
                if row_scenario != scenario:
                    raise ValueError(
                        f"{row_where}.scenario must be {scenario!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_element = row["element"]
                if not isinstance(row_element, str):
                    raise TypeError(f"{row_where}.element must be a str")
                if row_element != element:
                    raise ValueError(
                        f"{row_where}.element must be {element!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_driver = row["driver"]
                if not isinstance(row_driver, str):
                    raise TypeError(f"{row_where}.driver must be a str")
                if row_driver != driver:
                    raise ValueError(
                        f"{row_where}.driver must be {driver!r} for its "
                        "scenario-then-element-then-driver position"
                    )

                count = row["count"]
                if not isinstance(count, int) or isinstance(count, bool):
                    raise TypeError(f"{row_where}.count must be a non-bool int")
                if count < 0 or count > n_years:
                    raise ValueError(
                        f"{row_where}.count must be between 0 and the "
                        "number of years"
                    )

                slope = row["slope"]
                uncertainty = row["uncertainty"]
                _validate_number(slope, f"{row_where}.slope", nullable=True)
                _validate_number(
                    uncertainty, f"{row_where}.uncertainty", nullable=True
                )
                if (slope is None) != (uncertainty is None):
                    raise ValueError(
                        f"{row_where}: slope and uncertainty must be "
                        "both None or both present"
                    )
                if uncertainty is not None and uncertainty < 0:
                    raise ValueError(
                        f"{row_where}.uncertainty must be non-negative"
                    )

                row_index += 1

    return years, reference, scenarios, elements, drivers, data


def compare_trends(trend) -> dict:
    """Compare per-scenario layer trends against the first scenario.

    ``trend`` must be a complete :func:`layer_trend` result (schema
    ``climate-grid/lt-v1``) with exactly the keys ``schema, years,
    reference, scenarios, elements, drivers, data`` in that order; every
    member is validated against that contract, including the non-empty
    strictly increasing non-bool int ``years``, the non-empty str
    ``reference``, the flat scenario-then-element-then-driver ``data``
    row order, each row's key order ``scenario, element, driver, count,
    slope, uncertainty`` and the row invariants (``count`` a non-bool
    int between 0 and the number of years; ``slope`` and ``uncertainty``
    both ``None`` or both finite numbers, with ``uncertainty``
    non-negative).  ``scenarios`` must contain at least 2 items.

    The first scenario is the reference.  For every remaining scenario
    (in scenario order), element (in element order) and driver (in
    driver order), its row is paired with the reference scenario's row
    for the same element and driver and ``count`` is the smaller of the
    two row counts.  When either row's ``count`` is below 2 or either
    row's ``slope``/``uncertainty`` is ``None``, ``difference`` and
    ``uncertainty`` are both ``None``; otherwise they are
    ``slope - reference_slope`` and ``sqrt(u0 ** 2 + u1 ** 2)``
    respectively.

    The returned mapping uses the key order ``schema, years, reference,
    scenarios, elements, drivers, data``; ``schema`` is
    ``climate-grid/ltc-v1``, ``years`` echoes the input years,
    ``reference`` is the first scenario and ``scenarios`` lists the
    remaining scenarios, with ``elements`` and ``drivers`` echoing the
    input axes in their original order.  ``data`` is a flat list in
    scenario-then-element-then-driver order; each row uses the key order
    ``scenario, element, driver, count, difference, uncertainty``.
    ``count`` is an int and every output float is ``round(x, 12)`` with
    negative zero normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    (
        years,
        _reference,
        scenarios,
        elements,
        drivers,
        data,
    ) = _validate_layer_trend_result(trend)

    if len(scenarios) < 2:
        raise ValueError("trend.scenarios must contain at least 2 items")

    n_elements = len(elements)
    n_drivers = len(drivers)

    result_data = []
    for s_index, scenario in enumerate(scenarios[1:], start=1):
        for e_index, element in enumerate(elements):
            for d_index, driver in enumerate(drivers):
                reference_row = data[(0 * n_elements + e_index) * n_drivers + d_index]
                row = data[(s_index * n_elements + e_index) * n_drivers + d_index]

                count = min(reference_row["count"], row["count"])
                reference_slope = reference_row["slope"]
                slope = row["slope"]
                reference_uncertainty = reference_row["uncertainty"]
                uncertainty = row["uncertainty"]
                if (
                    reference_row["count"] < 2
                    or row["count"] < 2
                    or reference_slope is None
                    or slope is None
                    or reference_uncertainty is None
                    or uncertainty is None
                ):
                    difference = None
                    combined = None
                else:
                    difference = _round_output(slope - reference_slope)
                    combined = _round_output(
                        math.sqrt(
                            reference_uncertainty * reference_uncertainty
                            + uncertainty * uncertainty
                        )
                    )

                result_data.append(
                    {
                        "scenario": scenario,
                        "element": element,
                        "driver": driver,
                        "count": count,
                        "difference": difference,
                        "uncertainty": combined,
                    }
                )

    return {
        "schema": _LAYER_TREND_COMPARE_SCHEMA,
        "years": list(years),
        "reference": scenarios[0],
        "scenarios": list(scenarios[1:]),
        "elements": list(elements),
        "drivers": list(drivers),
        "data": result_data,
    }


_LAYER_TREND_COMPARE_RESULT_KEYS = (
    "schema",
    "years",
    "reference",
    "scenarios",
    "elements",
    "drivers",
    "data",
)
_LAYER_TREND_COMPARE_ROW_KEYS = (
    "scenario",
    "element",
    "driver",
    "count",
    "difference",
    "uncertainty",
)
_LAYER_TREND_SUMMARY_SCHEMA = "climate-grid/ltc-summary-v1"


def _validate_layer_trend_compare(
    comparison: Any, *, where: str = "comparison"
) -> tuple[list, str, list[str], list[str], list[str], list]:
    if not isinstance(comparison, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(comparison.keys()) != _LAYER_TREND_COMPARE_RESULT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, years, reference, "
            "scenarios, elements, drivers, data in order"
        )

    schema = comparison["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _LAYER_TREND_COMPARE_SCHEMA:
        raise ValueError(f"{where}.schema must be {_LAYER_TREND_COMPARE_SCHEMA!r}")

    years = comparison["years"]
    if not isinstance(years, list):
        raise TypeError(f"{where}.years must be a list")
    if len(years) == 0:
        raise ValueError(f"{where}.years must be non-empty")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"{where}.years[{index}] must be a non-bool int")
    for index in range(1, len(years)):
        if years[index] <= years[index - 1]:
            raise ValueError(f"{where}.years must be strictly increasing")

    reference = comparison["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    scenarios = _validate_string_list(
        comparison["scenarios"], f"{where}.scenarios"
    )
    if reference in scenarios:
        raise ValueError(
            f"{where}.reference must not appear in {where}.scenarios"
        )
    elements = _validate_string_list(
        comparison["elements"], f"{where}.elements"
    )
    drivers = _validate_string_list(comparison["drivers"], f"{where}.drivers")

    data = comparison["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    expected_rows = len(scenarios) * len(elements) * len(drivers)
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "scenario/element/driver combination, in "
            "scenario-then-element-then-driver order)"
        )

    n_years = len(years)
    row_index = 0
    for scenario in scenarios:
        for element in elements:
            for driver in drivers:
                row_where = f"{where}.data[{row_index}]"
                row = data[row_index]
                if not isinstance(row, dict):
                    raise TypeError(f"{row_where} must be a dict")
                if tuple(row.keys()) != _LAYER_TREND_COMPARE_ROW_KEYS:
                    raise ValueError(
                        f"{row_where} must have exactly the keys scenario, "
                        "element, driver, count, difference, uncertainty "
                        "in order"
                    )

                row_scenario = row["scenario"]
                if not isinstance(row_scenario, str):
                    raise TypeError(f"{row_where}.scenario must be a str")
                if row_scenario != scenario:
                    raise ValueError(
                        f"{row_where}.scenario must be {scenario!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_element = row["element"]
                if not isinstance(row_element, str):
                    raise TypeError(f"{row_where}.element must be a str")
                if row_element != element:
                    raise ValueError(
                        f"{row_where}.element must be {element!r} for its "
                        "scenario-then-element-then-driver position"
                    )
                row_driver = row["driver"]
                if not isinstance(row_driver, str):
                    raise TypeError(f"{row_where}.driver must be a str")
                if row_driver != driver:
                    raise ValueError(
                        f"{row_where}.driver must be {driver!r} for its "
                        "scenario-then-element-then-driver position"
                    )

                count = row["count"]
                if not isinstance(count, int) or isinstance(count, bool):
                    raise TypeError(f"{row_where}.count must be a non-bool int")
                if count < 0 or count > n_years:
                    raise ValueError(
                        f"{row_where}.count must be between 0 and the "
                        "number of years"
                    )

                difference = row["difference"]
                uncertainty = row["uncertainty"]
                _validate_number(difference, f"{row_where}.difference", nullable=True)
                _validate_number(
                    uncertainty, f"{row_where}.uncertainty", nullable=True
                )
                if (difference is None) != (uncertainty is None):
                    raise ValueError(
                        f"{row_where}: difference and uncertainty must be "
                        "both None or both present"
                    )
                if uncertainty is not None and uncertainty < 0:
                    raise ValueError(
                        f"{row_where}.uncertainty must be non-negative"
                    )

                row_index += 1

    return years, reference, scenarios, elements, drivers, data


def trend_difference_summary(comparison, *, min_drivers: int = 1) -> dict:
    """Summarize per-scenario trend differences across drivers.

    ``comparison`` must be a complete :func:`compare_trends` result (schema
    ``climate-grid/ltc-v1``) with exactly the keys ``schema, years,
    reference, scenarios, elements, drivers, data`` in that order; every
    member is validated against that contract, including the non-empty
    strictly increasing non-bool int ``years``, the non-empty str
    ``reference`` (which must not appear in ``scenarios``), the flat
    scenario-then-element-then-driver ``data``
    row order, each row's key order ``scenario, element, driver, count,
    difference, uncertainty`` and the row invariants (``count`` a non-bool
    int between 0 and the number of years; ``difference`` and
    ``uncertainty`` both ``None`` or both finite numbers, with
    ``uncertainty`` non-negative).  ``min_drivers`` must be a non-bool
    positive int.

    For every scenario (in scenario order) and element (in element order),
    the ``(difference, uncertainty)`` pairs whose members are both not
    ``None`` are collected in driver order and ``count`` is their number
    ``n``.  When ``n`` is below ``min_drivers``, ``mean``, ``min``, ``max``
    and ``uncertainty`` are all ``None``; otherwise they are
    ``sum(d) / n``, the smallest ``d``, the largest ``d`` and
    ``sqrt(sum(u ** 2)) / n`` respectively.

    The returned mapping uses the key order ``schema, years, reference,
    scenarios, elements, drivers, data``; ``schema`` is
    ``climate-grid/ltc-summary-v1`` and ``years``, ``reference``,
    ``scenarios``, ``elements`` and ``drivers`` echo the input metadata in
    their original order.  ``data`` is a flat list in
    scenario-then-element order; each row uses the key order ``scenario,
    element, count, mean, min, max, uncertainty``.  ``count`` is an int
    and every output float is ``round(x, 12)`` with negative zero
    normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    (
        years,
        reference,
        scenarios,
        elements,
        drivers,
        data,
    ) = _validate_layer_trend_compare(comparison)

    if not isinstance(min_drivers, int) or isinstance(min_drivers, bool):
        raise TypeError("min_drivers must be a non-bool int")
    if min_drivers < 1:
        raise ValueError("min_drivers must be positive")

    n_elements = len(elements)
    n_drivers = len(drivers)

    result_data = []
    for s_index, scenario in enumerate(scenarios):
        for e_index, element in enumerate(elements):
            pairs = []
            for d_index in range(n_drivers):
                row = data[(s_index * n_elements + e_index) * n_drivers + d_index]
                difference = row["difference"]
                uncertainty = row["uncertainty"]
                if difference is not None and uncertainty is not None:
                    pairs.append((difference, uncertainty))

            count = len(pairs)
            if count < min_drivers:
                mean = None
                minimum = None
                maximum = None
                combined = None
            else:
                mean = _round_output(sum(d for d, _u in pairs) / count)
                minimum = _round_output(min(d for d, _u in pairs))
                maximum = _round_output(max(d for d, _u in pairs))
                combined = _round_output(
                    math.sqrt(sum(u * u for _d, u in pairs)) / count
                )

            result_data.append(
                {
                    "scenario": scenario,
                    "element": element,
                    "count": count,
                    "mean": mean,
                    "min": minimum,
                    "max": maximum,
                    "uncertainty": combined,
                }
            )

    return {
        "schema": _LAYER_TREND_SUMMARY_SCHEMA,
        "years": list(years),
        "reference": reference,
        "scenarios": list(scenarios),
        "elements": list(elements),
        "drivers": list(drivers),
        "data": result_data,
    }


_LAYER_TREND_SUMMARY_RESULT_KEYS = (
    "schema",
    "years",
    "reference",
    "scenarios",
    "elements",
    "drivers",
    "data",
)
_LAYER_TREND_SUMMARY_ROW_KEYS = (
    "scenario",
    "element",
    "count",
    "mean",
    "min",
    "max",
    "uncertainty",
)
_LAYER_TREND_RANK_SCHEMA = "climate-grid/ltc-rank-v1"


def _validate_layer_trend_summary(
    summary: Any, *, where: str = "summary"
) -> tuple[list, str, list[str], list[str], list[str], list]:
    if not isinstance(summary, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(summary.keys()) != _LAYER_TREND_SUMMARY_RESULT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, years, reference, "
            "scenarios, elements, drivers, data in order"
        )

    schema = summary["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _LAYER_TREND_SUMMARY_SCHEMA:
        raise ValueError(f"{where}.schema must be {_LAYER_TREND_SUMMARY_SCHEMA!r}")

    years = summary["years"]
    if not isinstance(years, list):
        raise TypeError(f"{where}.years must be a list")
    if len(years) == 0:
        raise ValueError(f"{where}.years must be non-empty")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"{where}.years[{index}] must be a non-bool int")
    for index in range(1, len(years)):
        if years[index] <= years[index - 1]:
            raise ValueError(f"{where}.years must be strictly increasing")

    reference = summary["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    scenarios = _validate_string_list(
        summary["scenarios"], f"{where}.scenarios"
    )
    if reference in scenarios:
        raise ValueError(
            f"{where}.reference must not appear in {where}.scenarios"
        )
    elements = _validate_string_list(
        summary["elements"], f"{where}.elements"
    )
    drivers = _validate_string_list(summary["drivers"], f"{where}.drivers")

    data = summary["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    expected_rows = len(scenarios) * len(elements)
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "scenario/element combination, in scenario-then-element order)"
        )

    n_drivers = len(drivers)
    row_index = 0
    for scenario in scenarios:
        for element in elements:
            row_where = f"{where}.data[{row_index}]"
            row = data[row_index]
            if not isinstance(row, dict):
                raise TypeError(f"{row_where} must be a dict")
            if tuple(row.keys()) != _LAYER_TREND_SUMMARY_ROW_KEYS:
                raise ValueError(
                    f"{row_where} must have exactly the keys scenario, "
                    "element, count, mean, min, max, uncertainty in order"
                )

            row_scenario = row["scenario"]
            if not isinstance(row_scenario, str):
                raise TypeError(f"{row_where}.scenario must be a str")
            if row_scenario != scenario:
                raise ValueError(
                    f"{row_where}.scenario must be {scenario!r} for its "
                    "scenario-then-element position"
                )
            row_element = row["element"]
            if not isinstance(row_element, str):
                raise TypeError(f"{row_where}.element must be a str")
            if row_element != element:
                raise ValueError(
                    f"{row_where}.element must be {element!r} for its "
                    "scenario-then-element position"
                )

            count = row["count"]
            if not isinstance(count, int) or isinstance(count, bool):
                raise TypeError(f"{row_where}.count must be a non-bool int")
            if count < 0 or count > n_drivers:
                raise ValueError(
                    f"{row_where}.count must be between 0 and the "
                    "number of drivers"
                )

            mean = row["mean"]
            minimum = row["min"]
            maximum = row["max"]
            uncertainty = row["uncertainty"]
            _validate_number(mean, f"{row_where}.mean", nullable=True)
            _validate_number(minimum, f"{row_where}.min", nullable=True)
            _validate_number(maximum, f"{row_where}.max", nullable=True)
            _validate_number(
                uncertainty, f"{row_where}.uncertainty", nullable=True
            )
            none_flags = (
                mean is None,
                minimum is None,
                maximum is None,
                uncertainty is None,
            )
            if any(none_flags) and not all(none_flags):
                raise ValueError(
                    f"{row_where}: mean, min, max and uncertainty must be "
                    "all None or all present"
                )
            if uncertainty is not None and uncertainty < 0:
                raise ValueError(
                    f"{row_where}.uncertainty must be non-negative"
                )

            row_index += 1

    return years, reference, scenarios, elements, drivers, data


def rank_trend_difference(summary, *, min_drivers: int = 1) -> dict:
    """Rank scenarios per element by summarized trend differences.

    ``summary`` must be a complete :func:`trend_difference_summary` result
    (schema ``climate-grid/ltc-summary-v1``) with exactly the keys
    ``schema, years, reference, scenarios, elements, drivers, data`` in
    that order; every member is validated against that contract, including
    the non-empty strictly increasing non-bool int ``years``, the
    non-empty str ``reference`` (which must not appear in ``scenarios``),
    the flat scenario-then-element ``data`` row order, each row's key
    order ``scenario, element, count, mean, min, max, uncertainty`` and
    the row invariants (``count`` a non-bool int between 0 and the number
    of drivers; ``mean``, ``min``, ``max`` and ``uncertainty`` all
    ``None`` or all finite numbers, with ``uncertainty`` non-negative).
    ``min_drivers`` must be a non-bool positive int.

    Rows are ranked per element (in element order) across the scenarios:
    ``coverage`` is the row's ``count`` divided by the number of drivers.
    A row whose ``count`` is at least ``min_drivers`` and whose ``mean``
    and ``uncertainty`` are both not ``None`` is valid and gets
    ``difference`` set to its ``mean``; the valid rows are ordered by
    descending ``difference`` with ties keeping the original ``scenarios``
    order and ``rank`` runs consecutively from 1.  Invalid rows get
    ``difference``, ``uncertainty`` and ``rank`` all ``None`` and follow
    the valid ones in their original scenario order.

    The returned mapping uses the key order ``schema, years, reference,
    scenarios, elements, drivers, data``; ``schema`` is
    ``climate-grid/ltc-rank-v1`` and ``years``, ``reference``,
    ``scenarios``, ``elements`` and ``drivers`` echo the input metadata in
    their original order.  ``data`` is a flat list of rows in
    element-then-rank order; each row uses the key order ``element,
    scenario, count, coverage, difference, uncertainty, rank``.
    ``count`` is an int, ``rank`` is an int or ``None`` and every output
    float is ``round(x, 12)`` with negative zero normalized to ``0.0``.
    Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    (
        years,
        reference,
        scenarios,
        elements,
        drivers,
        data,
    ) = _validate_layer_trend_summary(summary)

    if not isinstance(min_drivers, int) or isinstance(min_drivers, bool):
        raise TypeError("min_drivers must be a non-bool int")
    if min_drivers < 1:
        raise ValueError("min_drivers must be positive")

    n_elements = len(elements)
    n_drivers = len(drivers)

    result_data = []
    for e_index, element in enumerate(elements):
        rows = [
            data[s_index * n_elements + e_index]
            for s_index in range(len(scenarios))
        ]
        valid = []
        invalid = []
        for row in rows:
            if (
                row["count"] >= min_drivers
                and row["mean"] is not None
                and row["uncertainty"] is not None
            ):
                valid.append(row)
            else:
                invalid.append(row)
        valid.sort(key=lambda row: row["mean"], reverse=True)
        for rank, row in enumerate(valid, start=1):
            result_data.append(
                {
                    "element": element,
                    "scenario": row["scenario"],
                    "count": row["count"],
                    "coverage": _round_output(row["count"] / n_drivers),
                    "difference": _round_output(row["mean"]),
                    "uncertainty": _round_output(row["uncertainty"]),
                    "rank": rank,
                }
            )
        for row in invalid:
            result_data.append(
                {
                    "element": element,
                    "scenario": row["scenario"],
                    "count": row["count"],
                    "coverage": _round_output(row["count"] / n_drivers),
                    "difference": None,
                    "uncertainty": None,
                    "rank": None,
                }
            )

    return {
        "schema": _LAYER_TREND_RANK_SCHEMA,
        "years": list(years),
        "reference": reference,
        "scenarios": list(scenarios),
        "elements": list(elements),
        "drivers": list(drivers),
        "data": result_data,
    }


_LAYER_TREND_SELECT_SCHEMA = "climate-grid/ltc-select-v1"
_LAYER_TREND_RANK_RESULT_KEYS = (
    "schema",
    "years",
    "reference",
    "scenarios",
    "elements",
    "drivers",
    "data",
)
_LAYER_TREND_RANK_ROW_KEYS = (
    "element",
    "scenario",
    "count",
    "coverage",
    "difference",
    "uncertainty",
    "rank",
)


def _validate_layer_trend_rank(
    ranking: Any, *, where: str = "ranking"
) -> tuple[list, str, list[str], list[str], list[str], list]:
    if not isinstance(ranking, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(ranking.keys()) != _LAYER_TREND_RANK_RESULT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, years, reference, "
            "scenarios, elements, drivers, data in order"
        )

    schema = ranking["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _LAYER_TREND_RANK_SCHEMA:
        raise ValueError(f"{where}.schema must be {_LAYER_TREND_RANK_SCHEMA!r}")

    years = ranking["years"]
    if not isinstance(years, list):
        raise TypeError(f"{where}.years must be a list")
    if len(years) == 0:
        raise ValueError(f"{where}.years must be non-empty")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"{where}.years[{index}] must be a non-bool int")
    for index in range(1, len(years)):
        if years[index] <= years[index - 1]:
            raise ValueError(f"{where}.years must be strictly increasing")

    reference = ranking["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    scenarios = _validate_string_list(
        ranking["scenarios"], f"{where}.scenarios"
    )
    if reference in scenarios:
        raise ValueError(
            f"{where}.reference must not appear in {where}.scenarios"
        )
    elements = _validate_string_list(
        ranking["elements"], f"{where}.elements"
    )
    drivers = _validate_string_list(ranking["drivers"], f"{where}.drivers")

    data = ranking["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    expected_rows = len(elements) * len(scenarios)
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "element/scenario combination, in element-then-rank order)"
        )

    n_scenarios = len(scenarios)
    n_drivers = len(drivers)
    scenario_index = {
        scenario: index for index, scenario in enumerate(scenarios)
    }

    row_index = 0
    for element in elements:
        seen_scenarios: set[str] = set()
        valid_count = 0
        previous_difference = None
        previous_order = -1
        null_phase = False
        invalid_order = -1
        for _ in range(n_scenarios):
            row_where = f"{where}.data[{row_index}]"
            row = data[row_index]
            if not isinstance(row, dict):
                raise TypeError(f"{row_where} must be a dict")
            if tuple(row.keys()) != _LAYER_TREND_RANK_ROW_KEYS:
                raise ValueError(
                    f"{row_where} must have exactly the keys element, "
                    "scenario, count, coverage, difference, uncertainty, "
                    "rank in order"
                )

            row_element = row["element"]
            if not isinstance(row_element, str):
                raise TypeError(f"{row_where}.element must be a str")
            if row_element != element:
                raise ValueError(
                    f"{row_where}.element must be {element!r} for its "
                    "element-then-rank position"
                )

            row_scenario = row["scenario"]
            if not isinstance(row_scenario, str):
                raise TypeError(f"{row_where}.scenario must be a str")
            if row_scenario == "":
                raise ValueError(f"{row_where}.scenario must be non-empty")
            if row_scenario not in scenario_index:
                raise ValueError(
                    f"{row_where}.scenario {row_scenario!r} must appear in "
                    f"{where}.scenarios"
                )
            if row_scenario in seen_scenarios:
                raise ValueError(
                    f"{row_where}: scenario {row_scenario!r} appears more "
                    f"than once for element {element!r}"
                )
            seen_scenarios.add(row_scenario)
            scenario_order = scenario_index[row_scenario]

            count = row["count"]
            if not isinstance(count, int) or isinstance(count, bool):
                raise TypeError(f"{row_where}.count must be a non-bool int")
            if count < 0 or count > n_drivers:
                raise ValueError(
                    f"{row_where}.count must be between 0 and the number of "
                    "drivers"
                )

            coverage = row["coverage"]
            _validate_number(coverage, f"{row_where}.coverage", nullable=False)
            if coverage < 0.0 or coverage > 1.0:
                raise ValueError(
                    f"{row_where}.coverage must be between 0 and 1"
                )

            difference = row["difference"]
            uncertainty = row["uncertainty"]
            rank = row["rank"]
            if rank is None:
                null_phase = True
                if difference is not None or uncertainty is not None:
                    raise ValueError(
                        f"{row_where}: difference and uncertainty must be "
                        "None when rank is None"
                    )
                if scenario_order <= invalid_order:
                    raise ValueError(
                        f"{row_where}: unranked rows must keep the original "
                        "scenario order"
                    )
                invalid_order = scenario_order
            else:
                if null_phase:
                    raise ValueError(
                        f"{row_where}: ranked rows must precede rows with "
                        "rank None"
                    )
                if not isinstance(rank, int) or isinstance(rank, bool):
                    raise TypeError(f"{row_where}.rank must be a non-bool int or None")
                valid_count += 1
                if rank != valid_count:
                    raise ValueError(
                        f"{row_where}.rank must run consecutively from 1"
                    )
                _validate_number(
                    difference, f"{row_where}.difference", nullable=True
                )
                _validate_number(
                    uncertainty, f"{row_where}.uncertainty", nullable=True
                )
                if difference is None or uncertainty is None:
                    raise ValueError(
                        f"{row_where}: difference and uncertainty must be "
                        "present when rank is not None"
                    )
                if uncertainty < 0:
                    raise ValueError(
                        f"{row_where}.uncertainty must be non-negative"
                    )
                if (
                    previous_difference is not None
                    and difference > previous_difference
                ):
                    raise ValueError(
                        f"{row_where}: ranked rows must be ordered by "
                        "descending difference"
                    )
                if (
                    previous_difference is not None
                    and difference == previous_difference
                    and scenario_order <= previous_order
                ):
                    raise ValueError(
                        f"{row_where}: tied differences must keep the "
                        "original scenario order"
                    )
                previous_order = scenario_order
                previous_difference = difference

            row_index += 1

    return years, reference, scenarios, elements, drivers, data


def _validate_select_requests(requests: Any, elements: list[str]) -> list:
    where = "requests"
    if not isinstance(requests, list):
        raise TypeError(f"{where} must be a list")
    if len(requests) == 0:
        raise ValueError(f"{where} must be non-empty")

    seen_elements: set[str] = set()
    for index, request in enumerate(requests):
        request_where = f"{where}[{index}]"
        if not isinstance(request, dict):
            raise TypeError(f"{request_where} must be a dict")
        if tuple(request.keys()) != ("element", "top"):
            raise ValueError(
                f"{request_where} must have exactly the keys element, top "
                "in order"
            )

        element = request["element"]
        if not isinstance(element, str):
            raise TypeError(f"{request_where}.element must be a str")
        if element == "":
            raise ValueError(f"{request_where}.element must be non-empty")
        if element not in elements:
            raise ValueError(
                f"{request_where}.element {element!r} must appear in "
                "ranking.elements"
            )
        if element in seen_elements:
            raise ValueError(
                f"{request_where}: duplicate element {element!r}"
            )
        seen_elements.add(element)

        top = request["top"]
        if not isinstance(top, int) or isinstance(top, bool):
            raise TypeError(f"{request_where}.top must be a non-bool int")
        if top < 1:
            raise ValueError(f"{request_where}.top must be positive")

    return requests


def select_rank(ranking, requests, *, min_coverage: float = 0.0) -> dict:
    """Select top-ranked trend-difference scenarios per element request.

    ``ranking`` must be a complete :func:`rank_trend_difference` result
    (schema ``climate-grid/ltc-rank-v1``) with exactly the keys
    ``schema, years, reference, scenarios, elements, drivers, data`` in
    that order; every member is validated against that contract, including
    the non-empty strictly increasing non-bool int ``years``, the
    non-empty str ``reference`` (which must not appear in ``scenarios``),
    the flat element-then-rank ``data`` row order, each row's key order
    ``element, scenario, count, coverage, difference, uncertainty, rank``
    and the row invariants (``count`` a non-bool int between 0 and the
    number of drivers; ``coverage`` a finite number between 0 and 1;
    ranked rows run consecutively from 1, precede the ``rank`` ``None``
    rows and are ordered by descending difference with ties keeping the
    original scenario order; ``difference`` and ``uncertainty`` are
    finite numbers exactly when ``rank`` is not ``None``, with
    ``uncertainty`` non-negative; the unranked rows keep the original
    scenario order; every scenario appears exactly once per element).

    ``requests`` must be a non-empty list of mappings with exactly the
    keys ``element, top`` in order; ``element`` must be a distinct
    non-empty str that appears in ``ranking.elements`` and ``top`` must
    be a non-bool positive int.  ``min_coverage`` must be a finite
    non-bool number between 0 and 1.

    For each request (in request order), the element's ranking rows are
    scanned in ranking order and the rows whose ``rank`` is not ``None``,
    whose ``rank`` does not exceed ``top`` and whose ``coverage`` is at
    least ``min_coverage`` are selected.  ``count`` is the selected
    number ``n``.  When ``n`` is zero, ``mean`` and ``uncertainty`` are
    both ``None``; otherwise they are ``sum(difference) / n`` and
    ``sqrt(sum(uncertainty ** 2)) / n`` over the selected rows.

    The returned mapping uses the key order ``schema, years, reference,
    drivers, data``; ``schema`` is ``climate-grid/ltc-select-v1`` and
    ``years``, ``reference`` and ``drivers`` echo the ranking metadata.
    ``data`` follows the request order; each item uses the key order
    ``element, top, count, mean, uncertainty, rows`` and ``rows`` copies
    the six keys ``scenario, count, coverage, difference, uncertainty,
    rank`` from each selected ranking row in ranking order.  ``count``
    is an int and every output float is ``round(x, 12)`` with negative
    zero normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    (
        years,
        reference,
        scenarios,
        elements,
        drivers,
        data,
    ) = _validate_layer_trend_rank(ranking)

    _validate_select_requests(requests, elements)

    if not isinstance(min_coverage, (int, float)) or isinstance(
        min_coverage, bool
    ):
        raise TypeError("min_coverage must be a non-bool number")
    if not math.isfinite(min_coverage):
        raise ValueError("min_coverage must be finite")
    if min_coverage < 0.0 or min_coverage > 1.0:
        raise ValueError("min_coverage must be between 0 and 1")

    n_scenarios = len(scenarios)

    rows_by_element: dict[str, list] = {
        element: [
            data[e_index * n_scenarios + s_index]
            for s_index in range(n_scenarios)
        ]
        for e_index, element in enumerate(elements)
    }

    result_data = []
    for request in requests:
        element = request["element"]
        top = request["top"]

        selected = []
        for row in rows_by_element[element]:
            if (
                row["rank"] is not None
                and row["rank"] <= top
                and row["coverage"] >= min_coverage
            ):
                selected.append(row)

        n = len(selected)
        if n == 0:
            mean = None
            combined = None
        else:
            mean = _round_output(
                sum(row["difference"] for row in selected) / n
            )
            combined = _round_output(
                math.sqrt(sum(row["uncertainty"] ** 2 for row in selected))
                / n
            )

        result_rows = [
            {
                "scenario": row["scenario"],
                "count": row["count"],
                "coverage": _round_output(row["coverage"]),
                "difference": _round_output(row["difference"]),
                "uncertainty": _round_output(row["uncertainty"]),
                "rank": row["rank"],
            }
            for row in selected
        ]

        result_data.append(
            {
                "element": element,
                "top": top,
                "count": n,
                "mean": mean,
                "uncertainty": combined,
                "rows": result_rows,
            }
        )

    return {
        "schema": _LAYER_TREND_SELECT_SCHEMA,
        "years": list(years),
        "reference": reference,
        "drivers": list(drivers),
        "data": result_data,
    }


_LAYER_TREND_SELECT_RESULT_KEYS = (
    "schema",
    "years",
    "reference",
    "drivers",
    "data",
)
_LAYER_TREND_SELECT_ITEM_KEYS = (
    "element",
    "top",
    "count",
    "mean",
    "uncertainty",
    "rows",
)
_LAYER_TREND_SELECT_ROW_KEYS = (
    "scenario",
    "count",
    "coverage",
    "difference",
    "uncertainty",
    "rank",
)
_LAYER_TREND_SELECTION_SUMMARY_SCHEMA = "climate-grid/ltc-ss-v1"


def _validate_layer_trend_select(
    selection: Any, *, where: str = "selection"
) -> tuple[list, str, list[str], list]:
    if not isinstance(selection, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(selection.keys()) != _LAYER_TREND_SELECT_RESULT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, years, reference, "
            "drivers, data in order"
        )

    schema = selection["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _LAYER_TREND_SELECT_SCHEMA:
        raise ValueError(f"{where}.schema must be {_LAYER_TREND_SELECT_SCHEMA!r}")

    years = selection["years"]
    if not isinstance(years, list):
        raise TypeError(f"{where}.years must be a list")
    if len(years) == 0:
        raise ValueError(f"{where}.years must be non-empty")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"{where}.years[{index}] must be a non-bool int")
    for index in range(1, len(years)):
        if years[index] <= years[index - 1]:
            raise ValueError(f"{where}.years must be strictly increasing")

    reference = selection["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    drivers = _validate_string_list(selection["drivers"], f"{where}.drivers")
    n_drivers = len(drivers)

    data = selection["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    if len(data) == 0:
        raise ValueError(f"{where}.data must be non-empty")

    seen_elements: set[str] = set()
    for index, item in enumerate(data):
        item_where = f"{where}.data[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{item_where} must be a dict")
        if tuple(item.keys()) != _LAYER_TREND_SELECT_ITEM_KEYS:
            raise ValueError(
                f"{item_where} must have exactly the keys element, top, "
                "count, mean, uncertainty, rows in order"
            )

        element = item["element"]
        if not isinstance(element, str):
            raise TypeError(f"{item_where}.element must be a str")
        if element == "":
            raise ValueError(f"{item_where}.element must be non-empty")
        if element in seen_elements:
            raise ValueError(
                f"{item_where}: duplicate element {element!r}"
            )
        seen_elements.add(element)

        top = item["top"]
        if not isinstance(top, int) or isinstance(top, bool):
            raise TypeError(f"{item_where}.top must be a non-bool int")
        if top < 1:
            raise ValueError(f"{item_where}.top must be positive")

        count = item["count"]
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError(f"{item_where}.count must be a non-bool int")
        if count < 0:
            raise ValueError(f"{item_where}.count must be non-negative")

        mean = item["mean"]
        uncertainty = item["uncertainty"]
        _validate_number(mean, f"{item_where}.mean", nullable=True)
        _validate_number(uncertainty, f"{item_where}.uncertainty", nullable=True)
        if count == 0:
            if mean is not None or uncertainty is not None:
                raise ValueError(
                    f"{item_where}: mean and uncertainty must be None when "
                    "count is 0"
                )
        else:
            if mean is None or uncertainty is None:
                raise ValueError(
                    f"{item_where}: mean and uncertainty must be present "
                    "when count is not 0"
                )
            if uncertainty < 0:
                raise ValueError(
                    f"{item_where}.uncertainty must be non-negative"
                )

        rows = item["rows"]
        if not isinstance(rows, list):
            raise TypeError(f"{item_where}.rows must be a list")
        if len(rows) != count:
            raise ValueError(
                f"{item_where}.rows must have exactly count rows"
            )

        seen_scenarios: set[str] = set()
        previous_rank = 0
        for row_index, row in enumerate(rows):
            row_where = f"{item_where}.rows[{row_index}]"
            if not isinstance(row, dict):
                raise TypeError(f"{row_where} must be a dict")
            if tuple(row.keys()) != _LAYER_TREND_SELECT_ROW_KEYS:
                raise ValueError(
                    f"{row_where} must have exactly the keys scenario, "
                    "count, coverage, difference, uncertainty, rank in order"
                )

            scenario = row["scenario"]
            if not isinstance(scenario, str):
                raise TypeError(f"{row_where}.scenario must be a str")
            if scenario == "":
                raise ValueError(f"{row_where}.scenario must be non-empty")
            if scenario in seen_scenarios:
                raise ValueError(
                    f"{row_where}: scenario {scenario!r} appears more than "
                    "once for this element"
                )
            seen_scenarios.add(scenario)

            row_count = row["count"]
            if not isinstance(row_count, int) or isinstance(row_count, bool):
                raise TypeError(f"{row_where}.count must be a non-bool int")
            if row_count < 0 or row_count > n_drivers:
                raise ValueError(
                    f"{row_where}.count must be between 0 and the number of "
                    "drivers"
                )

            coverage = row["coverage"]
            _validate_number(coverage, f"{row_where}.coverage", nullable=False)
            if coverage < 0.0 or coverage > 1.0:
                raise ValueError(
                    f"{row_where}.coverage must be between 0 and 1"
                )

            _validate_number(
                row["difference"], f"{row_where}.difference", nullable=False
            )
            row_uncertainty = row["uncertainty"]
            _validate_number(
                row_uncertainty, f"{row_where}.uncertainty", nullable=False
            )
            if row_uncertainty < 0:
                raise ValueError(
                    f"{row_where}.uncertainty must be non-negative"
                )

            rank = row["rank"]
            if not isinstance(rank, int) or isinstance(rank, bool):
                raise TypeError(f"{row_where}.rank must be a non-bool int")
            if rank < 1:
                raise ValueError(f"{row_where}.rank must be positive")
            if rank > top:
                raise ValueError(
                    f"{row_where}.rank must not exceed the item's top"
                )
            if rank <= previous_rank:
                raise ValueError(
                    f"{row_where}: rows must be ordered by increasing rank"
                )
            previous_rank = rank

        if count > 0:
            expected_mean = _round_output(
                sum(row["difference"] for row in rows) / count
            )
            if mean != expected_mean:
                raise ValueError(
                    f"{item_where}.mean must be the mean of the rows' "
                    "differences"
                )
            expected_uncertainty = _round_output(
                math.sqrt(sum(row["uncertainty"] ** 2 for row in rows))
                / count
            )
            if uncertainty != expected_uncertainty:
                raise ValueError(
                    f"{item_where}.uncertainty must be "
                    "sqrt(sum(uncertainty ** 2)) / count over the rows"
                )

    return years, reference, drivers, data


def selection_summary(selection, *, min_elements: int = 1) -> dict:
    """Summarize selected scenarios across the requested elements.

    ``selection`` must be a complete :func:`select_rank` result (schema
    ``climate-grid/ltc-select-v1``) with exactly the keys ``schema, years,
    reference, drivers, data`` in that order; every member is validated
    against that contract, including the non-empty strictly increasing
    non-bool int ``years``, the non-empty str ``reference``, the non-empty
    ``data`` items (each with exactly the keys ``element, top, count,
    mean, uncertainty, rows`` in order, a distinct non-empty str
    ``element``, a non-bool positive int ``top``, a non-bool non-negative
    int ``count`` equal to the number of ``rows``, and ``mean`` and
    ``uncertainty`` both ``None`` exactly when ``count`` is 0; when
    ``count`` is positive they must equal, with the usual rounding, the
    mean of the rows' ``difference`` values and
    ``sqrt(sum(uncertainty ** 2)) / count`` over the rows, respectively,
    and ``uncertainty`` must be non-negative) and each row's key
    order ``scenario, count, coverage, difference, uncertainty, rank``
    with the row invariants (a distinct non-empty str ``scenario``;
    ``count`` a non-bool int between 0 and the number of drivers;
    ``coverage`` a finite number between 0 and 1; ``difference`` and
    ``uncertainty`` finite numbers with ``uncertainty`` non-negative;
    ``rank`` a non-bool positive int not exceeding the item's ``top``,
    strictly increasing along the rows).  ``min_elements`` must be a
    non-bool positive int.

    The items are scanned in ``data`` order; ``elements`` takes that
    element order and ``scenarios`` takes the order in which each
    scenario first appears in the scanned rows.  For each scenario the
    rows are collected in element order: with ``n`` collected rows,
    ``covered`` is ``n``, ``driver_count`` is ``sum(count)`` and
    ``coverage`` is ``sum(coverage) / n``.  When ``n`` is below
    ``min_elements``, ``mean`` and ``uncertainty`` are both ``None``;
    otherwise they are ``sum(difference) / n`` and
    ``sqrt(sum(uncertainty ** 2)) / n`` over the collected rows.

    The returned mapping uses the key order ``schema, years, reference,
    drivers, elements, scenarios, data``; ``schema`` is
    ``climate-grid/ltc-ss-v1`` and ``years``, ``reference`` and
    ``drivers`` echo the selection metadata.  ``data`` follows the
    ``scenarios`` order; each row uses the key order ``scenario,
    covered, driver_count, coverage, mean, uncertainty``.  ``covered``
    and ``driver_count`` are ints and every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs
    are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    years, reference, drivers, data = _validate_layer_trend_select(selection)

    if not isinstance(min_elements, int) or isinstance(min_elements, bool):
        raise TypeError("min_elements must be a non-bool int")
    if min_elements < 1:
        raise ValueError("min_elements must be positive")

    elements = [item["element"] for item in data]

    scenarios: list[str] = []
    rows_by_scenario: dict[str, list] = {}
    for item in data:
        for row in item["rows"]:
            scenario = row["scenario"]
            if scenario not in rows_by_scenario:
                scenarios.append(scenario)
                rows_by_scenario[scenario] = [None] * len(data)
    for e_index, item in enumerate(data):
        for row in item["rows"]:
            rows_by_scenario[row["scenario"]][e_index] = row

    result_data = []
    for scenario in scenarios:
        collected = [
            row for row in rows_by_scenario[scenario] if row is not None
        ]
        n = len(collected)
        driver_count = sum(row["count"] for row in collected)
        coverage = _round_output(
            sum(row["coverage"] for row in collected) / n
        )
        if n < min_elements:
            mean = None
            combined = None
        else:
            mean = _round_output(
                sum(row["difference"] for row in collected) / n
            )
            combined = _round_output(
                math.sqrt(sum(row["uncertainty"] ** 2 for row in collected))
                / n
            )
        result_data.append(
            {
                "scenario": scenario,
                "covered": n,
                "driver_count": driver_count,
                "coverage": coverage,
                "mean": mean,
                "uncertainty": combined,
            }
        )

    return {
        "schema": _LAYER_TREND_SELECTION_SUMMARY_SCHEMA,
        "years": list(years),
        "reference": reference,
        "drivers": list(drivers),
        "elements": elements,
        "scenarios": scenarios,
        "data": result_data,
    }


_REGION_REPORT_SCHEMA = "climate-grid/rr-v1"
_MULTI_WINDOW_SCHEMA = "climate-grid/multi-window-v1"
_REGION_REPORT_RESULT_KEYS = (
    "schema",
    "years",
    "reference",
    "drivers",
    "scenarios",
    "elements",
    "windows",
    "regions",
    "data",
)
_SELECTION_SUMMARY_RESULT_KEYS = (
    "schema",
    "years",
    "reference",
    "drivers",
    "elements",
    "scenarios",
    "data",
)
_SELECTION_SUMMARY_ROW_KEYS = (
    "scenario",
    "covered",
    "driver_count",
    "coverage",
    "mean",
    "uncertainty",
)
_MULTI_WINDOW_RESULT_KEYS = (
    "schema",
    "elements",
    "windows",
    "regions",
    "data",
)
_MULTI_WINDOW_ROW_KEYS = (
    "window",
    "region",
    "element",
    "count",
    "mean",
    "min",
    "max",
    "uncertainty",
)
_REGION_REPORT_ROW_KEYS = (
    "scenario",
    "window",
    "region",
    "element",
    "selection",
    "statistics",
)


def _validate_selection_summary(
    summary: Any, *, where: str = "summary"
) -> tuple[list, str, list[str], list[str], list[str], list]:
    if not isinstance(summary, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(summary.keys()) != _SELECTION_SUMMARY_RESULT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, years, reference, "
            "drivers, elements, scenarios, data in order"
        )

    schema = summary["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _LAYER_TREND_SELECTION_SUMMARY_SCHEMA:
        raise ValueError(
            f"{where}.schema must be "
            f"{_LAYER_TREND_SELECTION_SUMMARY_SCHEMA!r}"
        )

    years = summary["years"]
    if not isinstance(years, list):
        raise TypeError(f"{where}.years must be a list")
    if len(years) == 0:
        raise ValueError(f"{where}.years must be non-empty")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"{where}.years[{index}] must be a non-bool int")
    for index in range(1, len(years)):
        if years[index] <= years[index - 1]:
            raise ValueError(f"{where}.years must be strictly increasing")

    reference = summary["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    drivers = _validate_string_list(summary["drivers"], f"{where}.drivers")
    elements = _validate_string_list(
        summary["elements"], f"{where}.elements"
    )
    scenarios = _validate_string_list(
        summary["scenarios"], f"{where}.scenarios"
    )

    data = summary["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    if len(data) != len(scenarios):
        raise ValueError(
            f"{where}.data must have exactly one row per scenario, in "
            "scenarios order"
        )

    for row_index, (row, scenario) in enumerate(zip(data, scenarios)):
        row_where = f"{where}.data[{row_index}]"
        if not isinstance(row, dict):
            raise TypeError(f"{row_where} must be a dict")
        if tuple(row.keys()) != _SELECTION_SUMMARY_ROW_KEYS:
            raise ValueError(
                f"{row_where} must have exactly the keys scenario, covered, "
                "driver_count, coverage, mean, uncertainty in order"
            )

        row_scenario = row["scenario"]
        if not isinstance(row_scenario, str):
            raise TypeError(f"{row_where}.scenario must be a str")
        if row_scenario != scenario:
            raise ValueError(
                f"{row_where}.scenario must be {scenario!r} for its "
                "scenarios position"
            )

        covered = row["covered"]
        if not isinstance(covered, int) or isinstance(covered, bool):
            raise TypeError(f"{row_where}.covered must be a non-bool int")
        if covered < 1 or covered > len(elements):
            raise ValueError(
                f"{row_where}.covered must be between 1 and the number of "
                "elements"
            )

        driver_count = row["driver_count"]
        if not isinstance(driver_count, int) or isinstance(driver_count, bool):
            raise TypeError(
                f"{row_where}.driver_count must be a non-bool int"
            )
        if driver_count < 0 or driver_count > covered * len(drivers):
            raise ValueError(
                f"{row_where}.driver_count must be between 0 and covered "
                "times the number of drivers"
            )

        _validate_number(
            row["coverage"], f"{row_where}.coverage", nullable=False
        )
        if row["coverage"] < 0.0 or row["coverage"] > 1.0:
            raise ValueError(
                f"{row_where}.coverage must be between 0 and 1"
            )

        mean = row["mean"]
        uncertainty = row["uncertainty"]
        _validate_number(mean, f"{row_where}.mean", nullable=True)
        _validate_number(
            uncertainty, f"{row_where}.uncertainty", nullable=True
        )
        if (mean is None) != (uncertainty is None):
            raise ValueError(
                f"{row_where}: mean and uncertainty must be both None or "
                "both present"
            )
        if uncertainty is not None and uncertainty < 0:
            raise ValueError(
                f"{row_where}.uncertainty must be non-negative"
            )

    return years, reference, drivers, elements, scenarios, data


def _validate_multi_window(
    regional: Any, *, where: str = "regional"
) -> tuple[list, list, list[str], list]:
    if not isinstance(regional, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(regional.keys()) != _MULTI_WINDOW_RESULT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, elements, windows, "
            "regions, data in order"
        )

    schema = regional["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _MULTI_WINDOW_SCHEMA:
        raise ValueError(f"{where}.schema must be {_MULTI_WINDOW_SCHEMA!r}")

    elements = regional["elements"]
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

    windows = regional["windows"]
    if not isinstance(windows, list):
        raise TypeError(f"{where}.windows must be a list")
    if len(windows) == 0:
        raise ValueError(f"{where}.windows must be non-empty")
    seen_windows: set[str] = set()
    for index, window in enumerate(windows):
        window_where = f"{where}.windows[{index}]"
        if not isinstance(window, dict):
            raise TypeError(f"{window_where} must be a dict")
        if list(window.keys()) != ["name", "start", "end"]:
            raise ValueError(
                f"{window_where} must have exactly the keys name, start, "
                "end in order"
            )
        name = window["name"]
        if not isinstance(name, str):
            raise TypeError(f"{window_where}.name must be a str")
        if name == "":
            raise ValueError(f"{window_where}.name must be non-empty")
        if name in seen_windows:
            raise ValueError(f"duplicate {where}.windows name: {name!r}")
        seen_windows.add(name)
        start_day = _parse_date(window["start"], f"{window_where}.start")
        end_day = _parse_date(window["end"], f"{window_where}.end")
        if start_day > end_day:
            raise ValueError(
                f"{window_where}.start must be on or before "
                f"{window_where}.end"
            )

    regions = regional["regions"]
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

    data = regional["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    expected_rows = len(windows) * len(regions) * len(elements)
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "window/region/element combination, in window-then-region-then-"
            "element order)"
        )

    row_index = 0
    for window in windows:
        for region in regions:
            for element in elements:
                row_where = f"{where}.data[{row_index}]"
                row = data[row_index]
                if not isinstance(row, dict):
                    raise TypeError(f"{row_where} must be a dict")
                if tuple(row.keys()) != _MULTI_WINDOW_ROW_KEYS:
                    raise ValueError(
                        f"{row_where} must have exactly the keys window, "
                        "region, element, count, mean, min, max, "
                        "uncertainty in order"
                    )

                row_window = row["window"]
                if not isinstance(row_window, str):
                    raise TypeError(f"{row_where}.window must be a str")
                if row_window != window["name"]:
                    raise ValueError(
                        f"{row_where}.window must be {window['name']!r} for "
                        "its window-then-region-then-element position"
                    )
                row_region = row["region"]
                if not isinstance(row_region, str):
                    raise TypeError(f"{row_where}.region must be a str")
                if row_region != region:
                    raise ValueError(
                        f"{row_where}.region must be {region!r} for its "
                        "window-then-region-then-element position"
                    )
                row_element = row["element"]
                if not isinstance(row_element, str):
                    raise TypeError(f"{row_where}.element must be a str")
                if row_element != element:
                    raise ValueError(
                        f"{row_where}.element must be {element!r} for its "
                        "window-then-region-then-element position"
                    )

                count = row["count"]
                if not isinstance(count, int) or isinstance(count, bool):
                    raise TypeError(f"{row_where}.count must be a non-bool int")
                if count < 0:
                    raise ValueError(f"{row_where}.count must be non-negative")

                mean = row["mean"]
                minimum = row["min"]
                maximum = row["max"]
                uncertainty = row["uncertainty"]
                _validate_number(mean, f"{row_where}.mean", nullable=True)
                _validate_number(minimum, f"{row_where}.min", nullable=True)
                _validate_number(maximum, f"{row_where}.max", nullable=True)
                _validate_number(
                    uncertainty, f"{row_where}.uncertainty", nullable=True
                )
                none_flags = (
                    mean is None,
                    minimum is None,
                    maximum is None,
                    uncertainty is None,
                )
                if any(none_flags) and not all(none_flags):
                    raise ValueError(
                        f"{row_where}: mean, min, max and uncertainty must "
                        "be all None or all present"
                    )
                if count == 0 and mean is not None:
                    raise ValueError(
                        f"{row_where}: mean, min, max and uncertainty must "
                        "be None when count is 0"
                    )
                if minimum is not None and minimum > maximum:
                    raise ValueError(
                        f"{row_where}.min must be less than or equal to "
                        f"{row_where}.max"
                    )
                if uncertainty is not None and uncertainty < 0:
                    raise ValueError(
                        f"{row_where}.uncertainty must be non-negative"
                    )

                row_index += 1

    return elements, windows, regions, data


def region_report(summary, regional) -> dict:
    """Combine a selection summary with multi-window regional statistics.

    ``summary`` must be a complete :func:`selection_summary` result
    (schema ``climate-grid/ltc-ss-v1``) with exactly the keys ``schema,
    years, reference, drivers, elements, scenarios, data`` in that order;
    every member is validated against that contract, including the
    non-empty strictly increasing non-bool int ``years``, the non-empty
    str ``reference``, the non-empty unique str lists ``drivers``,
    ``elements`` and ``scenarios``, one ``data`` row per scenario in
    ``scenarios`` order with key order ``scenario, covered, driver_count,
    coverage, mean, uncertainty``, and the row invariants (``covered`` a
    non-bool int between 1 and the number of elements; ``driver_count`` a
    non-bool int between 0 and ``covered`` times the number of drivers;
    ``coverage`` a finite number between 0 and 1; ``mean`` and
    ``uncertainty`` both ``None`` or both finite numbers, with
    ``uncertainty`` non-negative).

    ``regional`` must be a complete
    :func:`climate_grid.regional.aggregate_multi_window` result (schema
    ``climate-grid/multi-window-v1``) with exactly the keys ``schema,
    elements, windows, regions, data`` in that order; every member is
    validated against that contract, including the non-empty unique str
    lists ``elements`` and ``regions``, the non-empty ``windows`` (each a
    dict with exactly the keys ``name, start, end`` in order, a unique
    non-empty str ``name`` and valid ``YYYY-MM-DD`` dates with
    ``start <= end``), and the flat window-then-region-then-element
    ``data`` rows, each with key order ``window, region, element, count,
    mean, min, max, uncertainty`` and its position's names, ``count`` a
    non-bool non-negative int and ``mean``, ``min``, ``max`` and
    ``uncertainty`` all ``None`` or all finite numbers (with
    ``uncertainty`` non-negative).

    The two inputs must list the same elements in the same order.

    The returned mapping uses the key order ``schema, years, reference,
    drivers, scenarios, elements, windows, regions, data``; ``schema`` is
    ``climate-grid/rr-v1`` and ``years``, ``reference``, ``drivers``,
    ``scenarios``, ``elements``, ``windows`` and ``regions`` echo the
    input metadata in their original order.  ``data`` is a flat list of
    rows in scenario-then-window-then-region-then-element order; each row
    uses the key order ``scenario, window, region, element, selection,
    statistics``, where ``selection`` is the corresponding ``summary``
    row copied as-is and ``statistics`` is the corresponding ``regional``
    row copied as-is.  Counts stay ints, floats are ``round(x, 12)`` with
    negative zero normalized to ``0.0`` (already applied by the copied
    rows), and the inputs are not modified.

    Raises ``TypeError`` for wrong container/item types and ``ValueError``
    for any other contract violation.
    """
    (
        years,
        reference,
        drivers,
        summary_elements,
        scenarios,
        summary_data,
    ) = _validate_selection_summary(summary)
    (
        regional_elements,
        windows,
        regions,
        regional_data,
    ) = _validate_multi_window(regional)

    if regional_elements != summary_elements:
        raise ValueError(
            "regional.elements must match summary.elements in the same "
            "order"
        )

    n_scenarios = len(scenarios)
    n_windows = len(windows)
    n_regions = len(regions)
    n_elements = len(summary_elements)

    result_data = []
    for s_index in range(n_scenarios):
        selection_row = summary_data[s_index]
        for w_index in range(n_windows):
            for r_index in range(n_regions):
                for e_index in range(n_elements):
                    statistics_row = regional_data[
                        (w_index * n_regions + r_index) * n_elements
                        + e_index
                    ]
                    result_data.append(
                        {
                            "scenario": scenarios[s_index],
                            "window": windows[w_index]["name"],
                            "region": regions[r_index],
                            "element": summary_elements[e_index],
                            "selection": dict(selection_row),
                            "statistics": dict(statistics_row),
                        }
                    )

    return {
        "schema": _REGION_REPORT_SCHEMA,
        "years": list(years),
        "reference": reference,
        "drivers": list(drivers),
        "scenarios": list(scenarios),
        "elements": list(summary_elements),
        "windows": windows,
        "regions": list(regions),
        "data": result_data,
    }


_REGION_REPORT_COMPARE_SCHEMA = "climate-grid/rr-compare-v1"
_REGION_REPORT_COMPARE_RESULT_KEYS = (
    "schema",
    "years",
    "reference",
    "baseline",
    "scenarios",
    "elements",
    "windows",
    "regions",
    "data",
)
_REGION_REPORT_COMPARE_ROW_KEYS = (
    "scenario",
    "window",
    "region",
    "element",
    "difference",
    "uncertainty",
)


def _validate_region_report(
    report: Any, *, where: str = "report"
) -> tuple[list, str, list[str], list[str], list[str], list, list[str], list]:
    if not isinstance(report, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(report.keys()) != _REGION_REPORT_RESULT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, years, reference, "
            "drivers, scenarios, elements, windows, regions, data in order"
        )

    schema = report["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _REGION_REPORT_SCHEMA:
        raise ValueError(f"{where}.schema must be {_REGION_REPORT_SCHEMA!r}")

    years = report["years"]
    if not isinstance(years, list):
        raise TypeError(f"{where}.years must be a list")
    if len(years) == 0:
        raise ValueError(f"{where}.years must be non-empty")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"{where}.years[{index}] must be a non-bool int")
    for index in range(1, len(years)):
        if years[index] <= years[index - 1]:
            raise ValueError(f"{where}.years must be strictly increasing")

    reference = report["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    drivers = _validate_string_list(report["drivers"], f"{where}.drivers")
    scenarios = _validate_string_list(
        report["scenarios"], f"{where}.scenarios", minimum=2
    )
    elements = _validate_string_list(
        report["elements"], f"{where}.elements"
    )

    windows = report["windows"]
    if not isinstance(windows, list):
        raise TypeError(f"{where}.windows must be a list")
    if len(windows) == 0:
        raise ValueError(f"{where}.windows must be non-empty")
    seen_windows: set[str] = set()
    for index, window in enumerate(windows):
        window_where = f"{where}.windows[{index}]"
        if not isinstance(window, dict):
            raise TypeError(f"{window_where} must be a dict")
        if list(window.keys()) != ["name", "start", "end"]:
            raise ValueError(
                f"{window_where} must have exactly the keys name, start, "
                "end in order"
            )
        name = window["name"]
        if not isinstance(name, str):
            raise TypeError(f"{window_where}.name must be a str")
        if name == "":
            raise ValueError(f"{window_where}.name must be non-empty")
        if name in seen_windows:
            raise ValueError(f"duplicate {where}.windows name: {name!r}")
        seen_windows.add(name)
        start_day = _parse_date(window["start"], f"{window_where}.start")
        end_day = _parse_date(window["end"], f"{window_where}.end")
        if start_day > end_day:
            raise ValueError(
                f"{window_where}.start must be on or before "
                f"{window_where}.end"
            )

    regions = _validate_string_list(report["regions"], f"{where}.regions")

    data = report["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    n_scenarios = len(scenarios)
    n_windows = len(windows)
    n_regions = len(regions)
    n_elements = len(elements)
    expected_rows = n_scenarios * n_windows * n_regions * n_elements
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "scenario/window/region/element combination, in scenario-then-"
            "window-then-region-then-element order)"
        )

    row_index = 0
    for s_index, scenario in enumerate(scenarios):
        for w_index, window in enumerate(windows):
            for r_index, region in enumerate(regions):
                for e_index, element in enumerate(elements):
                    row_where = f"{where}.data[{row_index}]"
                    row = data[row_index]
                    if not isinstance(row, dict):
                        raise TypeError(f"{row_where} must be a dict")
                    if tuple(row.keys()) != _REGION_REPORT_ROW_KEYS:
                        raise ValueError(
                            f"{row_where} must have exactly the keys "
                            "scenario, window, region, element, selection, "
                            "statistics in order"
                        )

                    row_scenario = row["scenario"]
                    if not isinstance(row_scenario, str):
                        raise TypeError(f"{row_where}.scenario must be a str")
                    if row_scenario != scenario:
                        raise ValueError(
                            f"{row_where}.scenario must be {scenario!r} for "
                            "its scenario-then-window-then-region-then-"
                            "element position"
                        )
                    row_window = row["window"]
                    if not isinstance(row_window, str):
                        raise TypeError(f"{row_where}.window must be a str")
                    if row_window != window["name"]:
                        raise ValueError(
                            f"{row_where}.window must be {window['name']!r} "
                            "for its scenario-then-window-then-region-then-"
                            "element position"
                        )
                    row_region = row["region"]
                    if not isinstance(row_region, str):
                        raise TypeError(f"{row_where}.region must be a str")
                    if row_region != region:
                        raise ValueError(
                            f"{row_where}.region must be {region!r} for its "
                            "scenario-then-window-then-region-then-element "
                            "position"
                        )
                    row_element = row["element"]
                    if not isinstance(row_element, str):
                        raise TypeError(f"{row_where}.element must be a str")
                    if row_element != element:
                        raise ValueError(
                            f"{row_where}.element must be {element!r} for "
                            "its scenario-then-window-then-region-then-"
                            "element position"
                        )

                    selection = row["selection"]
                    selection_where = f"{row_where}.selection"
                    if not isinstance(selection, dict):
                        raise TypeError(f"{selection_where} must be a dict")
                    if tuple(selection.keys()) != _SELECTION_SUMMARY_ROW_KEYS:
                        raise ValueError(
                            f"{selection_where} must have exactly the keys "
                            "scenario, covered, driver_count, coverage, mean, "
                            "uncertainty in order"
                        )
                    selection_scenario = selection["scenario"]
                    if not isinstance(selection_scenario, str):
                        raise TypeError(
                            f"{selection_where}.scenario must be a str"
                        )
                    if selection_scenario != scenario:
                        raise ValueError(
                            f"{selection_where}.scenario must be "
                            f"{scenario!r} for its scenario position"
                        )
                    covered = selection["covered"]
                    if not isinstance(covered, int) or isinstance(covered, bool):
                        raise TypeError(
                            f"{selection_where}.covered must be a non-bool int"
                        )
                    if covered < 1 or covered > n_elements:
                        raise ValueError(
                            f"{selection_where}.covered must be between 1 "
                            "and the number of elements"
                        )
                    driver_count = selection["driver_count"]
                    if (
                        not isinstance(driver_count, int)
                        or isinstance(driver_count, bool)
                    ):
                        raise TypeError(
                            f"{selection_where}.driver_count must be a "
                            "non-bool int"
                        )
                    if driver_count < 0 or driver_count > covered * len(drivers):
                        raise ValueError(
                            f"{selection_where}.driver_count must be between "
                            "0 and covered times the number of drivers"
                        )
                    _validate_number(
                        selection["coverage"],
                        f"{selection_where}.coverage",
                        nullable=False,
                    )
                    if selection["coverage"] < 0.0 or selection["coverage"] > 1.0:
                        raise ValueError(
                            f"{selection_where}.coverage must be between 0 "
                            "and 1"
                        )
                    selection_mean = selection["mean"]
                    selection_uncertainty = selection["uncertainty"]
                    _validate_number(
                        selection_mean,
                        f"{selection_where}.mean",
                        nullable=True,
                    )
                    _validate_number(
                        selection_uncertainty,
                        f"{selection_where}.uncertainty",
                        nullable=True,
                    )
                    if (selection_mean is None) != (
                        selection_uncertainty is None
                    ):
                        raise ValueError(
                            f"{selection_where}: mean and uncertainty must "
                            "be both None or both present"
                        )
                    if (
                        selection_uncertainty is not None
                        and selection_uncertainty < 0
                    ):
                        raise ValueError(
                            f"{selection_where}.uncertainty must be "
                            "non-negative"
                        )

                    statistics = row["statistics"]
                    statistics_where = f"{row_where}.statistics"
                    if not isinstance(statistics, dict):
                        raise TypeError(f"{statistics_where} must be a dict")
                    if tuple(statistics.keys()) != _MULTI_WINDOW_ROW_KEYS:
                        raise ValueError(
                            f"{statistics_where} must have exactly the keys "
                            "window, region, element, count, mean, min, max, "
                            "uncertainty in order"
                        )
                    statistics_window = statistics["window"]
                    if not isinstance(statistics_window, str):
                        raise TypeError(
                            f"{statistics_where}.window must be a str"
                        )
                    if statistics_window != window["name"]:
                        raise ValueError(
                            f"{statistics_where}.window must be "
                            f"{window['name']!r} for its window position"
                        )
                    statistics_region = statistics["region"]
                    if not isinstance(statistics_region, str):
                        raise TypeError(
                            f"{statistics_where}.region must be a str"
                        )
                    if statistics_region != region:
                        raise ValueError(
                            f"{statistics_where}.region must be {region!r} "
                            "for its region position"
                        )
                    statistics_element = statistics["element"]
                    if not isinstance(statistics_element, str):
                        raise TypeError(
                            f"{statistics_where}.element must be a str"
                        )
                    if statistics_element != element:
                        raise ValueError(
                            f"{statistics_where}.element must be "
                            f"{element!r} for its element position"
                        )
                    count = statistics["count"]
                    if not isinstance(count, int) or isinstance(count, bool):
                        raise TypeError(
                            f"{statistics_where}.count must be a non-bool int"
                        )
                    if count < 0:
                        raise ValueError(
                            f"{statistics_where}.count must be non-negative"
                        )
                    stat_mean = statistics["mean"]
                    stat_min = statistics["min"]
                    stat_max = statistics["max"]
                    stat_uncertainty = statistics["uncertainty"]
                    _validate_number(
                        stat_mean, f"{statistics_where}.mean", nullable=True
                    )
                    _validate_number(
                        stat_min, f"{statistics_where}.min", nullable=True
                    )
                    _validate_number(
                        stat_max, f"{statistics_where}.max", nullable=True
                    )
                    _validate_number(
                        stat_uncertainty,
                        f"{statistics_where}.uncertainty",
                        nullable=True,
                    )
                    none_flags = (
                        stat_mean is None,
                        stat_min is None,
                        stat_max is None,
                        stat_uncertainty is None,
                    )
                    if any(none_flags) and not all(none_flags):
                        raise ValueError(
                            f"{statistics_where}: mean, min, max and "
                            "uncertainty must be all None or all present"
                        )
                    if count == 0 and stat_mean is not None:
                        raise ValueError(
                            f"{statistics_where}: mean, min, max and "
                            "uncertainty must be None when count is 0"
                        )
                    if stat_min is not None and stat_min > stat_max:
                        raise ValueError(
                            f"{statistics_where}.min must be less than or "
                            f"equal to {statistics_where}.max"
                        )
                    if stat_uncertainty is not None and stat_uncertainty < 0:
                        raise ValueError(
                            f"{statistics_where}.uncertainty must be "
                            "non-negative"
                        )

                    row_index += 1

    return years, reference, drivers, scenarios, elements, windows, regions, data


def compare_region_report(report) -> dict:
    """Compare each scenario of a region report against the first scenario.

    ``report`` must be a complete :func:`region_report` result (schema
    ``climate-grid/rr-v1``) with exactly the keys ``schema, years,
    reference, drivers, scenarios, elements, windows, regions, data`` in
    that order; every member is validated against that contract,
    including the nested ``selection`` rows (key order ``scenario,
    covered, driver_count, coverage, mean, uncertainty``) and
    ``statistics`` rows (key order ``window, region, element, count,
    mean, min, max, uncertainty``) and their invariants.  ``scenarios``
    must contain at least 2 items.

    The first scenario is the baseline.  For every remaining scenario
    (in scenario order), window (in window order), region (in region
    order) and element (in element order), its row is paired with the
    baseline scenario's row for the same window, region and element.
    The current selection ``mean``/``uncertainty`` are ``s``/``u`` and
    the baseline selection ``mean``/``uncertainty`` are ``s0``/``u0``;
    the current statistics ``mean``/``uncertainty`` are ``r``/``v`` and
    the baseline statistics ``mean``/``uncertainty`` are ``r0``/``v0``.
    When any of those eight values is ``None``, ``difference`` and
    ``uncertainty`` are both ``None``; otherwise they are
    ``s * r - s0 * r0`` and
    ``hypot(hypot(r * u, s * v), hypot(r0 * u0, s0 * v0))``
    respectively.

    The returned mapping uses the key order ``schema, years, reference,
    baseline, scenarios, elements, windows, regions, data``; ``schema``
    is ``climate-grid/rr-compare-v1``, ``years``, ``reference``,
    ``elements``, ``windows`` and ``regions`` echo the input metadata
    in their original order, ``baseline`` is the first scenario and
    ``scenarios`` lists the remaining scenarios.  ``data`` is a flat
    list in scenario-then-window-then-region-then-element order; each
    row uses the key order ``scenario, window, region, element,
    difference, uncertainty``.  Every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  The
    input is not modified.

    Raises ``TypeError`` for wrong container/item types and
    ``ValueError`` for any other contract violation.
    """
    (
        years,
        reference,
        _drivers,
        scenarios,
        elements,
        windows,
        regions,
        data,
    ) = _validate_region_report(report)

    n_scenarios = len(scenarios)
    n_windows = len(windows)
    n_regions = len(regions)
    n_elements = len(elements)
    scenario_stride = n_windows * n_regions * n_elements

    def _row(s_index: int, w_index: int, r_index: int, e_index: int) -> dict:
        return data[
            s_index * scenario_stride
            + (w_index * n_regions + r_index) * n_elements
            + e_index
        ]

    result_data = []
    for s_index in range(1, n_scenarios):
        for w_index, window in enumerate(windows):
            for r_index, region in enumerate(regions):
                for e_index, element in enumerate(elements):
                    row = _row(s_index, w_index, r_index, e_index)
                    baseline_row = _row(0, w_index, r_index, e_index)

                    selection = row["selection"]
                    baseline_selection = baseline_row["selection"]
                    statistics = row["statistics"]
                    baseline_statistics = baseline_row["statistics"]

                    s = selection["mean"]
                    u = selection["uncertainty"]
                    s0 = baseline_selection["mean"]
                    u0 = baseline_selection["uncertainty"]
                    r = statistics["mean"]
                    v = statistics["uncertainty"]
                    r0 = baseline_statistics["mean"]
                    v0 = baseline_statistics["uncertainty"]

                    if (
                        s is None
                        or u is None
                        or s0 is None
                        or u0 is None
                        or r is None
                        or v is None
                        or r0 is None
                        or v0 is None
                    ):
                        difference = None
                        combined = None
                    else:
                        difference = _round_output(s * r - s0 * r0)
                        combined = _round_output(
                            math.hypot(
                                math.hypot(r * u, s * v),
                                math.hypot(r0 * u0, s0 * v0),
                            )
                        )

                    result_data.append(
                        {
                            "scenario": scenarios[s_index],
                            "window": window["name"],
                            "region": region,
                            "element": element,
                            "difference": difference,
                            "uncertainty": combined,
                        }
                    )

    return {
        "schema": _REGION_REPORT_COMPARE_SCHEMA,
        "years": list(years),
        "reference": reference,
        "baseline": scenarios[0],
        "scenarios": list(scenarios[1:]),
        "elements": list(elements),
        "windows": windows,
        "regions": list(regions),
        "data": result_data,
    }


_REGION_REPORT_SUMMARY_SCHEMA = "climate-grid/rr-summary-v1"


def _validate_region_comparison(
    comparison: Any, *, where: str = "comparison"
) -> tuple[list, str, str, list[str], list[str], list, list[str], list]:
    if not isinstance(comparison, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(comparison.keys()) != _REGION_REPORT_COMPARE_RESULT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, years, reference, "
            "baseline, scenarios, elements, windows, regions, data in order"
        )

    schema = comparison["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _REGION_REPORT_COMPARE_SCHEMA:
        raise ValueError(
            f"{where}.schema must be {_REGION_REPORT_COMPARE_SCHEMA!r}"
        )

    years = comparison["years"]
    if not isinstance(years, list):
        raise TypeError(f"{where}.years must be a list")
    if len(years) == 0:
        raise ValueError(f"{where}.years must be non-empty")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"{where}.years[{index}] must be a non-bool int")
    for index in range(1, len(years)):
        if years[index] <= years[index - 1]:
            raise ValueError(f"{where}.years must be strictly increasing")

    reference = comparison["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    baseline = comparison["baseline"]
    if not isinstance(baseline, str):
        raise TypeError(f"{where}.baseline must be a str")
    if baseline == "":
        raise ValueError(f"{where}.baseline must be non-empty")

    scenarios = _validate_string_list(
        comparison["scenarios"], f"{where}.scenarios"
    )
    if baseline in scenarios:
        raise ValueError(
            f"{where}.baseline must not appear in {where}.scenarios"
        )
    elements = _validate_string_list(
        comparison["elements"], f"{where}.elements"
    )

    windows = comparison["windows"]
    if not isinstance(windows, list):
        raise TypeError(f"{where}.windows must be a list")
    if len(windows) == 0:
        raise ValueError(f"{where}.windows must be non-empty")
    seen_windows: set[str] = set()
    for index, window in enumerate(windows):
        window_where = f"{where}.windows[{index}]"
        if not isinstance(window, dict):
            raise TypeError(f"{window_where} must be a dict")
        if list(window.keys()) != ["name", "start", "end"]:
            raise ValueError(
                f"{window_where} must have exactly the keys name, start, "
                "end in order"
            )
        name = window["name"]
        if not isinstance(name, str):
            raise TypeError(f"{window_where}.name must be a str")
        if name == "":
            raise ValueError(f"{window_where}.name must be non-empty")
        if name in seen_windows:
            raise ValueError(f"duplicate {where}.windows name: {name!r}")
        seen_windows.add(name)
        start_day = _parse_date(window["start"], f"{window_where}.start")
        end_day = _parse_date(window["end"], f"{window_where}.end")
        if start_day > end_day:
            raise ValueError(
                f"{window_where}.start must be on or before "
                f"{window_where}.end"
            )

    regions = _validate_string_list(comparison["regions"], f"{where}.regions")

    data = comparison["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    n_scenarios = len(scenarios)
    n_windows = len(windows)
    n_regions = len(regions)
    n_elements = len(elements)
    expected_rows = n_scenarios * n_windows * n_regions * n_elements
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "scenario/window/region/element combination, in scenario-then-"
            "window-then-region-then-element order)"
        )

    row_index = 0
    for s_index, scenario in enumerate(scenarios):
        for w_index, window in enumerate(windows):
            for r_index, region in enumerate(regions):
                for e_index, element in enumerate(elements):
                    row_where = f"{where}.data[{row_index}]"
                    row = data[row_index]
                    if not isinstance(row, dict):
                        raise TypeError(f"{row_where} must be a dict")
                    if tuple(row.keys()) != _REGION_REPORT_COMPARE_ROW_KEYS:
                        raise ValueError(
                            f"{row_where} must have exactly the keys "
                            "scenario, window, region, element, difference, "
                            "uncertainty in order"
                        )

                    row_scenario = row["scenario"]
                    if not isinstance(row_scenario, str):
                        raise TypeError(f"{row_where}.scenario must be a str")
                    if row_scenario != scenario:
                        raise ValueError(
                            f"{row_where}.scenario must be {scenario!r} for "
                            "its scenario-then-window-then-region-then-"
                            "element position"
                        )
                    row_window = row["window"]
                    if not isinstance(row_window, str):
                        raise TypeError(f"{row_where}.window must be a str")
                    if row_window != window["name"]:
                        raise ValueError(
                            f"{row_where}.window must be {window['name']!r} "
                            "for its scenario-then-window-then-region-then-"
                            "element position"
                        )
                    row_region = row["region"]
                    if not isinstance(row_region, str):
                        raise TypeError(f"{row_where}.region must be a str")
                    if row_region != region:
                        raise ValueError(
                            f"{row_where}.region must be {region!r} for its "
                            "scenario-then-window-then-region-then-element "
                            "position"
                        )
                    row_element = row["element"]
                    if not isinstance(row_element, str):
                        raise TypeError(f"{row_where}.element must be a str")
                    if row_element != element:
                        raise ValueError(
                            f"{row_where}.element must be {element!r} for "
                            "its scenario-then-window-then-region-then-"
                            "element position"
                        )

                    difference = row["difference"]
                    uncertainty = row["uncertainty"]
                    _validate_number(
                        difference, f"{row_where}.difference", nullable=True
                    )
                    _validate_number(
                        uncertainty, f"{row_where}.uncertainty", nullable=True
                    )
                    if (difference is None) != (uncertainty is None):
                        raise ValueError(
                            f"{row_where}: difference and uncertainty must "
                            "be both None or both present"
                        )
                    if uncertainty is not None and uncertainty < 0:
                        raise ValueError(
                            f"{row_where}.uncertainty must be non-negative"
                        )

                    row_index += 1

    return years, reference, baseline, scenarios, elements, windows, regions, data


def summarize_region_comparison(
    comparison, *, min_scenarios: int = 1
) -> dict:
    """Summarize a :func:`compare_region_report` result across scenarios.

    ``comparison`` must be a complete :func:`compare_region_report` result
    (schema ``climate-grid/rr-compare-v1``) with exactly the keys
    ``schema, years, reference, baseline, scenarios, elements, windows,
    regions, data`` in that order; every member is validated against that
    contract, including the flat ``data`` rows (key order ``scenario,
    window, region, element, difference, uncertainty``) and their
    invariants.

    ``min_scenarios`` must be a non-bool positive ``int``; a wrong type
    raises ``TypeError`` and a non-positive value raises ``ValueError``.

    For each window (in window order), region (in region order) and
    element (in element order), the scenario rows for that combination
    are scanned in scenario order, collecting the ``(difference,
    uncertainty)`` pairs for which both values are non-``None``; let
    ``n`` be their count.  When ``n < min_scenarios`` the row reports
    ``count`` ``n`` with ``mean``, ``min``, ``max`` and
    ``uncertainty`` all ``None``.  Otherwise they are the mean of the
    differences ``sum(d) / n``, ``min(d)``, ``max(d)`` and the combined
    uncertainty ``hypot(*u) / n``.

    The returned mapping uses the same key order as ``comparison``:
    ``schema, years, reference, baseline, scenarios, elements, windows,
    regions, data``; ``schema`` is ``climate-grid/rr-summary-v1`` and
    the other metadata members echo the input in their original order.
    ``data`` is a flat list in window-then-region-then-element order;
    each row uses the key order ``window, region, element, count, mean,
    min, max, uncertainty``.  ``count`` is an ``int`` and every output
    float is ``round(x, 12)`` with negative zero normalized to ``0.0``.
    The input is not modified.

    Raises ``TypeError`` for wrong container/item types and
    ``ValueError`` for any other contract violation.
    """
    (
        years,
        reference,
        baseline,
        scenarios,
        elements,
        windows,
        regions,
        data,
    ) = _validate_region_comparison(comparison)

    if not isinstance(min_scenarios, int) or isinstance(min_scenarios, bool):
        raise TypeError("min_scenarios must be a non-bool int")
    if min_scenarios < 1:
        raise ValueError("min_scenarios must be positive")

    n_scenarios = len(scenarios)
    n_windows = len(windows)
    n_regions = len(regions)
    n_elements = len(elements)
    scenario_stride = n_windows * n_regions * n_elements

    def _row(s_index: int, w_index: int, r_index: int, e_index: int) -> dict:
        return data[
            s_index * scenario_stride
            + (w_index * n_regions + r_index) * n_elements
            + e_index
        ]

    result_data = []
    for w_index, window in enumerate(windows):
        for r_index, region in enumerate(regions):
            for e_index, element in enumerate(elements):
                differences: list[float] = []
                uncertainties: list[float] = []
                for s_index in range(n_scenarios):
                    row = _row(s_index, w_index, r_index, e_index)
                    difference = row["difference"]
                    uncertainty = row["uncertainty"]
                    if difference is not None and uncertainty is not None:
                        differences.append(difference)
                        uncertainties.append(uncertainty)

                count = len(differences)
                if count < min_scenarios:
                    mean = None
                    minimum = None
                    maximum = None
                    combined = None
                else:
                    mean = _round_output(sum(differences) / count)
                    minimum = _round_output(min(differences))
                    maximum = _round_output(max(differences))
                    combined = _round_output(
                        math.hypot(*uncertainties) / count
                    )

                result_data.append(
                    {
                        "window": window["name"],
                        "region": region,
                        "element": element,
                        "count": count,
                        "mean": mean,
                        "min": minimum,
                        "max": maximum,
                        "uncertainty": combined,
                    }
                )

    return {
        "schema": _REGION_REPORT_SUMMARY_SCHEMA,
        "years": list(years),
        "reference": reference,
        "baseline": baseline,
        "scenarios": list(scenarios),
        "elements": list(elements),
        "windows": windows,
        "regions": list(regions),
        "data": result_data,
    }


_REGION_REPORT_AGGREGATE_SCHEMA = "climate-grid/rr-aggregate-v1"
_REGION_REPORT_SUMMARY_RESULT_KEYS = (
    "schema",
    "years",
    "reference",
    "baseline",
    "scenarios",
    "elements",
    "windows",
    "regions",
    "data",
)
_REGION_REPORT_SUMMARY_ROW_KEYS = (
    "window",
    "region",
    "element",
    "count",
    "mean",
    "min",
    "max",
    "uncertainty",
)


def _validate_region_summary(
    summary: Any, *, where: str = "summary"
) -> tuple[list, str, str, list[str], list[str], list, list[str], list]:
    if not isinstance(summary, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(summary.keys()) != _REGION_REPORT_SUMMARY_RESULT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, years, reference, "
            "baseline, scenarios, elements, windows, regions, data in order"
        )

    schema = summary["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _REGION_REPORT_SUMMARY_SCHEMA:
        raise ValueError(
            f"{where}.schema must be {_REGION_REPORT_SUMMARY_SCHEMA!r}"
        )

    years = summary["years"]
    if not isinstance(years, list):
        raise TypeError(f"{where}.years must be a list")
    if len(years) == 0:
        raise ValueError(f"{where}.years must be non-empty")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"{where}.years[{index}] must be a non-bool int")
    for index in range(1, len(years)):
        if years[index] <= years[index - 1]:
            raise ValueError(f"{where}.years must be strictly increasing")

    reference = summary["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    baseline = summary["baseline"]
    if not isinstance(baseline, str):
        raise TypeError(f"{where}.baseline must be a str")
    if baseline == "":
        raise ValueError(f"{where}.baseline must be non-empty")

    scenarios = _validate_string_list(
        summary["scenarios"], f"{where}.scenarios"
    )
    if baseline in scenarios:
        raise ValueError(
            f"{where}.baseline must not appear in {where}.scenarios"
        )
    elements = _validate_string_list(
        summary["elements"], f"{where}.elements"
    )

    windows = summary["windows"]
    if not isinstance(windows, list):
        raise TypeError(f"{where}.windows must be a list")
    if len(windows) == 0:
        raise ValueError(f"{where}.windows must be non-empty")
    seen_windows: set[str] = set()
    for index, window in enumerate(windows):
        window_where = f"{where}.windows[{index}]"
        if not isinstance(window, dict):
            raise TypeError(f"{window_where} must be a dict")
        if list(window.keys()) != ["name", "start", "end"]:
            raise ValueError(
                f"{window_where} must have exactly the keys name, start, "
                "end in order"
            )
        name = window["name"]
        if not isinstance(name, str):
            raise TypeError(f"{window_where}.name must be a str")
        if name == "":
            raise ValueError(f"{window_where}.name must be non-empty")
        if name in seen_windows:
            raise ValueError(f"duplicate {where}.windows name: {name!r}")
        seen_windows.add(name)
        start_day = _parse_date(window["start"], f"{window_where}.start")
        end_day = _parse_date(window["end"], f"{window_where}.end")
        if start_day > end_day:
            raise ValueError(
                f"{window_where}.start must be on or before "
                f"{window_where}.end"
            )

    regions = _validate_string_list(summary["regions"], f"{where}.regions")

    data = summary["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    n_scenarios = len(scenarios)
    n_windows = len(windows)
    n_regions = len(regions)
    n_elements = len(elements)
    expected_rows = n_windows * n_regions * n_elements
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "window/region/element combination, in window-then-region-then-"
            "element order)"
        )

    row_index = 0
    for w_index, window in enumerate(windows):
        for r_index, region in enumerate(regions):
            for e_index, element in enumerate(elements):
                row_where = f"{where}.data[{row_index}]"
                row = data[row_index]
                if not isinstance(row, dict):
                    raise TypeError(f"{row_where} must be a dict")
                if tuple(row.keys()) != _REGION_REPORT_SUMMARY_ROW_KEYS:
                    raise ValueError(
                        f"{row_where} must have exactly the keys window, "
                        "region, element, count, mean, min, max, uncertainty "
                        "in order"
                    )

                row_window = row["window"]
                if not isinstance(row_window, str):
                    raise TypeError(f"{row_where}.window must be a str")
                if row_window != window["name"]:
                    raise ValueError(
                        f"{row_where}.window must be {window['name']!r} for "
                        "its window-then-region-then-element position"
                    )
                row_region = row["region"]
                if not isinstance(row_region, str):
                    raise TypeError(f"{row_where}.region must be a str")
                if row_region != region:
                    raise ValueError(
                        f"{row_where}.region must be {region!r} for its "
                        "window-then-region-then-element position"
                    )
                row_element = row["element"]
                if not isinstance(row_element, str):
                    raise TypeError(f"{row_where}.element must be a str")
                if row_element != element:
                    raise ValueError(
                        f"{row_where}.element must be {element!r} for its "
                        "window-then-region-then-element position"
                    )

                count = row["count"]
                if not isinstance(count, int) or isinstance(count, bool):
                    raise TypeError(f"{row_where}.count must be a non-bool int")
                if count < 0 or count > n_scenarios:
                    raise ValueError(
                        f"{row_where}.count must be between 0 and the number "
                        f"of scenarios ({n_scenarios})"
                    )

                mean = row["mean"]
                minimum = row["min"]
                maximum = row["max"]
                uncertainty = row["uncertainty"]
                _validate_number(mean, f"{row_where}.mean", nullable=True)
                _validate_number(minimum, f"{row_where}.min", nullable=True)
                _validate_number(maximum, f"{row_where}.max", nullable=True)
                _validate_number(
                    uncertainty, f"{row_where}.uncertainty", nullable=True
                )
                stats_present = (
                    mean is not None,
                    minimum is not None,
                    maximum is not None,
                    uncertainty is not None,
                )
                if not (all(stats_present) or not any(stats_present)):
                    raise ValueError(
                        f"{row_where}: mean, min, max and uncertainty must "
                        "be all None or all present"
                    )
                if any(stats_present) and count == 0:
                    raise ValueError(
                        f"{row_where}.count must be positive when mean, min, "
                        "max and uncertainty are present"
                    )
                if uncertainty is not None and uncertainty < 0:
                    raise ValueError(
                        f"{row_where}.uncertainty must be non-negative"
                    )
                if mean is not None and not minimum <= mean <= maximum:
                    raise ValueError(
                        f"{row_where} must satisfy min <= mean <= max when "
                        "mean, min and max are present"
                    )

                row_index += 1

    return years, reference, baseline, scenarios, elements, windows, regions, data


def aggregate_region_summary(
    summary, *, min_windows: int = 1
) -> dict:
    """Aggregate a :func:`summarize_region_comparison` result across windows.

    ``summary`` must be a complete :func:`summarize_region_comparison`
    result (schema ``climate-grid/rr-summary-v1``) with exactly the keys
    ``schema, years, reference, baseline, scenarios, elements, windows,
    regions, data`` in that order; every member is validated against that
    contract, including the flat ``data`` rows (key order ``window,
    region, element, count, mean, min, max, uncertainty``) and their
    invariants (in particular, rows with statistics present must satisfy
    ``min <= mean <= max``).

    ``min_windows`` must be a non-bool positive ``int``; a wrong type
    raises ``TypeError`` and a non-positive value raises ``ValueError``.

    For each region (in region order) and element (in element order),
    the window rows for that combination are scanned in window order,
    collecting those whose ``mean``, ``min``, ``max`` and
    ``uncertainty`` are all non-``None``; let ``k`` be their count and
    ``N`` the sum of their per-row ``count`` values.  When
    ``k < min_windows`` the row reports ``window_count`` ``k`` and
    ``count`` ``N`` with ``mean``, ``min``, ``max`` and ``uncertainty``
    all ``None``.  Otherwise, writing ``a = count / N`` for each
    collected row, the row reports ``mean`` ``sum(a * mean)``, ``min``
    and ``max`` as the extrema across the collected rows, and
    ``uncertainty`` ``hypot(a * uncertainty, ...)``.

    The returned mapping uses the key order ``schema, years, reference,
    baseline, scenarios, elements, windows, regions, data``; ``schema``
    is ``climate-grid/rr-aggregate-v1`` and the other metadata members
    echo the input in their original order.  ``data`` is a nested
    mapping keyed by region then element (in region/element order); each
    row uses the key order ``region, element, window_count, count, mean,
    min, max, uncertainty``.  ``window_count`` and ``count`` are
    ``int`` values (``k`` and ``N``) and every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  The
    input is not modified.

    Raises ``TypeError`` for wrong container/item types and
    ``ValueError`` for any other contract violation.
    """
    (
        years,
        reference,
        baseline,
        scenarios,
        elements,
        windows,
        regions,
        data,
    ) = _validate_region_summary(summary)

    if not isinstance(min_windows, int) or isinstance(min_windows, bool):
        raise TypeError("min_windows must be a non-bool int")
    if min_windows < 1:
        raise ValueError("min_windows must be positive")

    n_windows = len(windows)
    n_regions = len(regions)
    n_elements = len(elements)

    def _row(w_index: int, r_index: int, e_index: int) -> dict:
        return data[(w_index * n_regions + r_index) * n_elements + e_index]

    result_data = {}
    for r_index, region in enumerate(regions):
        result_data[region] = {}
        for e_index, element in enumerate(elements):
            window_rows = []
            for w_index in range(n_windows):
                row = _row(w_index, r_index, e_index)
                if (
                    row["mean"] is not None
                    and row["min"] is not None
                    and row["max"] is not None
                    and row["uncertainty"] is not None
                ):
                    window_rows.append(row)

            window_count = len(window_rows)
            total_count = sum(row["count"] for row in window_rows)
            if window_count < min_windows:
                mean = None
                minimum = None
                maximum = None
                combined = None
            else:
                weights = [row["count"] / total_count for row in window_rows]
                mean = _round_output(
                    sum(
                        weight * row["mean"]
                        for weight, row in zip(weights, window_rows)
                    )
                )
                minimum = _round_output(
                    min(row["min"] for row in window_rows)
                )
                maximum = _round_output(
                    max(row["max"] for row in window_rows)
                )
                combined = _round_output(
                    math.hypot(
                        *(
                            weight * row["uncertainty"]
                            for weight, row in zip(weights, window_rows)
                        )
                    )
                )

            result_data[region][element] = {
                "region": region,
                "element": element,
                "window_count": window_count,
                "count": total_count,
                "mean": mean,
                "min": minimum,
                "max": maximum,
                "uncertainty": combined,
            }

    return {
        "schema": _REGION_REPORT_AGGREGATE_SCHEMA,
        "years": list(years),
        "reference": reference,
        "baseline": baseline,
        "scenarios": list(scenarios),
        "elements": list(elements),
        "windows": windows,
        "regions": list(regions),
        "data": result_data,
    }


_REGION_REPORT_INTERVAL_SCHEMA = "climate-grid/rr-interval-v1"
_REGION_REPORT_AGGREGATE_ROW_KEYS = (
    "region",
    "element",
    "window_count",
    "count",
    "mean",
    "min",
    "max",
    "uncertainty",
)


def _validate_region_aggregate(
    aggregate: Any, *, where: str = "aggregate"
) -> tuple[list, str, str, list[str], list[str], list, list[str], dict]:
    if not isinstance(aggregate, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(aggregate.keys()) != _REGION_REPORT_SUMMARY_RESULT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, years, reference, "
            "baseline, scenarios, elements, windows, regions, data in order"
        )

    schema = aggregate["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _REGION_REPORT_AGGREGATE_SCHEMA:
        raise ValueError(
            f"{where}.schema must be {_REGION_REPORT_AGGREGATE_SCHEMA!r}"
        )

    years = aggregate["years"]
    if not isinstance(years, list):
        raise TypeError(f"{where}.years must be a list")
    if len(years) == 0:
        raise ValueError(f"{where}.years must be non-empty")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"{where}.years[{index}] must be a non-bool int")
    for index in range(1, len(years)):
        if years[index] <= years[index - 1]:
            raise ValueError(f"{where}.years must be strictly increasing")

    reference = aggregate["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    baseline = aggregate["baseline"]
    if not isinstance(baseline, str):
        raise TypeError(f"{where}.baseline must be a str")
    if baseline == "":
        raise ValueError(f"{where}.baseline must be non-empty")

    scenarios = _validate_string_list(
        aggregate["scenarios"], f"{where}.scenarios"
    )
    if baseline in scenarios:
        raise ValueError(
            f"{where}.baseline must not appear in {where}.scenarios"
        )
    elements = _validate_string_list(
        aggregate["elements"], f"{where}.elements"
    )

    windows = aggregate["windows"]
    if not isinstance(windows, list):
        raise TypeError(f"{where}.windows must be a list")
    if len(windows) == 0:
        raise ValueError(f"{where}.windows must be non-empty")
    seen_windows: set[str] = set()
    for index, window in enumerate(windows):
        window_where = f"{where}.windows[{index}]"
        if not isinstance(window, dict):
            raise TypeError(f"{window_where} must be a dict")
        if list(window.keys()) != ["name", "start", "end"]:
            raise ValueError(
                f"{window_where} must have exactly the keys name, start, "
                "end in order"
            )
        name = window["name"]
        if not isinstance(name, str):
            raise TypeError(f"{window_where}.name must be a str")
        if name == "":
            raise ValueError(f"{window_where}.name must be non-empty")
        if name in seen_windows:
            raise ValueError(f"duplicate {where}.windows name: {name!r}")
        seen_windows.add(name)
        start_day = _parse_date(window["start"], f"{window_where}.start")
        end_day = _parse_date(window["end"], f"{window_where}.end")
        if start_day > end_day:
            raise ValueError(
                f"{window_where}.start must be on or before "
                f"{window_where}.end"
            )

    regions = _validate_string_list(aggregate["regions"], f"{where}.regions")

    data = aggregate["data"]
    if not isinstance(data, dict):
        raise TypeError(f"{where}.data must be a dict")
    if tuple(data.keys()) != tuple(regions):
        raise ValueError(
            f"{where}.data must be keyed by region, in {where}.regions "
            "order"
        )
    n_scenarios = len(scenarios)
    n_windows = len(windows)
    for region in regions:
        region_where = f"{where}.data[{region!r}]"
        by_element = data[region]
        if not isinstance(by_element, dict):
            raise TypeError(f"{region_where} must be a dict")
        if tuple(by_element.keys()) != tuple(elements):
            raise ValueError(
                f"{region_where} must be keyed by element, in "
                f"{where}.elements order"
            )
        for element in elements:
            row_where = f"{region_where}[{element!r}]"
            row = by_element[element]
            if not isinstance(row, dict):
                raise TypeError(f"{row_where} must be a dict")
            if tuple(row.keys()) != _REGION_REPORT_AGGREGATE_ROW_KEYS:
                raise ValueError(
                    f"{row_where} must have exactly the keys region, "
                    "element, window_count, count, mean, min, max, "
                    "uncertainty in order"
                )

            row_region = row["region"]
            if not isinstance(row_region, str):
                raise TypeError(f"{row_where}.region must be a str")
            if row_region != region:
                raise ValueError(
                    f"{row_where}.region must be {region!r} for its "
                    "region-then-element position"
                )
            row_element = row["element"]
            if not isinstance(row_element, str):
                raise TypeError(f"{row_where}.element must be a str")
            if row_element != element:
                raise ValueError(
                    f"{row_where}.element must be {element!r} for its "
                    "region-then-element position"
                )

            window_count = row["window_count"]
            if not isinstance(window_count, int) or isinstance(
                window_count, bool
            ):
                raise TypeError(
                    f"{row_where}.window_count must be a non-bool int"
                )
            if window_count < 0 or window_count > n_windows:
                raise ValueError(
                    f"{row_where}.window_count must be between 0 and the "
                    f"number of windows ({n_windows})"
                )

            count = row["count"]
            if not isinstance(count, int) or isinstance(count, bool):
                raise TypeError(f"{row_where}.count must be a non-bool int")
            if count < 0 or count > n_scenarios * n_windows:
                raise ValueError(
                    f"{row_where}.count must be between 0 and the number "
                    "of scenario/window combinations "
                    f"({n_scenarios * n_windows})"
                )

            mean = row["mean"]
            minimum = row["min"]
            maximum = row["max"]
            uncertainty = row["uncertainty"]
            _validate_number(mean, f"{row_where}.mean", nullable=True)
            _validate_number(minimum, f"{row_where}.min", nullable=True)
            _validate_number(maximum, f"{row_where}.max", nullable=True)
            _validate_number(
                uncertainty, f"{row_where}.uncertainty", nullable=True
            )
            stats_present = (
                mean is not None,
                minimum is not None,
                maximum is not None,
                uncertainty is not None,
            )
            if not (all(stats_present) or not any(stats_present)):
                raise ValueError(
                    f"{row_where}: mean, min, max and uncertainty must "
                    "be all None or all present"
                )
            if any(stats_present):
                if window_count == 0:
                    raise ValueError(
                        f"{row_where}.window_count must be positive when "
                        "mean, min, max and uncertainty are present"
                    )
                if count == 0:
                    raise ValueError(
                        f"{row_where}.count must be positive when mean, "
                        "min, max and uncertainty are present"
                    )
            if uncertainty is not None and uncertainty < 0:
                raise ValueError(
                    f"{row_where}.uncertainty must be non-negative"
                )
            if mean is not None and not minimum <= mean <= maximum:
                raise ValueError(
                    f"{row_where} must satisfy min <= mean <= max when "
                    "mean, min and max are present"
                )

    return years, reference, baseline, scenarios, elements, windows, regions, data


def region_summary_intervals(aggregate, factors) -> dict:
    """Expand an :func:`aggregate_region_summary` result into intervals.

    ``aggregate`` must be a complete :func:`aggregate_region_summary`
    result (schema ``climate-grid/rr-aggregate-v1``) with exactly the
    keys ``schema, years, reference, baseline, scenarios, elements,
    windows, regions, data`` in that order; every member is validated
    against that contract, including the nested ``data`` rows (key order
    ``region, element, window_count, count, mean, min, max,
    uncertainty``) and their invariants.

    ``factors`` must be a non-empty, strictly increasing list of finite,
    non-bool, positive ``int``/``float`` values; a wrong item type raises
    ``TypeError`` and any other violation raises ``ValueError``.  An
    ``int`` too large to be represented as a finite float counts as
    non-finite, as does any computed bound that overflows to a
    non-finite value.

    For each region (in region order) and element (in element order) one
    flat row is emitted with the key order ``region, element,
    window_count, count, center, uncertainty, lower, upper``.  Writing
    ``U`` for the row's ``uncertainty``: when ``mean`` or ``U`` is
    ``None`` the row reports ``center`` and ``uncertainty`` as ``None``
    and ``lower``/``upper`` as all-``None`` lists with one entry per
    factor; otherwise ``center`` is ``mean``, ``uncertainty`` is ``U``
    and, for each factor ``f``, ``lower``/``upper`` are ``mean - f * U``
    and ``mean + f * U``.  A non-finite computed bound raises
    ``ValueError``.

    The returned mapping uses the key order ``schema, years, reference,
    baseline, scenarios, elements, windows, regions, factors, data``;
    ``schema`` is ``climate-grid/rr-interval-v1`` and the other metadata
    members echo the input in their original order.  Every computed
    float is ``round(x, 12)`` with negative zero normalized to ``0.0``.
    The input is not modified.

    Raises ``TypeError`` for wrong container/item types and
    ``ValueError`` for any other contract violation.
    """
    (
        years,
        reference,
        baseline,
        scenarios,
        elements,
        windows,
        regions,
        data,
    ) = _validate_region_aggregate(aggregate)

    if not isinstance(factors, list):
        raise TypeError("factors must be a list")
    if len(factors) == 0:
        raise ValueError("factors must be non-empty")
    for index, factor in enumerate(factors):
        if not isinstance(factor, (int, float)) or isinstance(factor, bool):
            raise TypeError(
                f"factors[{index}] must be a finite non-bool int or float"
            )
        if not _is_finite(factor):
            raise ValueError(f"factors[{index}] must be finite")
        if factor <= 0:
            raise ValueError(f"factors[{index}] must be positive")
    for index in range(1, len(factors)):
        if factors[index] <= factors[index - 1]:
            raise ValueError("factors must be strictly increasing")

    def _bound(mean, factor, uncertainty, upper: bool, where: str) -> float:
        try:
            product = factor * uncertainty
            value = mean + product if upper else mean - product
        except OverflowError:
            # Huge int factors/means overflow float arithmetic.
            raise ValueError(f"{where} must be finite") from None
        if not _is_finite(value):
            raise ValueError(f"{where} must be finite")
        return _round_output(value)

    result_rows = []
    for region in regions:
        for element in elements:
            row = data[region][element]
            row_where = f"aggregate.data[{region!r}][{element!r}]"
            mean = row["mean"]
            uncertainty = row["uncertainty"]
            if mean is None or uncertainty is None:
                center = None
                out_uncertainty = None
                lower = [None] * len(factors)
                upper = [None] * len(factors)
            else:
                center = mean
                out_uncertainty = uncertainty
                lower = [
                    _bound(
                        mean,
                        factor,
                        uncertainty,
                        False,
                        f"{row_where}.lower[{index}]",
                    )
                    for index, factor in enumerate(factors)
                ]
                upper = [
                    _bound(
                        mean,
                        factor,
                        uncertainty,
                        True,
                        f"{row_where}.upper[{index}]",
                    )
                    for index, factor in enumerate(factors)
                ]
            result_rows.append(
                {
                    "region": region,
                    "element": element,
                    "window_count": row["window_count"],
                    "count": row["count"],
                    "center": center,
                    "uncertainty": out_uncertainty,
                    "lower": lower,
                    "upper": upper,
                }
            )

    return {
        "schema": _REGION_REPORT_INTERVAL_SCHEMA,
        "years": list(years),
        "reference": reference,
        "baseline": baseline,
        "scenarios": list(scenarios),
        "elements": list(elements),
        "windows": windows,
        "regions": list(regions),
        "factors": list(factors),
        "data": result_rows,
    }


_REGION_REPORT_INTERVAL_RESULT_KEYS = (
    "schema",
    "years",
    "reference",
    "baseline",
    "scenarios",
    "elements",
    "windows",
    "regions",
    "factors",
    "data",
)
_REGION_REPORT_INTERVAL_ROW_KEYS = (
    "region",
    "element",
    "window_count",
    "count",
    "center",
    "uncertainty",
    "lower",
    "upper",
)
_REGION_REPORT_INTERVAL_COMPARE_SCHEMA = "climate-grid/rr-interval-compare-v1"


def _validate_region_intervals(
    intervals: Any, *, where: str = "intervals"
) -> tuple[list, str, str, list[str], list[str], list, list[str], list, list]:
    if not isinstance(intervals, dict):
        raise TypeError(f"{where} must be a dict")
    if tuple(intervals.keys()) != _REGION_REPORT_INTERVAL_RESULT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, years, reference, "
            "baseline, scenarios, elements, windows, regions, factors, data "
            "in order"
        )

    schema = intervals["schema"]
    if not isinstance(schema, str):
        raise TypeError(f"{where}.schema must be a str")
    if schema != _REGION_REPORT_INTERVAL_SCHEMA:
        raise ValueError(
            f"{where}.schema must be {_REGION_REPORT_INTERVAL_SCHEMA!r}"
        )

    years = intervals["years"]
    if not isinstance(years, list):
        raise TypeError(f"{where}.years must be a list")
    if len(years) == 0:
        raise ValueError(f"{where}.years must be non-empty")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"{where}.years[{index}] must be a non-bool int")
    for index in range(1, len(years)):
        if years[index] <= years[index - 1]:
            raise ValueError(f"{where}.years must be strictly increasing")

    reference = intervals["reference"]
    if not isinstance(reference, str):
        raise TypeError(f"{where}.reference must be a str")
    if reference == "":
        raise ValueError(f"{where}.reference must be non-empty")

    baseline = intervals["baseline"]
    if not isinstance(baseline, str):
        raise TypeError(f"{where}.baseline must be a str")
    if baseline == "":
        raise ValueError(f"{where}.baseline must be non-empty")

    scenarios = _validate_string_list(
        intervals["scenarios"], f"{where}.scenarios"
    )
    if baseline in scenarios:
        raise ValueError(
            f"{where}.baseline must not appear in {where}.scenarios"
        )
    elements = _validate_string_list(
        intervals["elements"], f"{where}.elements"
    )

    windows = intervals["windows"]
    if not isinstance(windows, list):
        raise TypeError(f"{where}.windows must be a list")
    if len(windows) == 0:
        raise ValueError(f"{where}.windows must be non-empty")
    seen_windows: set[str] = set()
    for index, window in enumerate(windows):
        window_where = f"{where}.windows[{index}]"
        if not isinstance(window, dict):
            raise TypeError(f"{window_where} must be a dict")
        if list(window.keys()) != ["name", "start", "end"]:
            raise ValueError(
                f"{window_where} must have exactly the keys name, start, "
                "end in order"
            )
        name = window["name"]
        if not isinstance(name, str):
            raise TypeError(f"{window_where}.name must be a str")
        if name == "":
            raise ValueError(f"{window_where}.name must be non-empty")
        if name in seen_windows:
            raise ValueError(f"duplicate {where}.windows name: {name!r}")
        seen_windows.add(name)
        start_day = _parse_date(window["start"], f"{window_where}.start")
        end_day = _parse_date(window["end"], f"{window_where}.end")
        if start_day > end_day:
            raise ValueError(
                f"{window_where}.start must be on or before "
                f"{window_where}.end"
            )

    regions = _validate_string_list(intervals["regions"], f"{where}.regions")

    factors = intervals["factors"]
    if not isinstance(factors, list):
        raise TypeError(f"{where}.factors must be a list")
    if len(factors) == 0:
        raise ValueError(f"{where}.factors must be non-empty")
    for index, factor in enumerate(factors):
        if not isinstance(factor, (int, float)) or isinstance(factor, bool):
            raise TypeError(
                f"{where}.factors[{index}] must be a finite non-bool int "
                "or float"
            )
        if not _is_finite(factor):
            raise ValueError(f"{where}.factors[{index}] must be finite")
        if factor <= 0:
            raise ValueError(f"{where}.factors[{index}] must be positive")
    for index in range(1, len(factors)):
        if factors[index] <= factors[index - 1]:
            raise ValueError(f"{where}.factors must be strictly increasing")

    data = intervals["data"]
    if not isinstance(data, list):
        raise TypeError(f"{where}.data must be a list")
    n_scenarios = len(scenarios)
    n_windows = len(windows)
    n_factors = len(factors)
    expected_rows = len(regions) * len(elements)
    if len(data) != expected_rows:
        raise ValueError(
            f"{where}.data must have {expected_rows} rows (one per "
            "region/element combination, in region-then-element order)"
        )

    row_index = 0
    for region in regions:
        for element in elements:
            row_where = f"{where}.data[{row_index}]"
            row = data[row_index]
            if not isinstance(row, dict):
                raise TypeError(f"{row_where} must be a dict")
            if tuple(row.keys()) != _REGION_REPORT_INTERVAL_ROW_KEYS:
                raise ValueError(
                    f"{row_where} must have exactly the keys region, "
                    "element, window_count, count, center, uncertainty, "
                    "lower, upper in order"
                )

            row_region = row["region"]
            if not isinstance(row_region, str):
                raise TypeError(f"{row_where}.region must be a str")
            if row_region != region:
                raise ValueError(
                    f"{row_where}.region must be {region!r} for its "
                    "region-then-element position"
                )
            row_element = row["element"]
            if not isinstance(row_element, str):
                raise TypeError(f"{row_where}.element must be a str")
            if row_element != element:
                raise ValueError(
                    f"{row_where}.element must be {element!r} for its "
                    "region-then-element position"
                )

            window_count = row["window_count"]
            if not isinstance(window_count, int) or isinstance(
                window_count, bool
            ):
                raise TypeError(
                    f"{row_where}.window_count must be a non-bool int"
                )
            if window_count < 0 or window_count > n_windows:
                raise ValueError(
                    f"{row_where}.window_count must be between 0 and the "
                    f"number of windows ({n_windows})"
                )

            count = row["count"]
            if not isinstance(count, int) or isinstance(count, bool):
                raise TypeError(f"{row_where}.count must be a non-bool int")
            if count < 0 or count > n_scenarios * n_windows:
                raise ValueError(
                    f"{row_where}.count must be between 0 and the number "
                    "of scenario/window combinations "
                    f"({n_scenarios * n_windows})"
                )

            center = row["center"]
            uncertainty = row["uncertainty"]
            _validate_number(center, f"{row_where}.center", nullable=True)
            _validate_number(
                uncertainty, f"{row_where}.uncertainty", nullable=True
            )

            lower = row["lower"]
            upper = row["upper"]
            for key, bounds in (("lower", lower), ("upper", upper)):
                if not isinstance(bounds, list):
                    raise TypeError(f"{row_where}.{key} must be a list")
                if len(bounds) != n_factors:
                    raise ValueError(
                        f"{row_where}.{key} must have one entry per "
                        f"factor ({n_factors})"
                    )
                for index, bound in enumerate(bounds):
                    _validate_number(
                        bound, f"{row_where}.{key}[{index}]", nullable=True
                    )

            stats_present = (
                (center is not None,)
                + (uncertainty is not None,)
                + tuple(bound is not None for bound in lower)
                + tuple(bound is not None for bound in upper)
            )
            if not (all(stats_present) or not any(stats_present)):
                raise ValueError(
                    f"{row_where}: center, uncertainty, lower and upper "
                    "must be all None or all present"
                )
            if any(stats_present):
                if window_count == 0:
                    raise ValueError(
                        f"{row_where}.window_count must be positive when "
                        "center, uncertainty, lower and upper are present"
                    )
                if count == 0:
                    raise ValueError(
                        f"{row_where}.count must be positive when center, "
                        "uncertainty, lower and upper are present"
                    )
                if uncertainty < 0:
                    raise ValueError(
                        f"{row_where}.uncertainty must be non-negative"
                    )
                for index in range(n_factors):
                    if not lower[index] <= center <= upper[index]:
                        raise ValueError(
                            f"{row_where} must satisfy lower <= center <= "
                            "upper for every factor"
                        )

            row_index += 1

    return (
        years,
        reference,
        baseline,
        scenarios,
        elements,
        windows,
        regions,
        factors,
        data,
    )


def compare_region_intervals(intervals, pairs) -> dict:
    """Compare paired regions of a :func:`region_summary_intervals` result.

    ``intervals`` must be a complete :func:`region_summary_intervals`
    result (schema ``climate-grid/rr-interval-v1``) with exactly the keys
    ``schema, years, reference, baseline, scenarios, elements, windows,
    regions, factors, data`` in that order; every member is validated
    against that contract, including the flat ``data`` rows (key order
    ``region, element, window_count, count, center, uncertainty, lower,
    upper``) and their invariants.

    ``pairs`` must be a non-empty list; each item must be a dict with
    exactly the keys ``name, left, right`` in that order, all non-empty
    ``str`` values.  Pair names must be unique and ``left``/``right``
    must be two different names from ``intervals.regions``.  A wrong
    container/item type raises ``TypeError`` and any other violation
    raises ``ValueError``.

    For each pair (in pair order), element (in element order) and factor
    (in factor order) one flat row is emitted with the key order ``pair,
    left, right, element, factor, center_delta, lower_delta, upper_delta,
    overlap``.  Writing ``L``/``R`` for the left/right region rows: when
    either interval is missing (its statistics are ``None``) all four
    results are ``None``; otherwise ``center_delta`` is
    ``R.center - L.center``, ``lower_delta`` is ``R.lower - L.upper``,
    ``upper_delta`` is ``R.upper - L.lower`` (all at that factor) and
    ``overlap`` tells whether the two intervals intersect.  Deltas are
    ``float`` or ``None`` and ``overlap`` is ``bool`` or ``None``; a
    non-finite computed delta raises ``ValueError``.

    The returned mapping uses the key order ``schema, years, reference,
    baseline, scenarios, elements, windows, regions, factors, pairs,
    data``; ``schema`` is ``climate-grid/rr-interval-compare-v1`` and the
    other metadata members echo the input in their original order, with
    ``pairs`` inserted before ``data``.  Every computed float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  The
    input is not modified.

    Raises ``TypeError`` for wrong container/item types and
    ``ValueError`` for any other contract violation.
    """
    (
        years,
        reference,
        baseline,
        scenarios,
        elements,
        windows,
        regions,
        factors,
        data,
    ) = _validate_region_intervals(intervals)

    if not isinstance(pairs, list):
        raise TypeError("pairs must be a list")
    if len(pairs) == 0:
        raise ValueError("pairs must be non-empty")
    seen_pairs: set[str] = set()
    for index, pair in enumerate(pairs):
        pair_where = f"pairs[{index}]"
        if not isinstance(pair, dict):
            raise TypeError(f"{pair_where} must be a dict")
        if list(pair.keys()) != ["name", "left", "right"]:
            raise ValueError(
                f"{pair_where} must have exactly the keys name, left, "
                "right in order"
            )
        name = pair["name"]
        left = pair["left"]
        right = pair["right"]
        for key, value in (("name", name), ("left", left), ("right", right)):
            if not isinstance(value, str):
                raise TypeError(f"{pair_where}.{key} must be a str")
            if value == "":
                raise ValueError(f"{pair_where}.{key} must be non-empty")
        if name in seen_pairs:
            raise ValueError(f"duplicate pairs name: {name!r}")
        seen_pairs.add(name)
        for key, value in (("left", left), ("right", right)):
            if value not in regions:
                raise ValueError(
                    f"{pair_where}.{key} must be a name in "
                    "intervals.regions"
                )
        if left == right:
            raise ValueError(
                f"{pair_where}.left and {pair_where}.right must be "
                "different regions"
            )

    def _delta(right_value, left_value, where: str) -> float:
        try:
            value = right_value - left_value
        except OverflowError:
            # Huge int operands overflow float arithmetic.
            raise ValueError(f"{where} must be finite") from None
        if not _is_finite(value):
            raise ValueError(f"{where} must be finite")
        return float(_round_output(value))

    n_elements = len(elements)
    region_index = {region: index for index, region in enumerate(regions)}
    result_rows = []
    for pair in pairs:
        name = pair["name"]
        left = pair["left"]
        right = pair["right"]
        left_base = region_index[left] * n_elements
        right_base = region_index[right] * n_elements
        for e_index, element in enumerate(elements):
            left_row = data[left_base + e_index]
            right_row = data[right_base + e_index]
            for f_index, factor in enumerate(factors):
                delta_where = (
                    f"data row for pair {name!r}, element {element!r}, "
                    f"factor {factor!r}"
                )
                if (
                    left_row["center"] is None
                    or right_row["center"] is None
                ):
                    center_delta = None
                    lower_delta = None
                    upper_delta = None
                    overlap = None
                else:
                    center_delta = _delta(
                        right_row["center"],
                        left_row["center"],
                        f"{delta_where}.center_delta",
                    )
                    lower_delta = _delta(
                        right_row["lower"][f_index],
                        left_row["upper"][f_index],
                        f"{delta_where}.lower_delta",
                    )
                    upper_delta = _delta(
                        right_row["upper"][f_index],
                        left_row["lower"][f_index],
                        f"{delta_where}.upper_delta",
                    )
                    overlap = (
                        left_row["lower"][f_index]
                        <= right_row["upper"][f_index]
                        and right_row["lower"][f_index]
                        <= left_row["upper"][f_index]
                    )
                result_rows.append(
                    {
                        "pair": name,
                        "left": left,
                        "right": right,
                        "element": element,
                        "factor": factor,
                        "center_delta": center_delta,
                        "lower_delta": lower_delta,
                        "upper_delta": upper_delta,
                        "overlap": overlap,
                    }
                )

    return {
        "schema": _REGION_REPORT_INTERVAL_COMPARE_SCHEMA,
        "years": list(years),
        "reference": reference,
        "baseline": baseline,
        "scenarios": list(scenarios),
        "elements": list(elements),
        "windows": windows,
        "regions": list(regions),
        "factors": list(factors),
        "pairs": [
            {"name": pair["name"], "left": pair["left"], "right": pair["right"]}
            for pair in pairs
        ],
        "data": result_rows,
    }
