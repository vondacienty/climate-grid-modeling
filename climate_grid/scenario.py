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


_SCENARIO_COVERAGE_SCHEMA = "climate-grid/scenario-cov-v1"
_ENSEMBLE_MULTI_KEYS = frozenset(
    {"schema", "elements", "times", "lats", "lons", "quantiles", "data"}
)
_ENSEMBLE_MULTI_SERIES_KEYS = frozenset(
    {"values", "status", "uncertainty", "quantile_values"}
)
_REGION_KEYS = frozenset({"name", "cells"})


def _validate_ensemble_multi(
    ensemble: Any,
) -> tuple[list, list, list, list, dict]:
    if not isinstance(ensemble, dict):
        raise TypeError("ensemble must be a dict")
    if set(ensemble.keys()) != _ENSEMBLE_MULTI_KEYS:
        raise ValueError(
            "ensemble must have exactly the keys "
            "schema, elements, times, lats, lons, quantiles, data"
        )
    if not isinstance(ensemble["schema"], str):
        raise TypeError("ensemble.schema must be a str")
    if ensemble["schema"] != _ENSEMBLE_MULTI_SCHEMA:
        raise ValueError(f"ensemble.schema must be {_ENSEMBLE_MULTI_SCHEMA!r}")

    elements = ensemble["elements"]
    if not isinstance(elements, list):
        raise TypeError("ensemble.elements must be a list")
    if len(elements) == 0:
        raise ValueError("ensemble.elements must be non-empty")
    seen_elements: set[str] = set()
    for index, element in enumerate(elements):
        if not isinstance(element, str):
            raise TypeError(f"ensemble.elements[{index}] must be a str")
        if element == "":
            raise ValueError(f"ensemble.elements[{index}] must be non-empty")
        if element in seen_elements:
            raise ValueError(f"duplicate element: {element!r}")
        seen_elements.add(element)

    times = ensemble["times"]
    if not isinstance(times, list):
        raise TypeError("ensemble.times must be a list")
    if len(times) == 0:
        raise ValueError("ensemble.times must be non-empty")
    days = [
        _parse_date(value, f"ensemble.times[{index}]")
        for index, value in enumerate(times)
    ]
    for index in range(1, len(days)):
        if (days[index] - days[index - 1]).days != 1:
            raise ValueError(
                "ensemble.times must be strictly increasing consecutive days"
            )

    lats = ensemble["lats"]
    lons = ensemble["lons"]
    _validate_axis(lats, "ensemble.lats")
    _validate_axis(lons, "ensemble.lons")

    quantiles = ensemble["quantiles"]
    if not isinstance(quantiles, list):
        raise TypeError("ensemble.quantiles must be a list")
    if len(quantiles) == 0:
        raise ValueError("ensemble.quantiles must be non-empty")
    for index, q in enumerate(quantiles):
        if not isinstance(q, (int, float)) or isinstance(q, bool):
            raise TypeError(
                f"ensemble.quantiles[{index}] must be a finite non-bool int or float"
            )
        if not math.isfinite(q):
            raise ValueError(f"ensemble.quantiles[{index}] must be finite")
        if q < 0.0 or q > 1.0:
            raise ValueError(f"ensemble.quantiles[{index}] must be between 0 and 1")
    for index in range(1, len(quantiles)):
        if quantiles[index] <= quantiles[index - 1]:
            raise ValueError("ensemble.quantiles must be strictly increasing")

    data = ensemble["data"]
    if not isinstance(data, dict):
        raise TypeError("ensemble.data must be a dict")
    if len(data) == 0:
        raise ValueError("ensemble.data must be non-empty")
    if list(data.keys()) != elements:
        raise ValueError(
            "ensemble.data keys must match ensemble.elements exactly and in order"
        )

    n_times = len(times)
    n_lat = len(lats)
    n_lon = len(lons)
    n_quantiles = len(quantiles)
    for element in elements:
        where = f"ensemble.data[{element!r}]"
        series = data[element]
        if not isinstance(series, dict):
            raise TypeError(f"{where} must be a dict")
        if set(series.keys()) != _ENSEMBLE_MULTI_SERIES_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys "
                "values, status, uncertainty, quantile_values"
            )
        _validate_series(series, n_times, n_lat, n_lon, where)
        quantile_values = series["quantile_values"]
        if not isinstance(quantile_values, list):
            raise TypeError(f"{where}.quantile_values must be a list")
        if len(quantile_values) != n_quantiles:
            raise ValueError(
                f"{where}.quantile_values must have {n_quantiles} frames "
                "(one per quantile)"
            )
        for q_index, frame in enumerate(quantile_values):
            _validate_member(
                frame,
                n_times,
                n_lat,
                n_lon,
                f"{where}.quantile_values[{q_index}]",
                "values",
            )

    return times, lats, lons, list(elements), data


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
            raise ValueError(f"{where}.start must be within ensemble.times")
        if end not in day_index:
            raise ValueError(f"{where}.end must be within ensemble.times")
        if start_day > end_day:
            raise ValueError(f"{where}.start must be on or before {where}.end")

        validated.append((name, start, end, day_index[start], day_index[end]))

    return validated


def _coverage_window_region(
    status: list,
    uncertainty: list,
    cells: list[tuple[int, int]],
    t_start: int,
    t_end: int,
    min_count: int,
) -> dict:
    total = (t_end - t_start + 1) * len(cells)
    available = 0
    observed = 0
    interpolated = 0
    uncertainty_sq_sum = 0.0
    for t in range(t_start, t_end + 1):
        for i, j in cells:
            cell_status = status[t][i][j]
            if cell_status == "missing":
                continue
            available += 1
            if cell_status == "observed":
                observed += 1
            else:
                interpolated += 1
            uncertainty_sq_sum += uncertainty[t][i][j] ** 2

    if available < min_count:
        return {
            "total": total,
            "available": available,
            "observed": observed,
            "interpolated": interpolated,
            "rate": None,
            "observed_rate": None,
            "interpolated_rate": None,
            "uncertainty": None,
        }

    return {
        "total": total,
        "available": available,
        "observed": observed,
        "interpolated": interpolated,
        "rate": _round_output(available / total),
        "observed_rate": _round_output(observed / total),
        "interpolated_rate": _round_output(interpolated / total),
        "uncertainty": _round_output(math.sqrt(uncertainty_sq_sum) / available),
    }


def coverage_multi(ensemble, regions, windows, *, min_count: int = 1) -> dict:
    """Aggregate an ensemble-multi result into per-window coverage stats.

    ``ensemble`` must be an :func:`ensemble_multi` result (schema
    ``climate-grid/ensemble-multi-v1``) with the key order ``schema,
    elements, times, lats, lons, quantiles, data``; every member is
    validated against that contract, including the nested
    ``[time][lat][lon]`` ``values``, ``status`` and ``uncertainty`` grids and
    the ``[quantile][time][lat][lon]`` ``quantile_values`` grid.  Wrong
    container, name or item types raise ``TypeError``; any other contract
    violation raises ``ValueError``.

    ``regions``, ``windows`` and ``min_count`` follow exactly the same
    validation and exceptions as :func:`climate_grid.regional.aggregate_window`:
    regions are unique-name dicts whose ``cells`` are strictly ascending
    unique in-grid ``[i, j]`` pairs, windows are unique-name dicts with
    exactly the keys ``name, start, end`` whose dates occur in
    ``ensemble.times`` with ``start <= end``, and ``min_count`` is a positive
    non-bool int.

    For every window (in window order), region (in region order) and element
    (in ensemble element order), statistics are computed over the closed
    window-interval grid: ``total`` is the number of days in the inclusive
    interval times the number of cells in the region; ``available`` counts
    non-``missing`` cells, split into ``observed`` and ``interpolated``.
    When ``available`` is below ``min_count``, ``rate``, ``observed_rate``,
    ``interpolated_rate`` and ``uncertainty`` are all ``None``.  Otherwise
    the three rates are ``available / total``, ``observed / total`` and
    ``interpolated / total`` and ``uncertainty`` is
    ``sqrt(sum(u ** 2)) / available`` over the available cells'
    uncertainties.

    The returned mapping uses the key order ``schema, elements, windows,
    regions, data``; ``schema`` is ``climate-grid/scenario-cov-v1``,
    ``elements`` and ``windows`` echo the arguments and ``regions`` lists the
    region names in input order.  ``data`` is a flat list of rows in
    window-then-region-then-element order; each row uses the key order
    ``window, region, element, total, available, observed, interpolated,
    rate, observed_rate, interpolated_rate, uncertainty``.  The four counts
    are ints and every output float is ``round(x, 12)`` with negative zero
    normalized to ``0.0``.  Inputs are not modified.
    """
    times, lats, lons, elements, data = _validate_ensemble_multi(ensemble)

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
        "elements": elements,
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }
