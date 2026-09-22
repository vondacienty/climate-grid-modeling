"""Regional aggregation of reconstructed daily climate grids."""

from __future__ import annotations

import datetime
import math
import re
from typing import Any

_SCHEMA = "climate-grid/regional-v1"
_MULTI_SCHEMA = "climate-grid/regional-multi-v1"
_WINDOW_SCHEMA = "climate-grid/window-v1"
_MULTI_WINDOW_SCHEMA = "climate-grid/multi-window-v1"
_QUANTILE_SCHEMA = "climate-grid/quantile-window-v1"
_EXCEEDANCE_SCHEMA = "climate-grid/exceedance-window-v1"
_COVERAGE_SCHEMA = "climate-grid/coverage-window-v1"
_VARIANCE_SCHEMA = "climate-grid/variance-window-v1"
_TREND_SCHEMA = "climate-grid/trend-window-v1"
_CORRELATION_SCHEMA = "climate-grid/correlation-window-v1"
_REGRESSION_SCHEMA = "climate-grid/regression-window-v1"
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


def _variance_window_region(
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
            "variance": None,
            "stddev": None,
            "uncertainty": None,
        }

    mean = sum(window_values) / count
    variance = sum((v - mean) ** 2 for v in window_values) / count
    return {
        "count": count,
        "mean": _round_output(mean),
        "variance": _round_output(variance),
        "stddev": _round_output(math.sqrt(variance)),
        "uncertainty": _round_output(
            math.sqrt(sum(u ** 2 for u in window_uncertainties)) / count
        ),
    }


def _trend_window_region(
    values: list,
    status: list,
    uncertainty: list,
    cells: list[tuple[int, int]],
    t_start: int,
    t_end: int,
    min_count: int,
) -> dict:
    day_offsets: list[int] = []
    day_means: list[float] = []
    day_uncertainties: list[float] = []
    for t in range(t_start, t_end + 1):
        kept_values = []
        kept_uncertainties = []
        for i, j in cells:
            if status[t][i][j] != "missing":
                kept_values.append(values[t][i][j])
                kept_uncertainties.append(uncertainty[t][i][j])
        n = len(kept_values)
        if n < min_count:
            continue
        day_offsets.append(t - t_start)
        day_means.append(sum(kept_values) / n)
        day_uncertainties.append(
            math.sqrt(sum(u ** 2 for u in kept_uncertainties)) / n
        )

    count = len(day_offsets)
    if count < 2:
        return {
            "count": count,
            "mean": None,
            "slope": None,
            "intercept": None,
            "uncertainty": None,
        }

    x_mean = sum(day_offsets) / count
    m_mean = sum(day_means) / count
    denominator = sum((x - x_mean) ** 2 for x in day_offsets)
    slope = (
        sum(
            (x - x_mean) * (m - m_mean)
            for x, m in zip(day_offsets, day_means)
        )
        / denominator
    )
    return {
        "count": count,
        "mean": _round_output(m_mean),
        "slope": _round_output(slope),
        "intercept": _round_output(m_mean - slope * x_mean),
        "uncertainty": _round_output(
            math.sqrt(sum(u ** 2 for u in day_uncertainties)) / count
        ),
    }


def _correlation_window_region(
    values_x: list,
    status_x: list,
    uncertainty_x: list,
    values_y: list,
    status_y: list,
    uncertainty_y: list,
    cells: list[tuple[int, int]],
    t_start: int,
    t_end: int,
    min_count: int,
) -> dict:
    paired_vx: list[float] = []
    paired_vy: list[float] = []
    paired_ux: list[float] = []
    paired_uy: list[float] = []
    for t in range(t_start, t_end + 1):
        for i, j in cells:
            if status_x[t][i][j] != "missing" and status_y[t][i][j] != "missing":
                paired_vx.append(values_x[t][i][j])
                paired_vy.append(values_y[t][i][j])
                paired_ux.append(uncertainty_x[t][i][j])
                paired_uy.append(uncertainty_y[t][i][j])

    count = len(paired_vx)
    if count < min_count:
        return {
            "count": count,
            "covariance": None,
            "correlation": None,
            "uncertainty": None,
        }

    mean_x = sum(paired_vx) / count
    mean_y = sum(paired_vy) / count
    covariance = (
        sum((vx - mean_x) * (vy - mean_y) for vx, vy in zip(paired_vx, paired_vy))
        / count
    )
    variance_x = sum((vx - mean_x) ** 2 for vx in paired_vx) / count
    variance_y = sum((vy - mean_y) ** 2 for vy in paired_vy) / count
    if variance_x == 0.0 or variance_y == 0.0:
        correlation = None
    else:
        correlation = covariance / math.sqrt(variance_x * variance_y)
    uncertainty = math.sqrt(
        sum(ux ** 2 + uy ** 2 for ux, uy in zip(paired_ux, paired_uy))
    ) / count

    return {
        "count": count,
        "covariance": _round_output(covariance),
        "correlation": (
            None if correlation is None else _round_output(correlation)
        ),
        "uncertainty": _round_output(uncertainty),
    }


def _regression_window_region(
    values_x: list,
    status_x: list,
    uncertainty_x: list,
    values_y: list,
    status_y: list,
    uncertainty_y: list,
    cells: list[tuple[int, int]],
    t_start: int,
    t_end: int,
    min_count: int,
) -> dict:
    paired_vx: list[float] = []
    paired_vy: list[float] = []
    paired_ux: list[float] = []
    paired_uy: list[float] = []
    for t in range(t_start, t_end + 1):
        for i, j in cells:
            if status_x[t][i][j] != "missing" and status_y[t][i][j] != "missing":
                paired_vx.append(values_x[t][i][j])
                paired_vy.append(values_y[t][i][j])
                paired_ux.append(uncertainty_x[t][i][j])
                paired_uy.append(uncertainty_y[t][i][j])

    count = len(paired_vx)
    if count < min_count:
        return {
            "count": count,
            "slope": None,
            "intercept": None,
            "uncertainty": None,
        }

    mean_x = sum(paired_vx) / count
    mean_y = sum(paired_vy) / count
    sxx = sum((vx - mean_x) ** 2 for vx in paired_vx)
    sxy = sum(
        (vx - mean_x) * (vy - mean_y)
        for vx, vy in zip(paired_vx, paired_vy)
    )
    if sxx == 0.0:
        slope = None
        intercept = None
    else:
        slope = sxy / sxx
        intercept = mean_y - slope * mean_x
    uncertainty = math.sqrt(
        sum(ux ** 2 + uy ** 2 for ux, uy in zip(paired_ux, paired_uy))
    ) / count

    return {
        "count": count,
        "slope": None if slope is None else _round_output(slope),
        "intercept": (
            None if intercept is None else _round_output(intercept)
        ),
        "uncertainty": _round_output(uncertainty),
    }


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


def _exceedance_window_region(
    values: list,
    status: list,
    uncertainty: list,
    cells: list[tuple[int, int]],
    t_start: int,
    t_end: int,
    threshold: float,
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
            "exceed": None,
            "rate": None,
            "mean_excess": None,
            "uncertainty": None,
        }

    exceed = sum(1 for v in window_values if v > threshold)
    excess = sum(max(v - threshold, 0.0) for v in window_values)
    return {
        "count": count,
        "exceed": exceed,
        "rate": _round_output(exceed / count),
        "mean_excess": _round_output(excess / count),
        "uncertainty": _round_output(
            math.sqrt(sum(u ** 2 for u in window_uncertainties)) / count
        ),
    }


def _quantile_window_region(
    values: list,
    status: list,
    uncertainty: list,
    cells: list[tuple[int, int]],
    t_start: int,
    t_end: int,
    quantiles: list[float],
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
            "quantiles": [None for _q in quantiles],
            "uncertainty": None,
        }

    window_values.sort()
    quantile_values = []
    for q in quantiles:
        h = (count - 1) * q
        a = math.floor(h)
        b = math.ceil(h)
        quantile_values.append(
            _round_output(
                window_values[a] + (h - a) * (window_values[b] - window_values[a])
            )
        )

    return {
        "count": count,
        "quantiles": quantile_values,
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


def aggregate_variance(
    temporal, element, regions, windows, *, min_count: int = 1
) -> dict:
    """Aggregate a reconstructed grid series into per-window variance stats.

    Behaves like :func:`aggregate_window` — ``temporal``, ``element``,
    ``regions``, ``windows`` and ``min_count`` follow the same validation,
    exceptions, input order, closed-interval and non-``missing`` sampling
    rules — but instead of mean/min/max it reports spread statistics.

    For every window (in window order) and region (in region order), every
    non-``missing`` cell of every day of the inclusive interval contributes a
    sample ``(v, u)`` of value and uncertainty.  ``count`` is the number of
    samples ``n``; when ``n`` is below ``min_count``, ``mean``, ``variance``,
    ``stddev`` and ``uncertainty`` are all ``None``.  Otherwise, with
    ``m = sum(v) / n``, they are ``m``, ``sum((v - m) ** 2) / n``,
    ``sqrt(sum((v - m) ** 2) / n)`` and ``sqrt(sum(u ** 2)) / n`` over the
    samples' uncertainties.

    The returned mapping uses the key order ``schema, element, windows,
    regions, data``; ``schema`` is ``climate-grid/variance-window-v1``,
    ``element`` and ``windows`` echo the arguments unchanged and ``regions``
    lists the region names in input order.  ``data`` follows the window
    order; each entry uses the key order ``name, regions``, where each region
    entry uses the key order ``name, count, mean, variance, stddev,
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
            stats = _variance_window_region(
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
        "schema": _VARIANCE_SCHEMA,
        "element": element,
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


def aggregate_coverage(temporal, element, regions, windows, *, min_count: int = 1) -> dict:
    """Aggregate a reconstructed grid series into per-window coverage stats.

    Behaves like :func:`aggregate_window` — ``temporal``, ``element``,
    ``regions``, ``windows`` and ``min_count`` follow the same validation,
    exceptions and sampling rules — but it reports data coverage rather than
    value statistics.

    For every window (in window order) and region (in region order),
    ``total`` is the number of days in the inclusive interval times the
    number of cells in the region.  ``available`` counts non-``missing``
    cells, split into ``observed`` and ``interpolated``; all four counts are
    ints.  When ``available`` is below ``min_count``, ``rate``,
    ``observed_rate``, ``interpolated_rate`` and ``uncertainty`` are all
    ``None``.  Otherwise they are ``available / total``, ``observed /
    total``, ``interpolated / total`` and ``sqrt(sum(u ** 2)) / available``
    over the available cells' uncertainties.

    The returned mapping uses the key order ``schema, element, windows,
    regions, data``; ``schema`` is ``climate-grid/coverage-window-v1``,
    ``element`` echoes the argument, ``windows`` lists three-key dicts in
    input order and ``regions`` lists the region names in input order.
    ``data`` follows the window order; each entry uses the key order
    ``name, regions``, where each region entry uses the key order ``name,
    total, available, observed, interpolated, rate, observed_rate,
    interpolated_rate, uncertainty``.  Every output float is
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
    status = series["status"]
    uncertainty = series["uncertainty"]

    result_data = []
    for w_name, _start, _end, t_start, t_end in validated_windows:
        region_stats = []
        for r_name, cells in validated_regions:
            stats = _coverage_window_region(
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
        "schema": _COVERAGE_SCHEMA,
        "element": element,
        "windows": [
            {"name": name, "start": start, "end": end}
            for name, start, end, _t_start, _t_end in validated_windows
        ],
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


def aggregate_exceedance(
    temporal, element, regions, windows, threshold, *, min_count: int = 1
) -> dict:
    """Aggregate a reconstructed grid series into per-window exceedance stats.

    Behaves like :func:`aggregate_window` — ``temporal``, ``element``,
    ``regions``, ``windows`` and ``min_count`` follow the same validation,
    exceptions and sampling rules — but instead of mean/min/max it reports
    threshold-exceedance statistics.  ``threshold`` must be a finite non-bool
    int or float; a wrong type raises ``TypeError`` and a non-finite value
    raises ``ValueError``.

    For every window (in window order) and region (in region order), every
    non-``missing`` cell of every day of the inclusive interval contributes a
    sample ``(v, u)`` of value and uncertainty.  ``count`` is the number of
    samples; ``exceed`` is the count of samples with ``v > threshold``;
    ``rate`` is ``exceed / count``; ``mean_excess`` is
    ``sum(max(v - threshold, 0)) / count`` and ``uncertainty`` is
    ``sqrt(sum(u ** 2)) / count`` over the samples' uncertainties.  When
    ``count`` is below ``min_count``, everything except ``count`` (``exceed``,
    ``rate``, ``mean_excess`` and ``uncertainty``) is ``None``.

    The returned mapping uses the key order ``schema, element, threshold,
    windows, regions, data``; ``schema`` is
    ``climate-grid/exceedance-window-v1``, ``element`` and ``threshold`` echo
    the arguments, ``windows`` is passed through unchanged and ``regions``
    lists the region names in input order.  ``data`` follows the window order;
    each entry uses the key order ``name, regions``, where each region entry
    uses the key order ``name, count, exceed, rate, mean_excess,
    uncertainty``.  ``count`` and ``exceed`` are ints and every output float
    is ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs are
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

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a finite non-bool int or float")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")

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
            stats = _exceedance_window_region(
                values,
                status,
                uncertainty,
                cells,
                t_start,
                t_end,
                threshold,
                min_count,
            )
            region_stats.append({"name": r_name, **stats})
        result_data.append({"name": w_name, "regions": region_stats})

    return {
        "schema": _EXCEEDANCE_SCHEMA,
        "element": element,
        "threshold": threshold,
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


def _validate_quantiles(quantiles: Any) -> list[float]:
    if not isinstance(quantiles, list):
        raise TypeError("quantiles must be a list")
    if len(quantiles) == 0:
        raise ValueError("quantiles must be non-empty")
    validated: list[float] = []
    for index, q in enumerate(quantiles):
        if not isinstance(q, (int, float)) or isinstance(q, bool):
            raise TypeError(
                f"quantiles[{index}] must be a finite non-bool int or float"
            )
        if not math.isfinite(q):
            raise ValueError(f"quantiles[{index}] must be finite")
        if q < 0.0 or q > 1.0:
            raise ValueError(f"quantiles[{index}] must satisfy 0 <= q <= 1")
        if index > 0 and q <= validated[index - 1]:
            raise ValueError("quantiles must be strictly increasing")
        validated.append(q)
    return validated


def aggregate_quantile(
    temporal, element, regions, windows, quantiles, *, min_count: int = 1
) -> dict:
    """Aggregate a reconstructed grid series into per-window region quantiles.

    Behaves like :func:`aggregate_window` — ``temporal``, ``element``,
    ``regions``, ``windows`` and ``min_count`` follow the same validation,
    exceptions and sampling rules — but instead of mean/min/max it reports
    the requested quantiles.  ``quantiles`` is a non-empty list of finite
    non-bool numbers with ``0 <= q <= 1`` in strictly increasing order;
    wrong item types raise ``TypeError`` while an empty, out-of-range or
    non-increasing list raises ``ValueError``.

    For every window (in window order) and region (in region order), every
    non-``missing`` cell of every day of the inclusive interval contributes a
    sample.  ``count`` is the number of samples; when it is below
    ``min_count``, ``quantiles`` is a list of ``None`` of the same length as
    the argument and ``uncertainty`` is ``None``.  Otherwise the samples are
    sorted ascending and each quantile ``q`` is computed as
    ``h = (count - 1) * q``, ``a = floor(h)``, ``b = ceil(h)``,
    ``v[a] + (h - a) * (v[b] - v[a])``; ``uncertainty`` is
    ``sqrt(sum(u ** 2)) / count`` over the samples' uncertainties.

    The returned mapping uses the key order ``schema, element, quantiles,
    windows, regions, data``; ``schema`` is
    ``climate-grid/quantile-window-v1``, ``element`` echoes the argument,
    ``quantiles`` and ``windows`` are echoed in input order and ``regions``
    lists the region names in input order.  ``data`` follows the window
    order; each entry uses the key order ``name, regions``, where each region
    entry uses the key order ``name, count, quantiles, uncertainty``.
    ``count`` is an int and every output float is ``round(x, 12)`` with
    negative zero normalized to ``0.0``.  Inputs are not modified.
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
    validated_quantiles = _validate_quantiles(quantiles)

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
            stats = _quantile_window_region(
                values,
                status,
                uncertainty,
                cells,
                t_start,
                t_end,
                validated_quantiles,
                min_count,
            )
            region_stats.append({"name": r_name, **stats})
        result_data.append({"name": w_name, "regions": region_stats})

    return {
        "schema": _QUANTILE_SCHEMA,
        "element": element,
        "quantiles": quantiles,
        "windows": [
            {"name": name, "start": start, "end": end}
            for name, start, end, _t_start, _t_end in validated_windows
        ],
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


def aggregate_multi_window(
    temporal, elements, regions, windows, *, min_count: int = 1
) -> dict:
    """Aggregate several elements into per-window region statistics at once.

    Combines :func:`aggregate_multi` and :func:`aggregate_window`:
    ``temporal``, ``regions`` and ``min_count`` follow :func:`aggregate`;
    ``elements`` is a non-empty list of unique non-empty str element names
    that must all exist in ``temporal.data`` (wrong item types raise
    ``TypeError``; an empty, duplicate or unknown element raises
    ``ValueError``); ``windows`` follows :func:`aggregate_window` — a
    non-empty list of dicts with exactly the keys ``name``, ``start`` and
    ``end`` in that order, where ``name`` is a unique non-empty str and
    ``start``/``end`` are valid ``YYYY-MM-DD`` dates occurring in
    ``temporal.times`` with ``start <= end``.

    For every window (in window order), region (in region order) and element
    (in element order), every non-``missing`` cell of every day of the
    inclusive interval contributes a sample.  ``count`` is the number of
    samples; when it is below ``min_count``, ``mean``, ``min``, ``max`` and
    ``uncertainty`` are all ``None``.  Otherwise they are the arithmetic
    mean, minimum and maximum of the sample values, and
    ``sqrt(sum(u ** 2)) / count`` over the samples' uncertainties.

    The returned mapping uses the key order ``schema, elements, windows,
    regions, data``; ``schema`` is ``climate-grid/multi-window-v1``,
    ``elements`` and ``windows`` are passed through unchanged and ``regions``
    lists the region names in input order.  ``data`` is a flat list of rows
    in window-then-region-then-element order; each row uses the key order
    ``window, region, element, count, mean, min, max, uncertainty``.
    ``count`` is an int and every output float is ``round(x, 12)`` with
    negative zero normalized to ``0.0``.  Inputs are not modified.
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
        "schema": _MULTI_WINDOW_SCHEMA,
        "elements": elements,
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


def aggregate_trend(
    temporal, element, regions, windows, *, min_count: int = 1
) -> dict:
    """Aggregate a reconstructed grid series into per-window trend statistics.

    Behaves like :func:`aggregate_window` — ``temporal``, ``element``,
    ``regions``, ``windows`` and ``min_count`` follow the same validation,
    exceptions, input order, closed-interval and non-``missing`` sampling
    rules — but instead of pooling all samples it first reduces each day to
    a regional mean and then fits a linear trend over the retained days.

    For every window (in window order) and region (in region order), each day
    of the inclusive interval whose number ``n`` of non-``missing`` cells
    satisfies ``n >= min_count`` is retained, contributing the daily mean
    ``m = sum(v) / n`` and daily uncertainty ``sqrt(sum(u ** 2)) / n`` over
    those cells.  With ``x`` the day offset from the window start, ``k`` the
    number of retained days, ``x_bar`` and ``m_bar`` the means of the offsets
    and daily means, the statistics are ``slope = sum((x - x_bar) *
    (m - m_bar)) / sum((x - x_bar) ** 2)``, ``intercept = m_bar - slope *
    x_bar``, ``mean = m_bar`` and ``uncertainty = sqrt(sum(u ** 2)) / k``
    over the retained days' uncertainties.  ``count`` is ``k``; when ``k``
    is below 2, ``mean``, ``slope``, ``intercept`` and ``uncertainty`` are
    all ``None``.

    The returned mapping uses the key order ``schema, element, windows,
    regions, data``; ``schema`` is ``climate-grid/trend-window-v1``,
    ``element`` echoes the argument, ``windows`` is passed through unchanged
    and ``regions`` lists the region names in input order.  ``data`` follows
    the window order; each entry uses the key order ``name, regions``, where
    each region entry uses the key order ``name, count, mean, slope,
    intercept, uncertainty``.  ``count`` is an int and every output float is
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
            stats = _trend_window_region(
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
        "schema": _TREND_SCHEMA,
        "element": element,
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


def aggregate_correlation(
    temporal, element_x, element_y, regions, windows, *, min_count: int = 1
) -> dict:
    """Aggregate two reconstructed grid elements into per-window correlation stats.

    Behaves like :func:`aggregate_window` — ``temporal``, ``regions``,
    ``windows`` and ``min_count`` follow the same validation, exceptions and
    closed-interval sampling rules — but it pairs two elements instead of
    summarizing one.  ``element_x`` and ``element_y`` must each be a non-empty
    str naming an element of ``temporal.data`` and they must name different
    elements; wrong element types raise ``TypeError`` while an empty, unknown
    or repeated element raises ``ValueError``.

    For every window (in window order) and region (in region order), a cell of
    a day contributes a paired sample ``(vx, vy, ux, uy)`` only when its
    status is non-``missing`` for *both* elements.  ``count`` is the number of
    paired samples ``n``; when ``n`` is below ``min_count``, ``covariance``,
    ``correlation`` and ``uncertainty`` are all ``None``.  Otherwise
    ``x_bar = sum(vx) / n`` and ``y_bar = sum(vy) / n``; ``covariance`` is
    ``sum((vx - x_bar) * (vy - y_bar)) / n``; the variances are
    ``sum((vx - x_bar) ** 2) / n`` and ``sum((vy - y_bar) ** 2) / n``;
    ``correlation`` is ``covariance / sqrt(sx2 * sy2)``, or ``None`` when
    either variance is zero; and ``uncertainty`` is
    ``sqrt(sum(ux ** 2 + uy ** 2)) / n`` over the paired samples.

    The returned mapping uses the key order ``schema, element_x, element_y,
    windows, regions, data``; ``schema`` is
    ``climate-grid/correlation-window-v1``, ``element_x`` and ``element_y``
    echo the arguments, ``windows`` is passed through unchanged and
    ``regions`` lists the region names in input order.  ``data`` follows the
    window order; each entry uses the key order ``name, regions``, where each
    region entry uses the key order ``name, count, covariance, correlation,
    uncertainty``.  ``count`` is an int and every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs are
    not modified.
    """
    times, lats, lons, data = _validate_temporal(temporal)

    if not isinstance(element_x, str):
        raise TypeError("element_x must be a str")
    if element_x == "":
        raise ValueError("element_x must be non-empty")
    if element_x not in data:
        raise ValueError(f"unknown element: {element_x!r}")
    if not isinstance(element_y, str):
        raise TypeError("element_y must be a str")
    if element_y == "":
        raise ValueError("element_y must be non-empty")
    if element_y not in data:
        raise ValueError(f"unknown element: {element_y!r}")
    if element_y == element_x:
        raise ValueError("element_x and element_y must be different")

    validated_regions = _validate_regions(regions, len(lats), len(lons))
    validated_windows = _validate_windows(windows, times)

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    series_x = data[element_x]
    series_y = data[element_y]

    result_data = []
    for w_name, _start, _end, t_start, t_end in validated_windows:
        region_stats = []
        for r_name, cells in validated_regions:
            stats = _correlation_window_region(
                series_x["values"],
                series_x["status"],
                series_x["uncertainty"],
                series_y["values"],
                series_y["status"],
                series_y["uncertainty"],
                cells,
                t_start,
                t_end,
                min_count,
            )
            region_stats.append({"name": r_name, **stats})
        result_data.append({"name": w_name, "regions": region_stats})

    return {
        "schema": _CORRELATION_SCHEMA,
        "element_x": element_x,
        "element_y": element_y,
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }


def aggregate_regression(
    temporal, element_x, element_y, regions, windows, *, min_count: int = 1
) -> dict:
    """Aggregate two reconstructed grid elements into per-window regression stats.

    Behaves like :func:`aggregate_correlation` — ``temporal``, ``regions``,
    ``windows`` and ``min_count`` follow the same validation, exceptions and
    closed-interval paired sampling rules, and ``element_x`` and
    ``element_y`` must each be a non-empty str naming an element of
    ``temporal.data`` and name different elements (wrong element types raise
    ``TypeError`` while an empty, unknown or repeated element raises
    ``ValueError``) — but it fits a least-squares line instead of computing
    correlation.

    For every window (in window order) and region (in region order), a cell of
    a day contributes a paired sample ``(x, y, ux, uy)`` only when its status
    is non-``missing`` for *both* elements.  ``count`` is the number of
    paired samples ``n``; when ``n`` is below ``min_count``, ``slope``,
    ``intercept`` and ``uncertainty`` are all ``None``.  Otherwise
    ``x_bar = sum(x) / n`` and ``y_bar = sum(y) / n``; with
    ``Sxx = sum((x - x_bar) ** 2)`` and
    ``Sxy = sum((x - x_bar) * (y - y_bar))``, ``slope`` and ``intercept``
    are ``None`` when ``Sxx`` is zero and otherwise ``slope = Sxy / Sxx``
    and ``intercept = y_bar - slope * x_bar``; ``uncertainty`` is
    ``sqrt(sum(ux ** 2 + uy ** 2)) / n`` over the paired samples.

    The returned mapping uses the key order ``schema, element_x, element_y,
    windows, regions, data``; ``schema`` is
    ``climate-grid/regression-window-v1``, ``element_x`` and ``element_y``
    echo the arguments, ``windows`` is passed through unchanged and
    ``regions`` lists the region names in input order.  ``data`` follows the
    window order; each entry uses the key order ``name, regions``, where each
    region entry uses the key order ``name, count, slope, intercept,
    uncertainty``.  ``count`` is an int and every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs are
    not modified.
    """
    times, lats, lons, data = _validate_temporal(temporal)

    if not isinstance(element_x, str):
        raise TypeError("element_x must be a str")
    if element_x == "":
        raise ValueError("element_x must be non-empty")
    if element_x not in data:
        raise ValueError(f"unknown element: {element_x!r}")
    if not isinstance(element_y, str):
        raise TypeError("element_y must be a str")
    if element_y == "":
        raise ValueError("element_y must be non-empty")
    if element_y not in data:
        raise ValueError(f"unknown element: {element_y!r}")
    if element_y == element_x:
        raise ValueError("element_x and element_y must be different")

    validated_regions = _validate_regions(regions, len(lats), len(lons))
    validated_windows = _validate_windows(windows, times)

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError("min_count must be a non-bool int")
    if min_count < 1:
        raise ValueError("min_count must be positive")

    series_x = data[element_x]
    series_y = data[element_y]

    result_data = []
    for w_name, _start, _end, t_start, t_end in validated_windows:
        region_stats = []
        for r_name, cells in validated_regions:
            stats = _regression_window_region(
                series_x["values"],
                series_x["status"],
                series_x["uncertainty"],
                series_y["values"],
                series_y["status"],
                series_y["uncertainty"],
                cells,
                t_start,
                t_end,
                min_count,
            )
            region_stats.append({"name": r_name, **stats})
        result_data.append({"name": w_name, "regions": region_stats})

    return {
        "schema": _REGRESSION_SCHEMA,
        "element_x": element_x,
        "element_y": element_y,
        "windows": windows,
        "regions": [name for name, _ in validated_regions],
        "data": result_data,
    }
