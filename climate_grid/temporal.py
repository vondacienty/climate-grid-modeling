"""Temporal reconstruction of daily gridded climate datasets."""

from __future__ import annotations

import datetime
import math
import re
from typing import Any

_SCHEMA = "climate-grid/temporal-v1"
_DATASET_SCHEMA = "climate-grid/dataset-v1"
_IDW_SCHEMA = "climate-grid/idw-v1"
_FRAME_KEYS = frozenset({"time", "dataset"})
_DATASET_KEYS = frozenset({"schema", "data"})
_GRID_KEYS = frozenset({"schema", "lats", "lons", "values", "counts"})
_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _round_output(value: float) -> float:
    value = round(value, 12)
    if value == 0.0:
        # Normalize negative zero.
        return 0.0
    return value


def _parse_date(value: Any, where: str) -> datetime.date:
    if not isinstance(value, str):
        raise TypeError(f"{where}.time must be a str")
    if _DATE_RE.fullmatch(value) is None:
        raise ValueError(f"{where}.time must be a YYYY-MM-DD date")
    year, month, day = (int(part) for part in value.split("-"))
    try:
        return datetime.date(year, month, day)
    except ValueError:
        raise ValueError(f"{where}.time must be a valid Gregorian date") from None


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


def _validate_matrix(
    matrix: Any, n_lat: int, n_lon: int, name: str, *, counts: bool
) -> None:
    if not isinstance(matrix, list):
        raise TypeError(f"{name} must be a list")
    if len(matrix) != n_lat:
        raise ValueError(f"{name} must have {n_lat} rows (one per lat)")

    for row_index, row in enumerate(matrix):
        where = f"{name}[{row_index}]"
        if not isinstance(row, list):
            raise TypeError(f"{where} must be a list")
        if len(row) != n_lon:
            raise ValueError(f"{where} must have {n_lon} cells (one per lon)")

        for cell_index, cell in enumerate(row):
            target = f"{where}[{cell_index}]"
            if counts:
                if not isinstance(cell, int) or isinstance(cell, bool):
                    raise TypeError(f"{target} must be a non-bool int")
                if cell < 0:
                    raise ValueError(f"{target} must be non-negative")
            elif cell is not None:
                if not isinstance(cell, (int, float)) or isinstance(cell, bool):
                    raise TypeError(f"{target} must be a number or None")
                if not math.isfinite(cell):
                    raise ValueError(f"{target} must be finite")


def _validate_grid(grid: Any, where: str) -> tuple[list, list, list]:
    if not isinstance(grid, dict):
        raise TypeError(f"{where} must be a dict")
    if set(grid.keys()) != _GRID_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys schema, lats, lons, values, counts"
        )
    if grid["schema"] != _IDW_SCHEMA:
        raise ValueError(f"{where}.schema must be {_IDW_SCHEMA!r}")

    lats = grid["lats"]
    lons = grid["lons"]
    _validate_axis(lats, f"{where}.lats")
    _validate_axis(lons, f"{where}.lons")

    _validate_matrix(
        grid["values"], len(lats), len(lons), f"{where}.values", counts=False
    )
    _validate_matrix(
        grid["counts"], len(lats), len(lons), f"{where}.counts", counts=True
    )

    return lats, lons, grid["values"]


def _validate_dataset(
    dataset: Any, frame_index: int
) -> tuple[list[str], dict[str, tuple[list, list, list]]]:
    where = f"frames[{frame_index}].dataset"
    if not isinstance(dataset, dict):
        raise TypeError(f"{where} must be a dict")
    if set(dataset.keys()) != _DATASET_KEYS:
        raise ValueError(f"{where} must have exactly the keys schema, data")
    if dataset["schema"] != _DATASET_SCHEMA:
        raise ValueError(f"{where}.schema must be {_DATASET_SCHEMA!r}")

    data = dataset["data"]
    if not isinstance(data, dict):
        raise TypeError(f"{where}.data must be a dict")
    if len(data) == 0:
        raise ValueError(f"{where}.data must be non-empty")

    order: list[str] = []
    grids: dict[str, tuple[list, list, list]] = {}
    for element, grid in data.items():
        if not isinstance(element, str):
            raise TypeError(f"{where}.data element names must be str")
        if element == "":
            raise ValueError(f"{where}.data element names must be non-empty")
        order.append(element)
        grids[element] = _validate_grid(grid, f"{where}.data[{element!r}]")

    return order, grids


def reconstruct(frames, *, max_gap: int = 2) -> dict:
    """Reconstruct a gap-filled daily time series of gridded datasets.

    Each frame is a dict with exactly the keys ``time`` and ``dataset``:
    ``time`` is a unique, strictly formatted Gregorian ``YYYY-MM-DD`` str and
    ``dataset`` is a complete :func:`climate_grid.dataset.build` result
    (schema ``climate-grid/dataset-v1``), whose per-element values are
    complete :func:`climate_grid.interpolation.idw_grid` results.  Frames may
    be given out of order, but every frame must contain the same elements in
    the same order, and each element's ``lats``/``lons`` must be identical
    across frames.

    The output time axis spans the earliest through the latest frame date,
    one entry per calendar day in ascending order; days without a frame are
    treated as all-``None``.  For every element and grid cell, known values
    are kept as ``observed``.  A consecutive run of missing values is
    ``interpolated`` by linear interpolation against calendar date only when
    known values bound the run on both sides and the distance in days between
    those bounding values is at most ``max_gap`` (a non-bool non-negative
    int); otherwise the run stays ``missing``, as do un-bounded runs at
    either end of the axis.

    The returned mapping uses the key order ``schema, times, lats, lons,
    data``; ``schema`` is ``climate-grid/temporal-v1`` and ``times`` is the
    list of ``YYYY-MM-DD`` strings.  ``data`` follows the first frame's
    element order and each item uses the key order ``values, status,
    uncertainty``; the three members are nested ``[time][lat][lon]``.
    Status is one of ``observed``, ``interpolated`` or ``missing``, and
    uncertainty is ``0.0`` for observed values, half the absolute difference
    of the two bounding values for interpolated values, and ``None`` for
    missing values.  Every output number (including known values and the
    shared ``lats``/``lons``) is ``round(x, 12)`` with negative zero
    normalized to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for malformed dates, duplicate times, bad schemas or
    grids, empty frames, invalid ``max_gap``, or elements/axes that are not
    consistent across frames.
    """
    if not isinstance(frames, list):
        raise TypeError("frames must be a list of frame dicts")
    if len(frames) == 0:
        raise ValueError("frames must be non-empty")

    # (date, element order, {element: (lats, lons, values)}) per input frame.
    parsed: list[tuple[datetime.date, list[str], dict]] = []
    seen_dates: set[datetime.date] = set()
    for index, frame in enumerate(frames):
        where = f"frames[{index}]"
        if not isinstance(frame, dict):
            raise TypeError(f"{where} must be a dict")
        if set(frame.keys()) != _FRAME_KEYS:
            raise ValueError(f"{where} must have exactly the keys time, dataset")

        day = _parse_date(frame["time"], where)
        if day in seen_dates:
            raise ValueError(f"duplicate frame time: {frame['time']!r}")
        seen_dates.add(day)

        order, grids = _validate_dataset(frame["dataset"], index)
        parsed.append((day, order, grids))

    if not isinstance(max_gap, int) or isinstance(max_gap, bool):
        raise TypeError("max_gap must be a non-bool int")
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")

    reference_order = parsed[0][1]
    reference_grids = parsed[0][2]
    for _, order, grids in parsed[1:]:
        if order != reference_order:
            raise ValueError(
                "all frames must contain the same elements in the same order"
            )
        for element in reference_order:
            ref_lats, ref_lons, _ = reference_grids[element]
            lats, lons, _ = grids[element]
            if lats != ref_lats or lons != ref_lons:
                raise ValueError(
                    f"lats/lons for element {element!r} must match across frames"
                )

    ordered = sorted(parsed, key=lambda item: item[0])
    start = ordered[0][0]
    end = ordered[-1][0]
    n_times = (end - start).days + 1
    times = [
        (start + datetime.timedelta(days=offset)).isoformat()
        for offset in range(n_times)
    ]
    # (day offset from start, grids) for each present frame, ascending.
    present = [((day - start).days, grids) for day, _, grids in ordered]

    first_element = reference_order[0]
    out_lats = [
        _round_output(value) for value in reference_grids[first_element][0]
    ]
    out_lons = [
        _round_output(value) for value in reference_grids[first_element][1]
    ]

    data: dict[str, dict] = {}
    for element in reference_order:
        ref_lats, ref_lons, _ = reference_grids[element]
        n_lat = len(ref_lats)
        n_lon = len(ref_lons)
        values = [
            [[None for _ in range(n_lon)] for _ in range(n_lat)]
            for _ in range(n_times)
        ]
        status = [
            [[None for _ in range(n_lon)] for _ in range(n_lat)]
            for _ in range(n_times)
        ]
        uncertainty = [
            [[None for _ in range(n_lon)] for _ in range(n_lat)]
            for _ in range(n_times)
        ]

        for lat_index in range(n_lat):
            for lon_index in range(n_lon):
                # Known value per day offset; None means missing.
                series: list[float | None] = [None] * n_times
                for offset, grids in present:
                    value = grids[element][2][lat_index][lon_index]
                    if value is not None:
                        series[offset] = value

                tick = 0
                while tick < n_times:
                    if series[tick] is not None:
                        values[tick][lat_index][lon_index] = _round_output(
                            series[tick]
                        )
                        status[tick][lat_index][lon_index] = "observed"
                        uncertainty[tick][lat_index][lon_index] = 0.0
                        tick += 1
                        continue

                    # Run of consecutive missing values [tick, run_end).
                    run_end = tick
                    while run_end < n_times and series[run_end] is None:
                        run_end += 1
                    left = tick - 1
                    gap_days = run_end - left

                    if (
                        left >= 0
                        and run_end < n_times
                        and gap_days <= max_gap
                    ):
                        left_value = series[left]
                        right_value = series[run_end]
                        half_gap = _round_output(
                            abs(right_value - left_value) / 2.0
                        )
                        for missing in range(tick, run_end):
                            fraction = (missing - left) / gap_days
                            interpolated = (
                                left_value
                                + (right_value - left_value) * fraction
                            )
                            values[missing][lat_index][lon_index] = (
                                _round_output(interpolated)
                            )
                            status[missing][lat_index][lon_index] = (
                                "interpolated"
                            )
                            uncertainty[missing][lat_index][lon_index] = half_gap
                    else:
                        for missing in range(tick, run_end):
                            status[missing][lat_index][lon_index] = "missing"

                    tick = run_end

        data[element] = {
            "values": values,
            "status": status,
            "uncertainty": uncertainty,
        }

    return {
        "schema": _SCHEMA,
        "times": times,
        "lats": out_lats,
        "lons": out_lons,
        "data": data,
    }
