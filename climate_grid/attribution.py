"""Attribution of yearly element intensity to an external driver."""

from __future__ import annotations

import math
from typing import Any

_SCHEMA = "climate-grid/attribution-v1"
_TRENDS_SCHEMA = "climate-grid/trends-v1"
_TRENDS_KEYS = frozenset({"schema", "years", "data"})
_ELEMENT_KEYS = frozenset(
    {"count", "days", "cells", "intensity", "uncertainty", "slopes"}
)


def _round_output(value: float) -> float:
    value = round(value, 12)
    if value == 0.0:
        # Normalize negative zero.
        return 0.0
    return value


def _validate_optional_number(value: Any, where: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(
            f"{where} must be a finite non-bool int or float or None"
        )
    if not math.isfinite(value):
        raise ValueError(f"{where} must be finite")


def _validate_trends(trends: Any) -> None:
    if not isinstance(trends, dict):
        raise TypeError("trends must be a dict")
    if set(trends.keys()) != _TRENDS_KEYS:
        raise ValueError(
            "trends must have exactly the keys schema, years, data"
        )
    schema = trends["schema"]
    if not isinstance(schema, str):
        raise TypeError("trends.schema must be a str")
    if schema != _TRENDS_SCHEMA:
        raise ValueError(f"trends.schema must be {_TRENDS_SCHEMA!r}")
    years = trends["years"]
    if not isinstance(years, list):
        raise TypeError("trends.years must be a list")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"trends.years[{index}] must be a non-bool int")
    if not isinstance(trends["data"], dict):
        raise TypeError("trends.data must be a dict")


def _validate_series(values: Any, where: str, length: int) -> None:
    if not isinstance(values, list):
        raise TypeError(f"{where} must be a list")
    if len(values) != length:
        raise ValueError(f"{where} must have one item per year")
    for index, value in enumerate(values):
        _validate_optional_number(value, f"{where}[{index}]")


def attribute(trends, element, driver) -> dict:
    """Attribute an element's yearly intensity to an external driver.

    ``trends`` must be a complete :func:`climate_grid.trends.summarize`
    result (schema ``climate-grid/trends-v1``).  ``element`` is a non-empty
    str key of ``trends["data"]``.  ``driver`` is a list with one item per
    year of ``trends["years"]``; each item is a finite non-bool int or
    float, or ``None``.

    The element's ``intensity`` series is used as y and ``driver`` as x;
    only years where both are not ``None`` are kept.  Fewer than two kept
    years, or a zero sum of squared x deviations, raises ``ValueError``.
    ``b`` is the ordinary least-squares slope of y on x over the kept
    years, with ``x̄``/``ȳ`` the arithmetic means of the kept values.

    The returned mapping uses the key order ``schema, years, data``;
    ``schema`` is ``climate-grid/attribution-v1`` and ``years`` holds the
    kept years.  ``data`` is a list of the same length whose items use the
    key order ``contribution, uncertainty`` with values ``b * (x - x̄)``
    and the corresponding original ``uncertainty`` value.  Output floats
    are ``round(x, 12)`` with negative zero normalized to ``0.0``.
    Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    _validate_trends(trends)
    years = trends["years"]
    data = trends["data"]

    if not isinstance(element, str):
        raise TypeError("element must be a str")
    if element == "":
        raise ValueError("element must be non-empty")
    if element not in data:
        raise ValueError(f"element {element!r} is not present in trends.data")

    entry = data[element]
    where = f"trends.data[{element!r}]"
    if not isinstance(entry, dict):
        raise TypeError(f"{where} must be a dict")
    if set(entry.keys()) != _ELEMENT_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys "
            "count, days, cells, intensity, uncertainty, slopes"
        )
    intensity = entry["intensity"]
    uncertainty = entry["uncertainty"]
    _validate_series(intensity, f"{where}.intensity", len(years))
    _validate_series(uncertainty, f"{where}.uncertainty", len(years))

    if not isinstance(driver, list):
        raise TypeError("driver must be a list")
    if len(driver) != len(years):
        raise ValueError("driver must have one item per year")
    for index, value in enumerate(driver):
        _validate_optional_number(value, f"driver[{index}]")

    kept = [
        (year, x, y, u)
        for year, x, y, u in zip(years, driver, intensity, uncertainty)
        if x is not None and y is not None
    ]
    if len(kept) < 2:
        raise ValueError(
            "fewer than two years with both driver and intensity present"
        )
    n = len(kept)
    mean_x = sum(x for _, x, _, _ in kept) / n
    mean_y = sum(y for _, _, y, _ in kept) / n
    sxx = sum((x - mean_x) ** 2 for _, x, _, _ in kept)
    if sxx == 0:
        raise ValueError("kept driver values must not all be equal")
    slope = sum((x - mean_x) * (y - mean_y) for _, x, y, _ in kept) / sxx

    result_data = [
        {
            "contribution": _round_output(slope * (x - mean_x)),
            "uncertainty": u,
        }
        for _, x, _, u in kept
    ]
    return {
        "schema": _SCHEMA,
        "years": [year for year, _, _, _ in kept],
        "data": result_data,
    }
