"""Attribution of extreme-event intensity trends to an external driver."""

from __future__ import annotations

import math
from typing import Any

_SCHEMA = "climate-grid/attribution-v1"
_TRENDS_SCHEMA = "climate-grid/trends-v1"
_TRENDS_KEYS = frozenset({"schema", "years", "data"})
_METRIC_KEYS = ("count", "days", "cells", "intensity", "uncertainty")
_ENTRY_KEYS = frozenset(_METRIC_KEYS + ("slopes",))


def _round_output(value: float) -> float:
    value = round(value, 12)
    if value == 0.0:
        # Normalize negative zero.
        return 0.0
    return value


def _validate_number(value: Any, target: str, *, nullable: bool) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(
            f"{target} must be a finite non-bool int or float"
            + (" or None" if nullable else "")
        )
    if not math.isfinite(value):
        raise ValueError(f"{target} must be finite")


def _validate_trends(trends: Any) -> tuple[list, dict]:
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
    if len(years) == 0:
        raise ValueError("trends.years must be non-empty")
    for index, year in enumerate(years):
        if not isinstance(year, int) or isinstance(year, bool):
            raise TypeError(f"trends.years[{index}] must be a non-bool int")
    for index in range(1, len(years)):
        if years[index] <= years[index - 1]:
            raise ValueError("trends.years must be strictly increasing")

    data = trends["data"]
    if not isinstance(data, dict):
        raise TypeError("trends.data must be a dict")
    if len(data) == 0:
        raise ValueError("trends.data must be non-empty")

    n_years = len(years)
    for element, entry in data.items():
        if not isinstance(element, str):
            raise TypeError("trends.data element names must be str")
        if element == "":
            raise ValueError("trends.data element names must be non-empty")
        where = f"trends.data[{element!r}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{where} must be a dict")
        if set(entry.keys()) != _ENTRY_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys "
                "count, days, cells, intensity, uncertainty, slopes"
            )

        count = entry["count"]
        if not isinstance(count, list):
            raise TypeError(f"{where}.count must be a list")
        if len(count) != n_years:
            raise ValueError(
                f"{where}.count must have one value per year"
            )
        for index, value in enumerate(count):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{where}.count[{index}] must be a non-bool int")

        for metric in _METRIC_KEYS[1:]:
            series = entry[metric]
            if not isinstance(series, list):
                raise TypeError(f"{where}.{metric} must be a list")
            if len(series) != n_years:
                raise ValueError(
                    f"{where}.{metric} must have one value per year"
                )
            for index, value in enumerate(series):
                _validate_number(
                    value, f"{where}.{metric}[{index}]", nullable=True
                )

        slopes = entry["slopes"]
        if not isinstance(slopes, dict):
            raise TypeError(f"{where}.slopes must be a dict")
        if set(slopes.keys()) != set(_METRIC_KEYS):
            raise ValueError(
                f"{where}.slopes must have exactly the keys "
                "count, days, cells, intensity, uncertainty"
            )
        for metric in _METRIC_KEYS:
            _validate_number(
                slopes[metric], f"{where}.slopes.{metric}", nullable=True
            )

    return years, data


def attribute(trends, element, driver) -> dict:
    """Attribute an element's intensity trend to an external driver.

    ``trends`` must be a complete :func:`climate_grid.trends.summarize`
    result (schema ``climate-grid/trends-v1``); ``element`` a non-empty str
    naming one of its elements; and ``driver`` a list with one entry per
    year in ``trends.years``, each either ``None`` or a finite non-bool
    number.

    The element's yearly ``intensity`` is the regressand y and the driver
    the regressor x.  Years where either x or y is ``None`` are dropped,
    preserving year order.  Fewer than two retained years, or zero
    variance in the retained x values (``sum((x - xbar) ** 2) == 0``),
    raise ``ValueError``.  With arithmetic means xbar and ybar of the
    retained values, the slope is
    ``b = sum((x - xbar) * (y - ybar)) / sum((x - xbar) ** 2)``.

    The returned mapping uses the key order ``schema, years, data``;
    ``schema`` is ``climate-grid/attribution-v1`` and ``years`` lists the
    retained years.  ``data`` is an equally long list, one dict per
    retained year in the key order ``contribution, uncertainty``: the
    contribution is ``b * (x - xbar)`` and the uncertainty is the
    element's original uncertainty for that year.  Every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs
    are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    years, data = _validate_trends(trends)

    if not isinstance(element, str):
        raise TypeError("element must be a str")
    if element == "":
        raise ValueError("element must be non-empty")
    if element not in data:
        raise ValueError(f"unknown element: {element!r}")

    if not isinstance(driver, list):
        raise TypeError("driver must be a list")
    if len(driver) != len(years):
        raise ValueError("driver must have one entry per year")
    for index, value in enumerate(driver):
        _validate_number(value, f"driver[{index}]", nullable=True)

    entry = data[element]
    intensity = entry["intensity"]
    uncertainty = entry["uncertainty"]
    points = [
        (index, years[index], driver[index], intensity[index])
        for index in range(len(years))
        if driver[index] is not None and intensity[index] is not None
    ]
    if len(points) < 2:
        raise ValueError(
            "at least two years with both driver and intensity are required"
        )

    n = len(points)
    mean_x = sum(x for _, _, x, _ in points) / n
    mean_y = sum(y for _, _, _, y in points) / n
    denominator = sum((x - mean_x) ** 2 for _, _, x, _ in points)
    if denominator == 0:
        raise ValueError("retained driver values must not all be equal")
    numerator = sum(
        (x - mean_x) * (y - mean_y) for _, _, x, y in points
    )
    b = numerator / denominator

    retained_years: list = []
    result_data: list = []
    for index, year, x, _ in points:
        retained_years.append(year)
        result_data.append(
            {
                "contribution": _round_output(b * (x - mean_x)),
                "uncertainty": _round_output(uncertainty[index]),
            }
        )

    return {"schema": _SCHEMA, "years": retained_years, "data": result_data}
