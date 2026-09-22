"""Attribution of extreme-event intensity trends to an external driver."""

from __future__ import annotations

import math
from typing import Any

_SCHEMA = "climate-grid/attribution-v1"
_BATCH_SCHEMA = "climate-grid/batch-attribution-v1"
_MULTI_SCHEMA = "climate-grid/multi-attribution-v1"
_TRENDS_SCHEMA = "climate-grid/trends-v1"
_TRENDS_KEYS = frozenset({"schema", "years", "data"})
_METRIC_KEYS = ("count", "days", "cells", "intensity", "uncertainty")
_ENTRY_KEYS = frozenset(_METRIC_KEYS + ("slopes",))
_JOB_KEYS = frozenset({"element", "driver"})


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


def _validate_element(element: Any, data: dict, where: str) -> None:
    if not isinstance(element, str):
        raise TypeError(f"{where} must be a str")
    if element == "":
        raise ValueError(f"{where} must be non-empty")
    if element not in data:
        raise ValueError(f"unknown element: {element!r}")


def _validate_driver(driver: Any, n_years: int, where: str) -> None:
    if not isinstance(driver, list):
        raise TypeError(f"{where} must be a list")
    if len(driver) != n_years:
        raise ValueError(f"{where} must have one entry per year")
    for index, value in enumerate(driver):
        _validate_number(value, f"{where}[{index}]", nullable=True)


def _validate_drivers(drivers: Any, n_years: int) -> None:
    if not isinstance(drivers, dict):
        raise TypeError("drivers must be a dict")
    if len(drivers) == 0:
        raise ValueError("drivers must be non-empty")
    for name, driver in drivers.items():
        if not isinstance(name, str):
            raise TypeError("drivers keys must be str")
        if name == "":
            raise ValueError("drivers keys must be non-empty")
        _validate_driver(driver, n_years, f"drivers[{name!r}]")


def _attribute_entry(years: list, entry: dict, driver: list) -> tuple[list, list]:
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
    return retained_years, result_data


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
    _validate_element(element, data, "element")
    _validate_driver(driver, len(years), "driver")

    retained_years, result_data = _attribute_entry(
        years, data[element], driver
    )
    return {"schema": _SCHEMA, "years": retained_years, "data": result_data}


def batch_attribute(trends, jobs) -> dict:
    """Attribute several element/driver pairs in one call.

    ``trends`` must be a complete :func:`climate_grid.trends.summarize`
    result (schema ``climate-grid/trends-v1``), validated exactly as for
    :func:`attribute`.  ``jobs`` must be a non-empty list of dicts, each
    with exactly the keys ``element`` and ``driver``, satisfying the same
    contract as the :func:`attribute` arguments of those names:
    ``element`` a non-empty str naming one of the trends elements and
    ``driver`` a list with one entry per year in ``trends.years``, each
    either ``None`` or a finite non-bool number.

    Each job is processed with the same logic as :func:`attribute`:
    years where the driver or the element's ``intensity`` is ``None``
    are dropped, and fewer than two retained years or zero variance in
    the retained driver values raise ``ValueError``.

    The returned mapping uses the key order ``schema, results``;
    ``schema`` is ``climate-grid/batch-attribution-v1`` and ``results``
    has one entry per job, in job order.  Each entry uses the key order
    ``element, years, data``: ``element`` echoes the job's element,
    ``years`` lists the retained years and ``data`` is an equally long
    list, one dict per retained year in the key order ``contribution,
    uncertainty`` — the contribution is ``b * (x - xbar)`` and the
    uncertainty is the element's original uncertainty for that year.
    Every output float is ``round(x, 12)`` with negative zero normalized
    to ``0.0``.  Inputs are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    years, data = _validate_trends(trends)

    if not isinstance(jobs, list):
        raise TypeError("jobs must be a list")
    if len(jobs) == 0:
        raise ValueError("jobs must be non-empty")
    for index, job in enumerate(jobs):
        where = f"jobs[{index}]"
        if not isinstance(job, dict):
            raise TypeError(f"{where} must be a dict")
        if set(job.keys()) != _JOB_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys element, driver"
            )
        _validate_element(job["element"], data, f"{where}.element")
        _validate_driver(job["driver"], len(years), f"{where}.driver")

    results: list = []
    for job in jobs:
        retained_years, result_data = _attribute_entry(
            years, data[job["element"]], job["driver"]
        )
        results.append(
            {
                "element": job["element"],
                "years": retained_years,
                "data": result_data,
            }
        )
    return {"schema": _BATCH_SCHEMA, "results": results}


def multi_attribute(trends, element, drivers) -> dict:
    """Attribute an element's intensity trend to several drivers at once.

    ``trends`` must be a complete :func:`climate_grid.trends.summarize`
    result (schema ``climate-grid/trends-v1``), validated exactly as for
    :func:`attribute`.  ``element`` must be a non-empty str naming one of
    its elements.  ``drivers`` must be a non-empty dict mapping non-empty
    str driver names to per-year series; each series is a list with one
    entry per year in ``trends.years``, each either ``None`` or a finite
    non-bool number.

    The element's yearly ``intensity`` is the regressand y and each
    driver a regressor x.  Only years where the intensity and every
    driver are non-``None`` are retained, preserving year order.  Fewer
    than two retained years, or zero variance in any driver's retained
    values (``sum((x - xbar) ** 2) == 0``), raise ``ValueError``.  With
    arithmetic means xbar and ybar of the retained values, each driver's
    slope is ``b = sum((x - xbar) * (y - ybar)) / sum((x - xbar) ** 2)``.

    The returned mapping uses the key order ``schema, element, years,
    drivers, total, uncertainty``; ``schema`` is
    ``climate-grid/multi-attribution-v1``, ``element`` echoes the
    argument and ``years`` lists the retained years.  ``drivers``
    follows the input driver order and each entry uses the key order
    ``slope, contribution``: the slope is ``b`` and the contribution is
    one ``b * (x - xbar)`` value per retained year.  ``total`` and
    ``uncertainty`` are as long as ``years``: the total is the
    per-year sum of all drivers' contributions, and the uncertainty is
    ``None`` where the element's own uncertainty for that year is
    ``None``, otherwise ``sqrt(d * u ** 2)`` with ``d`` the number of
    drivers and ``u`` that year's uncertainty.  Every output float is
    ``round(x, 12)`` with negative zero normalized to ``0.0``.  Inputs
    are not modified.

    Raises ``TypeError`` for wrong container/item/argument types and
    ``ValueError`` for any other contract violation.
    """
    years, data = _validate_trends(trends)
    _validate_element(element, data, "element")
    _validate_drivers(drivers, len(years))

    entry = data[element]
    intensity = entry["intensity"]
    uncertainty = entry["uncertainty"]
    names = list(drivers.keys())
    series = [drivers[name] for name in names]

    indices = [
        index
        for index in range(len(years))
        if intensity[index] is not None
        and all(driver[index] is not None for driver in series)
    ]
    if len(indices) < 2:
        raise ValueError(
            "at least two years with intensity and all drivers are required"
        )

    n = len(indices)
    ys = [intensity[index] for index in indices]
    mean_y = sum(ys) / n

    contributions: list[list] = []
    driver_entries: dict = {}
    for name, driver in zip(names, series):
        xs = [driver[index] for index in indices]
        mean_x = sum(xs) / n
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator == 0:
            raise ValueError(
                f"retained values of driver {name!r} must not all be equal"
            )
        numerator = sum(
            (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
        )
        b = numerator / denominator
        contribution = [_round_output(b * (x - mean_x)) for x in xs]
        contributions.append(contribution)
        driver_entries[name] = {
            "slope": _round_output(b),
            "contribution": contribution,
        }

    n_drivers = len(names)
    total: list = []
    result_uncertainty: list = []
    for position, index in enumerate(indices):
        total.append(
            _round_output(
                sum(contribution[position] for contribution in contributions)
            )
        )
        u = uncertainty[index]
        result_uncertainty.append(
            None
            if u is None
            else _round_output(math.sqrt(n_drivers * u ** 2))
        )

    return {
        "schema": _MULTI_SCHEMA,
        "element": element,
        "years": [years[index] for index in indices],
        "drivers": driver_entries,
        "total": total,
        "uncertainty": result_uncertainty,
    }
