"""Spatial interpolation utilities for gridded climate modelling."""

from __future__ import annotations

import math
from typing import Any

EARTH_RADIUS_KM = 6371.0088
_SCHEMA = "climate-grid/idw-v1"
_STATION_KEYS = frozenset({"id", "lat", "lon", "value"})


def _check_number_type(value: Any, name: str) -> None:
    """Raise TypeError unless *value* is a non-bool int or float."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite non-bool int or float")


def _check_finite(value: Any, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_stations(stations: Any) -> list[dict]:
    if not isinstance(stations, list):
        raise TypeError("stations must be a list of station dicts")

    seen_ids: set[str] = set()
    for index, station in enumerate(stations):
        where = f"stations[{index}]"
        if not isinstance(station, dict):
            raise TypeError(f"{where} must be a dict")
        if set(station.keys()) != _STATION_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys id, lat, lon, value"
            )

        station_id = station["id"]
        if not isinstance(station_id, str):
            raise TypeError(f"{where}.id must be a str")
        if station_id == "":
            raise ValueError(f"{where}.id must be a non-empty str")
        if station_id in seen_ids:
            raise ValueError(f"duplicate station id: {station_id!r}")
        seen_ids.add(station_id)

        for key in ("lat", "lon", "value"):
            _check_number_type(station[key], f"{where}.{key}")
            _check_finite(station[key], f"{where}.{key}")

        lat = station["lat"]
        lon = station["lon"]
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"{where}.lat must be within [-90, 90]")
        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"{where}.lon must be within [-180, 180]")

    return stations


def _validate_axis(axis: Any, name: str) -> None:
    if not isinstance(axis, list):
        raise TypeError(f"{name} must be a list")
    if len(axis) == 0:
        raise ValueError(f"{name} must be non-empty")
    for index, value in enumerate(axis):
        _check_number_type(value, f"{name}[{index}]")
        _check_finite(value, f"{name}[{index}]")
    for index in range(1, len(axis)):
        if axis[index] <= axis[index - 1]:
            raise ValueError(f"{name} must be strictly increasing")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km using the standard Haversine formula."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    # Clamp to [0, 1] to absorb floating-point overshoot for far/antipodal points.
    a = min(1.0, max(0.0, a))
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _round_output(value: float) -> float:
    value = round(value, 12)
    if value == 0.0:
        # Normalize negative zero.
        return 0.0
    return value


def idw_grid(
    stations,
    lats,
    lons,
    *,
    power: float = 2.0,
    radius_km: float | None = None,
    min_points: int = 1,
) -> dict:
    """Inverse-distance-weighted interpolation onto a regular lat/lon grid.

    See the project data contract (schema ``climate-grid/idw-v1``) for the
    exact semantics of inputs and the returned mapping.
    """
    stations = _validate_stations(stations)
    _validate_axis(lats, "lats")
    _validate_axis(lons, "lons")

    _check_number_type(power, "power")
    _check_finite(power, "power")
    if power <= 0:
        raise ValueError("power must be a positive number")

    if radius_km is not None:
        _check_number_type(radius_km, "radius_km")
        _check_finite(radius_km, "radius_km")
        if radius_km <= 0:
            raise ValueError("radius_km must be None or a positive number")

    if not isinstance(min_points, int) or isinstance(min_points, bool):
        raise TypeError("min_points must be a non-bool int")
    if min_points < 1:
        raise ValueError("min_points must be a positive integer")

    values: list[list[float | None]] = []
    counts: list[list[int]] = []

    for grid_lat in lats:
        row_values: list[float | None] = []
        row_counts: list[int] = []

        for grid_lon in lons:
            neighbors = []
            for station in stations:
                distance = _haversine_km(
                    grid_lat, grid_lon, station["lat"], station["lon"]
                )
                neighbors.append((distance, station["id"], station["value"]))
            neighbors.sort()  # (distance, id, value) ascending

            zero_hits = [item for item in neighbors if item[0] == 0.0]
            if zero_hits:
                candidates = zero_hits
                cell_count = len(candidates)
                if cell_count < min_points:
                    cell_value: float | None = None
                else:
                    cell_value = _round_output(
                        sum(item[2] for item in candidates) / cell_count
                    )
            else:
                if radius_km is None:
                    candidates = neighbors
                else:
                    candidates = [
                        item for item in neighbors if item[0] <= radius_km
                    ]
                cell_count = len(candidates)
                if cell_count < min_points:
                    cell_value = None
                else:
                    weight_sum = 0.0
                    weighted_sum = 0.0
                    for distance, _, station_value in candidates:
                        weight = distance ** (-power)
                        weight_sum += weight
                        weighted_sum += weight * station_value
                    cell_value = _round_output(weighted_sum / weight_sum)

            row_values.append(cell_value)
            row_counts.append(cell_count)

        values.append(row_values)
        counts.append(row_counts)

    return {
        "schema": _SCHEMA,
        "lats": [_round_output(value) for value in lats],
        "lons": [_round_output(value) for value in lons],
        "values": values,
        "counts": counts,
    }
