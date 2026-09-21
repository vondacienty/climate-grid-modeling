"""Inverse-distance-weighted (IDW) interpolation onto a regular grid."""

from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0088
SCHEMA = "climate-grid/idw-v1"

_STATION_KEYS = {"id", "lat", "lon", "value"}


def _is_number(x: object) -> bool:
    """A 数: a non-bool int or float."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_number(x: object, name: str) -> None:
    if not _is_number(x):
        raise TypeError(f"{name} must be an int or float (non-bool), got {type(x).__name__}")
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x!r}")


def _round12(x: float) -> float:
    v = round(x, 12)
    # normalize -0.0 to 0.0
    return 0.0 if v == 0.0 else v


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    return EARTH_RADIUS_KM * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _validate_stations(stations: object) -> list[tuple[str, float, float, float]]:
    if not isinstance(stations, list):
        raise TypeError(f"stations must be a list, got {type(stations).__name__}")
    seen_ids: set[str] = set()
    parsed: list[tuple[str, float, float, float]] = []
    for index, st in enumerate(stations):
        where = f"stations[{index}]"
        if not isinstance(st, dict):
            raise TypeError(f"{where} must be a dict, got {type(st).__name__}")
        if set(st.keys()) != _STATION_KEYS:
            raise ValueError(f"{where} keys must be exactly {sorted(_STATION_KEYS)}, got {sorted(st.keys())}")
        sid = st["id"]
        if not isinstance(sid, str):
            raise TypeError(f"{where}['id'] must be a str, got {type(sid).__name__}")
        if sid == "":
            raise ValueError(f"{where}['id'] must be non-empty")
        if sid in seen_ids:
            raise ValueError(f"duplicate station id: {sid!r}")
        seen_ids.add(sid)
        lat = st["lat"]
        lon = st["lon"]
        value = st["value"]
        _check_number(lat, f"{where}['lat']")
        _check_number(lon, f"{where}['lon']")
        _check_number(value, f"{where}['value']")
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"{where}['lat'] must be in [-90, 90], got {lat!r}")
        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"{where}['lon'] must be in [-180, 180], got {lon!r}")
        parsed.append((sid, lat, lon, value))
    return parsed


def _validate_axis(axis: object, name: str) -> None:
    if not isinstance(axis, list):
        raise TypeError(f"{name} must be a list, got {type(axis).__name__}")
    if not axis:
        raise ValueError(f"{name} must be non-empty")
    for i, x in enumerate(axis):
        _check_number(x, f"{name}[{i}]")
    for i in range(len(axis) - 1):
        if not axis[i] < axis[i + 1]:
            raise ValueError(f"{name} must be strictly increasing")


def idw_grid(
    stations: list[dict],
    lats: list[float],
    lons: list[float],
    *,
    power: float = 2.0,
    radius_km: float | None = None,
    min_points: int = 1,
) -> dict:
    """Interpolate station values onto a lat/lon grid via IDW.

    Distances use the Haversine formula with Earth radius 6371.0088 km.
    Candidates (stations within ``radius_km`` of a grid point, or all
    stations when ``radius_km`` is None) are ordered by (distance, id).
    Zero-distance candidates yield the arithmetic mean of their values;
    otherwise the value is the ``d ** (-power)`` weighted mean. Grid
    points with fewer than ``min_points`` candidates get None.
    """
    parsed = _validate_stations(stations)
    _validate_axis(lats, "lats")
    _validate_axis(lons, "lons")

    if not _is_number(power):
        raise TypeError(f"power must be an int or float (non-bool), got {type(power).__name__}")
    if not math.isfinite(power) or power <= 0:
        raise ValueError(f"power must be a positive finite number, got {power!r}")

    if radius_km is not None:
        if not _is_number(radius_km):
            raise TypeError(f"radius_km must be None or a number, got {type(radius_km).__name__}")
        if not math.isfinite(radius_km) or radius_km <= 0:
            raise ValueError(f"radius_km must be a positive finite number, got {radius_km!r}")

    if isinstance(min_points, bool) or not isinstance(min_points, int):
        raise TypeError(f"min_points must be a non-bool int, got {type(min_points).__name__}")
    if min_points < 1:
        raise ValueError(f"min_points must be a positive integer, got {min_points!r}")

    values: list[list[float | None]] = []
    counts: list[list[int]] = []
    for glat in lats:
        row_values: list[float | None] = []
        row_counts: list[int] = []
        for glon in lons:
            candidates = []
            for sid, slat, slon, sval in parsed:
                d = _haversine_km(glat, glon, slat, slon)
                if radius_km is None or d <= radius_km:
                    candidates.append((d, sid, sval))
            candidates.sort(key=lambda c: (c[0], c[1]))
            row_counts.append(len(candidates))
            if len(candidates) < min_points:
                row_values.append(None)
            else:
                zero_vals = [v for d, _sid, v in candidates if d == 0.0]
                if zero_vals:
                    row_values.append(_round12(sum(zero_vals) / len(zero_vals)))
                else:
                    weights = [d ** (-power) for d, _sid, _v in candidates]
                    weighted = sum(w * v for w, (_d, _sid, v) in zip(weights, candidates))
                    row_values.append(_round12(weighted / sum(weights)))
        values.append(row_values)
        counts.append(row_counts)

    return {
        "schema": SCHEMA,
        "lats": [_round12(x) for x in lats],
        "lons": [_round12(x) for x in lons],
        "values": values,
        "counts": counts,
    }
