"""Per-element dataset construction: observation fusion plus IDW gridding."""

from __future__ import annotations

import math

from .interpolation import idw_grid

_SCHEMA = "climate-grid/dataset-v1"
_OBSERVATION_KEYS = frozenset(
    {"source", "priority", "id", "lat", "lon", "value", "quality", "element"}
)
_QUALITIES = frozenset({"good", "suspect", "bad"})
_QUALITY_RANK = {"good": 0, "suspect": 1, "bad": 2}


def _validate_elements(elements: list[str]) -> list[str]:
    if not isinstance(elements, list):
        raise TypeError("elements must be a list of element names")
    if len(elements) == 0:
        raise ValueError("elements must be non-empty")
    seen: set[str] = set()
    for index, element in enumerate(elements):
        where = f"elements[{index}]"
        if not isinstance(element, str):
            raise TypeError(f"{where} must be a str")
        if element == "":
            raise ValueError(f"{where} must be a non-empty str")
        if element in seen:
            raise ValueError(f"duplicate element: {element!r}")
        seen.add(element)
    return elements


def _fuse_by_element(
    observations: list[dict], known_elements: set[str]
) -> dict[tuple[str, str], dict]:
    """Validate observations and fuse them per ``(element, id)`` pair.

    Mirrors :func:`climate_grid.fusion.fuse_observations`, with an extra
    non-empty ``element`` key and uniqueness on ``(source, id, element)``.
    Coordinate consistency is enforced per ``id`` across all elements.
    Returns a mapping ``(element, id) -> selected record``.
    """
    if not isinstance(observations, list):
        raise TypeError("observations must be a list of observation dicts")

    seen_triples: set[tuple[str, str, str]] = set()
    # id -> {"lat": float, "lon": float}
    coordinates: dict[str, dict[str, float]] = {}
    # (element, id) -> list of validated records (value not None, quality != "bad")
    groups: dict[tuple[str, str], list[dict]] = {}

    for index, observation in enumerate(observations):
        where = f"observations[{index}]"
        if not isinstance(observation, dict):
            raise TypeError(f"{where} must be a dict")
        if set(observation.keys()) != _OBSERVATION_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys "
                "source, priority, id, lat, lon, value, quality, element"
            )

        source = observation["source"]
        station_id = observation["id"]
        element = observation["element"]
        if not isinstance(source, str):
            raise TypeError(f"{where}.source must be a str")
        if not isinstance(station_id, str):
            raise TypeError(f"{where}.id must be a str")
        if not isinstance(element, str):
            raise TypeError(f"{where}.element must be a str")
        if source == "":
            raise ValueError(f"{where}.source must be a non-empty str")
        if station_id == "":
            raise ValueError(f"{where}.id must be a non-empty str")
        if element == "":
            raise ValueError(f"{where}.element must be a non-empty str")
        if element not in known_elements:
            raise ValueError(f"{where}.element is not in elements: {element!r}")

        triple = (source, station_id, element)
        if triple in seen_triples:
            raise ValueError(
                f"duplicate observation (source, id, element): "
                f"({source!r}, {station_id!r}, {element!r})"
            )
        seen_triples.add(triple)

        priority = observation["priority"]
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError(f"{where}.priority must be a non-bool int")

        lat = observation["lat"]
        lon = observation["lon"]
        for name, coord in (("lat", lat), ("lon", lon)):
            if not isinstance(coord, (int, float)) or isinstance(coord, bool):
                raise TypeError(
                    f"{where}.{name} must be a finite non-bool int or float"
                )
            if not math.isfinite(coord):
                raise ValueError(f"{where}.{name} must be finite")
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"{where}.lat must be within [-90, 90]")
        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"{where}.lon must be within [-180, 180]")

        value = observation["value"]
        if value is not None:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"{where}.value must be a number or None"
                )
            if not math.isfinite(value):
                raise ValueError(f"{where}.value must be finite")

        quality = observation["quality"]
        if not isinstance(quality, str):
            raise TypeError(f"{where}.quality must be a str")
        if quality not in _QUALITIES:
            raise ValueError(
                f"{where}.quality must be one of good, suspect, bad"
            )

        previous = coordinates.get(station_id)
        if previous is None:
            coordinates[station_id] = {"lat": lat, "lon": lon}
        elif previous["lat"] != lat or previous["lon"] != lon:
            raise ValueError(
                f"coordinate conflict for id {station_id!r}"
            )

        if value is None or quality == "bad":
            continue
        groups.setdefault((element, station_id), []).append(observation)

    selected: dict[tuple[str, str], dict] = {}
    for key, records in groups.items():
        selected[key] = min(
            records,
            key=lambda record: (
                _QUALITY_RANK[record["quality"]],
                record["priority"],
                record["source"],
            ),
        )
    return selected


def build(
    observations: list[dict],
    elements: list[str],
    lats: list[float],
    lons: list[float],
    *,
    power: float = 2.0,
    radius_km: float | None = None,
    min_points: int = 1,
) -> dict:
    """Fuse observations per element and interpolate each onto the grid.

    Observations use the ``fuse_observations`` contract plus a non-empty
    ``element`` key; uniqueness is on ``(source, id, element)`` and station
    coordinates must agree for the same ``id`` across elements.  Each element
    of ``elements`` (which must be non-empty, distinct, non-empty strings) is
    fused and gridded with :func:`climate_grid.interpolation.idw_grid` in
    order; elements without valid stations yield all-``None`` values and
    all-zero counts.

    Returns ``{"schema": "climate-grid/dataset-v1", "data": {...}}`` where
    ``data`` maps each element, in ``elements`` order, to its full IDW result.
    """
    elements = _validate_elements(elements)
    selected = _fuse_by_element(observations, set(elements))

    stations_by_element: dict[str, list[dict]] = {
        element: [] for element in elements
    }
    for (element, station_id), record in selected.items():
        stations_by_element[element].append(
            {
                "id": station_id,
                "lat": record["lat"],
                "lon": record["lon"],
                "value": record["value"],
            }
        )

    data: dict[str, dict] = {}
    for element in elements:
        stations = sorted(
            stations_by_element[element], key=lambda station: station["id"]
        )
        data[element] = idw_grid(
            stations,
            lats,
            lons,
            power=power,
            radius_km=radius_km,
            min_points=min_points,
        )

    return {"schema": _SCHEMA, "data": data}
