"""Multi-element dataset building: per-element fusion followed by IDW gridding."""

from __future__ import annotations

import math

from .fusion import fuse_observations
from .interpolation import idw_grid

_SCHEMA = "climate-grid/dataset-v1"
_OBSERVATION_KEYS = frozenset(
    {"source", "priority", "id", "lat", "lon", "value", "quality", "element"}
)
_QUALITIES = frozenset({"good", "suspect", "bad"})


def _validate_elements(elements: object) -> list[str]:
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

    return elements  # type: ignore[return-value]


def build(
    observations: list[dict],
    elements: list[str],
    lats,
    lons,
    *,
    power: float = 2.0,
    radius_km: float | None = None,
    min_points: int = 1,
) -> dict:
    """Fuse observations per element and interpolate each onto a lat/lon grid.

    Observations follow the :func:`fuse_observations` contract with one extra
    non-empty ``element`` str key: the accepted keys are
    ``source, priority, id, lat, lon, value, quality, element``.  Field
    constraints, dropping of ``None`` values and ``bad`` records, and the
    quality/priority/source selection rule are identical to
    :func:`fuse_observations`.  Uniqueness is on the triple
    ``(source, id, element)``, and records sharing an ``id`` (including across
    elements) must carry identical coordinates.

    ``elements`` must be a non-empty list of distinct non-empty strings; each
    observation's ``element`` must be one of them.  Elements are processed in
    list order: the observations of each element are fused exactly as in
    :func:`fuse_observations` and the resulting stations are passed to
    :func:`idw_grid` together with ``lats``, ``lons``, ``power``,
    ``radius_km`` and ``min_points``.  Axis/parameter validation, computation
    and rounding are therefore unchanged; an element without valid stations
    still yields all-``None`` ``values`` and all-zero ``counts``.

    The returned mapping uses the key order ``schema, data``; ``schema`` is
    ``climate-grid/dataset-v1`` and ``data`` is keyed in ``elements`` order,
    each value being the complete :func:`idw_grid` result dictionary for that
    element.  Inputs are not modified.

    Raises ``TypeError`` for containers or items of the wrong list/dict/str
    type and ``ValueError`` for empty or duplicate elements, unknown
    elements, coordinate conflicts, or any of the field/axis/parameter
    violations inherited from :func:`fuse_observations` and :func:`idw_grid`.
    """
    elements = _validate_elements(elements)
    element_set = frozenset(elements)

    if not isinstance(observations, list):
        raise TypeError("observations must be a list of observation dicts")

    seen_triples: set[tuple[str, str, str]] = set()
    # id -> {"lat": float, "lon": float}; shared across all elements.
    coordinates: dict[str, dict[str, float]] = {}
    # element -> validated 7-key observation dicts with usable values.
    groups: dict[str, list[dict]] = {element: [] for element in elements}

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
        if element not in element_set:
            raise ValueError(f"unknown element: {element!r}")

        triple = (source, station_id, element)
        if triple in seen_triples:
            raise ValueError(
                "duplicate observation (source, id, element): "
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
                raise TypeError(f"{where}.value must be a number or None")
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
            raise ValueError(f"coordinate conflict for id {station_id!r}")

        if value is None or quality == "bad":
            continue

        # Pass copies with the seven fuse_observations keys so that caller
        # dicts are never mutated.
        groups[element].append(
            {
                "source": source,
                "priority": priority,
                "id": station_id,
                "lat": lat,
                "lon": lon,
                "value": value,
                "quality": quality,
            }
        )

    data: dict[str, dict] = {}
    for element in elements:
        fused = fuse_observations(groups[element])
        stations = [
            {
                "id": record["id"],
                "lat": record["lat"],
                "lon": record["lon"],
                "value": record["value"],
            }
            for record in fused
        ]
        data[element] = idw_grid(
            stations,
            lats,
            lons,
            power=power,
            radius_km=radius_km,
            min_points=min_points,
        )

    return {"schema": _SCHEMA, "data": data}
