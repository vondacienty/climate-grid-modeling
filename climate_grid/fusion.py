"""Fusion of station observations coming from multiple sources."""

from __future__ import annotations

import math
from typing import Any

_OBSERVATION_KEYS = frozenset(
    {"source", "priority", "id", "lat", "lon", "value", "quality"}
)
_QUALITIES = frozenset({"good", "suspect", "bad"})
_QUALITY_RANK = {"good": 0, "suspect": 1, "bad": 2}


def fuse_observations(observations: list[dict]) -> list[dict]:
    """Fuse multi-source observations, one per station id.

    Records with a ``None`` value or ``bad`` quality are dropped.  The
    remaining records are grouped by ``id``; the first record is chosen by
    quality (``good`` before ``suspect``), then ascending ``priority``, then
    lexicographic ``source``.

    The returned list is sorted by ``id`` and each item uses the key order
    ``id, lat, lon, value, quality, source`` with values taken directly from
    the selected record.

    Raises ``TypeError`` for value-type problems and ``ValueError`` for bad
    key sets, empty strings, duplicate ``(source, id)`` pairs, out-of-range or
    non-finite coordinates, illegal qualities, or conflicting coordinates for
    the same id.
    """
    if not isinstance(observations, list):
        raise TypeError("observations must be a list of observation dicts")

    seen_pairs: set[tuple[str, str]] = set()
    # id -> {"lat": float, "lon": float}
    coordinates: dict[str, dict[str, float]] = {}
    # id -> list of validated records (value not None, quality != "bad")
    groups: dict[str, list[dict]] = {}

    for index, observation in enumerate(observations):
        where = f"observations[{index}]"
        if not isinstance(observation, dict):
            raise TypeError(f"{where} must be a dict")
        if set(observation.keys()) != _OBSERVATION_KEYS:
            raise ValueError(
                f"{where} must have exactly the keys "
                "source, priority, id, lat, lon, value, quality"
            )

        source = observation["source"]
        station_id = observation["id"]
        if not isinstance(source, str):
            raise TypeError(f"{where}.source must be a str")
        if not isinstance(station_id, str):
            raise TypeError(f"{where}.id must be a str")
        if source == "":
            raise ValueError(f"{where}.source must be a non-empty str")
        if station_id == "":
            raise ValueError(f"{where}.id must be a non-empty str")

        pair = (source, station_id)
        if pair in seen_pairs:
            raise ValueError(
                f"duplicate observation (source, id): "
                f"({source!r}, {station_id!r})"
            )
        seen_pairs.add(pair)

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
        groups.setdefault(station_id, []).append(observation)

    fused: list[dict] = []
    for station_id in sorted(groups):
        records = groups[station_id]
        selected = min(
            records,
            key=lambda record: (
                _QUALITY_RANK[record["quality"]],
                record["priority"],
                record["source"],
            ),
        )
        fused.append(
            {
                "id": selected["id"],
                "lat": selected["lat"],
                "lon": selected["lon"],
                "value": selected["value"],
                "quality": selected["quality"],
                "source": selected["source"],
            }
        )

    return fused
