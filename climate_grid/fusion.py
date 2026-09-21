"""Fusion of observations contributed by multiple sources."""

from __future__ import annotations

import math
from typing import Any

_OBSERVATION_KEYS = frozenset(
    {"source", "priority", "id", "lat", "lon", "value", "quality"}
)
_QUALITY_RANK = {"good": 0, "suspect": 1, "bad": 2}


def fuse_observations(observations: list[dict]) -> list[dict]:
    """Pick one observation per station id from a list of sourced records.

    Each observation must be a dict with exactly the keys ``source``,
    ``priority``, ``id``, ``lat``, ``lon``, ``value`` and ``quality``.
    Records whose ``value`` is ``None`` or whose ``quality`` is ``bad`` are
    discarded. For every remaining id the record is chosen by, in order,
    better quality (good before suspect), lower priority and
    lexicographically smaller source. The returned list is sorted by id and
    each result dict has the key order id, lat, lon, value, quality, source
    with values taken directly from the selected record.
    """
    if not isinstance(observations, list):
        raise TypeError("observations must be a list of observation dicts")

    records: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    coords_by_id: dict[str, tuple[Any, Any]] = {}

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
        obs_id = observation["id"]
        priority = observation["priority"]
        lat = observation["lat"]
        lon = observation["lon"]
        value = observation["value"]
        quality = observation["quality"]

        if not isinstance(source, str):
            raise TypeError(f"{where}.source must be a str")
        if not isinstance(obs_id, str):
            raise TypeError(f"{where}.id must be a str")
        if source == "":
            raise ValueError(f"{where}.source must be a non-empty str")
        if obs_id == "":
            raise ValueError(f"{where}.id must be a non-empty str")

        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError(f"{where}.priority must be a non-bool int")

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

        if value is not None:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{where}.value must be a number or None")
            if not math.isfinite(value):
                raise ValueError(f"{where}.value must be finite")

        if not isinstance(quality, str):
            raise TypeError(f"{where}.quality must be a str")
        if quality not in _QUALITY_RANK:
            raise ValueError(
                f"{where}.quality must be one of good, suspect, bad"
            )

        pair = (source, obs_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate observation (source, id): {pair!r}")
        seen_pairs.add(pair)

        coords = (lat, lon)
        if obs_id in coords_by_id:
            if coords_by_id[obs_id] != coords:
                raise ValueError(
                    f"coordinate conflict for observation id: {obs_id!r}"
                )
        else:
            coords_by_id[obs_id] = coords

        records.append(observation)

    candidates = [
        record
        for record in records
        if record["value"] is not None and record["quality"] != "bad"
    ]
    candidates.sort(
        key=lambda record: (
            record["id"],
            _QUALITY_RANK[record["quality"]],
            record["priority"],
            record["source"],
        )
    )

    fused: list[dict] = []
    chosen_ids: set[str] = set()
    for record in candidates:
        obs_id = record["id"]
        if obs_id in chosen_ids:
            continue
        chosen_ids.add(obs_id)
        fused.append(
            {
                "id": record["id"],
                "lat": record["lat"],
                "lon": record["lon"],
                "value": record["value"],
                "quality": record["quality"],
                "source": record["source"],
            }
        )
    return fused
