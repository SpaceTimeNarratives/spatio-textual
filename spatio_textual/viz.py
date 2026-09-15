from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .geocode import GeoResolver

PLACE_TYPES = {"GPE", "LOC", "FAC", "COUNTRY", "CITY", "CONTINENT", "CAMP", "PLACE", "REGION"}


def _coord_from_entity(ent: dict[str, Any], geocoder: Callable[[str], Any] | None = None):
    if ent.get("lat") is not None and ent.get("lon") is not None:
        return float(ent["lat"]), float(ent["lon"])
    if ent.get("latitude") is not None and ent.get("longitude") is not None:
        return float(ent["latitude"]), float(ent["longitude"])
    if geocoder:
        result = geocoder(ent.get("text", ""))
        if result is None:
            return None
        if isinstance(result, dict):
            lat = result.get("lat") or result.get("latitude")
            lon = result.get("lon") or result.get("lng") or result.get("longitude")
            if lat is not None and lon is not None:
                return float(lat), float(lon)
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            return float(result[0]), float(result[1])
    return None


def to_geojson(records: Sequence[dict[str, Any]], geocoder: Callable[[str], Any] | None = None) -> dict:
    features: list[dict] = []
    for rec in records:
        for ent in rec.get("entities") or []:
            label = ent.get("place_type") or ent.get("label")
            if label not in PLACE_TYPES and ent.get("label") not in PLACE_TYPES:
                continue
            coord = _coord_from_entity(ent, geocoder)
            if not coord:
                continue
            lat, lon = coord
            props = {k: v for k, v in ent.items() if k not in {"lat", "lon", "latitude", "longitude"}}
            props.update({"fileId": rec.get("fileId"), "segId": rec.get("segId")})
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": props,
            })
    return {"type": "FeatureCollection", "features": features}


def journeys_to_geojson(
    journeys: Sequence[dict[str, Any]],
    *,
    resolver: GeoResolver | None = None,
    allow_ambiguous: bool = False,
) -> dict[str, Any]:
    """Convert grounded journey records to auditable route GeoJSON.

    A route is emitted only when both endpoints can be resolved. Ambiguous
    resolutions are skipped by default rather than silently placing a route on
    the map. Skipped items are preserved in a FeatureCollection foreign member
    named ``audit`` so mapping omissions remain inspectable.
    """
    resolver = resolver or GeoResolver()
    features: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for journey in journeys:
        jid = journey.get("journeyId")
        start_name = journey.get("start_location")
        end_name = journey.get("end_location")
        if not start_name or not end_name:
            audit.append({"journeyId": jid, "mapped": False, "reason": "missing_endpoint"})
            continue

        start = resolver.resolve(str(start_name), label="GPE", context=journey.get("evidence_quote"))
        end = resolver.resolve(str(end_name), label="GPE", context=journey.get("evidence_quote"))
        if not start or not end or start.get("lat") is None or end.get("lat") is None:
            audit.append({
                "journeyId": jid,
                "mapped": False,
                "reason": "unresolved_endpoint",
                "start_resolution": start,
                "end_resolution": end,
            })
            continue

        ambiguous = bool(start.get("ambiguous") or end.get("ambiguous"))
        if ambiguous and not allow_ambiguous:
            audit.append({
                "journeyId": jid,
                "mapped": False,
                "reason": "ambiguous_endpoint",
                "start_resolution": start,
                "end_resolution": end,
            })
            continue

        props = {
            "journeyId": jid,
            "fileId": journey.get("fileId"),
            "segId": journey.get("segId"),
            "start_location": start_name,
            "end_location": end_name,
            "resolved_start": start.get("resolved_name"),
            "resolved_end": end.get("resolved_name"),
            "transport_mode": journey.get("transport_mode"),
            "date": journey.get("date"),
            "journey_reason": journey.get("journey_reason"),
            "evidence_quote": journey.get("evidence_quote"),
            "explicit_or_inferred": journey.get("explicit_or_inferred"),
            "confidence": journey.get("confidence"),
            "requires_review": bool(journey.get("requires_review") or ambiguous),
            "start_geo_source": start.get("geo_source"),
            "end_geo_source": end.get("geo_source"),
            "start_geo_confidence": start.get("geo_confidence"),
            "end_geo_confidence": end.get("geo_confidence"),
            "ambiguous_endpoint": ambiguous,
        }
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(start["lon"]), float(start["lat"])],
                    [float(end["lon"]), float(end["lat"])],
                ],
            },
            "properties": props,
        })
        audit.append({"journeyId": jid, "mapped": True, "reason": None})

    return {"type": "FeatureCollection", "features": features, "audit": audit}


def _geometry_points(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    """Return (lat, lon) pairs from Point/LineString geometries."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if kind == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        return [(float(coords[1]), float(coords[0]))]
    if kind == "LineString" and isinstance(coords, (list, tuple)):
        out = []
        for pair in coords:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                out.append((float(pair[1]), float(pair[0])))
        return out
    return []


def make_map_geojson(geojson: dict, out_html: str | Path = "map.html") -> str:
    import folium

    points: list[tuple[float, float]] = []
    for feature in geojson.get("features", []):
        points.extend(_geometry_points(feature.get("geometry") or {}))
    if points:
        center = [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]
    else:
        center = [52.0, 0.0]
    fmap = folium.Map(location=center, zoom_start=4)
    folium.GeoJson(geojson, name="spatio-textual evidence").add_to(fmap)
    out = str(out_html)
    fmap.save(out)
    return out


def build_cooccurrence(records: Sequence[dict[str, Any]], nodes: tuple[str, ...] = ("PERSON", "GPE", "LOC", "CAMP", "PLACE"), window: int = 1) -> list[tuple[str, str, int]]:
    counts: Counter[tuple[str, str]] = Counter()
    for rec in records:
        ents = [e for e in rec.get("entities") or [] if (e.get("label") in nodes or e.get("place_type") in nodes)]
        names = sorted({e.get("text", "").strip() for e in ents if e.get("text")})
        for i, u in enumerate(names):
            for v in names[i + 1:]:
                if u != v:
                    counts[(u, v)] += 1
    return [(u, v, w) for (u, v), w in counts.most_common()]
