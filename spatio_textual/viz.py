from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

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


def make_map_geojson(geojson: dict, out_html: str | Path = "map.html") -> str:
    import folium

    coords = [f["geometry"]["coordinates"] for f in geojson.get("features", [])]
    if coords:
        center = [sum(c[1] for c in coords) / len(coords), sum(c[0] for c in coords) / len(coords)]
    else:
        center = [52.0, 0.0]
    fmap = folium.Map(location=center, zoom_start=4)
    folium.GeoJson(geojson, name="spatio-textual entities").add_to(fmap)
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
