from __future__ import annotations

"""
viz.py — Minimal helpers to produce map/graph-friendly outputs.

- to_geojson(records, geocoder=None)
- make_map_geojson(geojson, out_html="map.html") using Folium (optional)
- build_cooccurrence(records, nodes=["GPE","PERSON"], window=1) -> edge list
"""
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --- GeoJSON ---

def to_geojson(records: Sequence[Dict], geocoder: Optional[Callable[[str], Tuple[float, float]]] = None,
               place_labels: Sequence[str] = ("CITY", "COUNTRY", "US-STATE", "PLACE", "GPE", "LOC")) -> Dict:
    feats = []
    for r in records:
        ents = r.get("entities") or []
        for e in ents:
            tag = e.get("tag") or e.get("label")
            name = e.get("token")
            if tag not in place_labels:
                continue
            coords = None
            if geocoder is not None:
                try:
                    lat, lon = geocoder(name)
                    coords = (lon, lat)
                except Exception:
                    coords = None
            if coords is None:
                # skip if no geocoding available; could also put null geometry
                continue
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": list(coords)},
                "properties": {
                    "name": name,
                    "tag": tag,
                    "fileId": r.get("fileId"),
                    "segId": r.get("segId"),
                },
            })
    return {"type": "FeatureCollection", "features": feats}


# --- Folium map ---

def make_map_geojson(geojson: Dict, out_html: str = "map.html") -> Optional[str]:
    try:
        import folium  # optional
    except Exception:
        return None
    feats = geojson.get("features", [])
    # center heuristically
    lat = [f["geometry"]["coordinates"][1] for f in feats]
    lon = [f["geometry"]["coordinates"][0] for f in feats]
    center = (sum(lat) / len(lat), sum(lon) / len(lon)) if feats else (0, 0)
    m = folium.Map(location=center, zoom_start=3)
    for f in feats:
        y = f["geometry"]["coordinates"][1]
        x = f["geometry"]["coordinates"][0]
        props = f.get("properties", {})
        folium.Marker([y, x], tooltip=props.get("name"), popup=str(props)).add_to(m)
    m.save(out_html)
    return out_html


# --- Co-occurrence graph ---

def build_cooccurrence(records: Sequence[Dict], nodes: Sequence[str] = ("GPE", "PERSON"), window: int = 1) -> List[Tuple[str, str, int]]:
    """
    Build a simple co-occurrence edge list (u, v, weight) within a sliding window of segments.
    Nodes are entity tokens whose tag/label intersect with `nodes`.
    """
    edges = {}
    def add(u: str, v: str):
        if u == v: return
        a, b = sorted([u, v])
        edges[(a, b)] = edges.get((a, b), 0) + 1

    # collect per-segment nodes
    per_seg: List[List[str]] = []
    for r in records:
        ents = r.get("entities") or []
        names = [e.get("token") for e in ents if (e.get("tag") in nodes or e.get("label") in nodes)]
        per_seg.append(sorted(set(names)))

    # slide over segments
    for i in range(len(per_seg)):
        bag = []
        for j in range(max(0, i - window), min(len(per_seg), i + window + 1)):
            bag.extend(per_seg[j])
        uniq = sorted(set(bag))
        for idx, u in enumerate(uniq):
            for v in uniq[idx + 1:]:
                add(u, v)

    return [(u, v, w) for (u, v), w in sorted(edges.items(), key=lambda kv: (-kv[1], kv[0]))]
