from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Iterable, Optional

CONTINENT_COORDS = {
    "africa": (1.6508, 17.6791),
    "asia": (34.0479, 100.6197),
    "europe": (54.5260, 15.2551),
    "north america": (54.5260, -105.2551),
    "south america": (-8.7832, -55.4915),
    "oceania": (-22.7359, 140.0188),
    "antarctica": (-82.8628, 135.0000),
}

COUNTRY_ALIASES = {
    "america": "United States",
    "usa": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "the united states": "United States",
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "czechoslovakia": "Czechia",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


@lru_cache(maxsize=1)
def _gc():
    try:
        from geonamescache import GeonamesCache
        return GeonamesCache()
    except Exception:
        return None


class GeoResolver:
    """Offline entity linker/geocoder using geonamescache, with safe fallback.

    The resolver prioritises exact country and continent matches, then city
    candidates ranked by population. Ambiguous places are marked so the app can
    surface them for human review before export.
    """

    def __init__(self, prefer_country: str | None = None, max_candidates: int = 5):
        self.prefer_country = prefer_country
        self.max_candidates = max_candidates
        self.gc = _gc()
        self._cities = None
        self._countries = None

    @property
    def cities(self) -> list[dict[str, Any]]:
        if self._cities is None:
            if not self.gc:
                self._cities = []
            else:
                self._cities = list(self.gc.get_cities().values())
        return self._cities

    @property
    def countries(self) -> dict[str, dict[str, Any]]:
        if self._countries is None:
            if not self.gc:
                self._countries = {}
            else:
                raw = self.gc.get_countries()
                out = {}
                for c in raw.values():
                    out[_norm(c.get("name", ""))] = c
                    for key in ("iso", "iso3"):
                        if c.get(key):
                            out[_norm(c[key])] = c
                self._countries = out
        return self._countries

    def resolve(self, name: str, label: str | None = None, context: str | None = None) -> dict[str, Any] | None:
        label = label or ""
        low = _norm(name)
        if not low or label == "GEONOUN":
            return None

        if low in CONTINENT_COORDS:
            lat, lon = CONTINENT_COORDS[low]
            return self._result(name, lat, lon, "CONTINENT", "offline:continent", 1.0, False, [])

        canonical = COUNTRY_ALIASES.get(low, name)
        country = self.countries.get(_norm(canonical))
        if country:
            lat, lon = country.get("latitude"), country.get("longitude")
            if lat is not None and lon is not None:
                return self._result(
                    country.get("name", name), float(lat), float(lon), "COUNTRY",
                    "geonamescache:country", 0.95, False, []
                )

        candidates = [c for c in self.cities if _norm(c.get("name", "")) == low]
        if not candidates:
            # lightweight common historical aliases/fallbacks for demos/tests
            fallback = {
                "london": (51.5072, -0.1276, "United Kingdom"),
                "amsterdam": (52.3676, 4.9041, "Netherlands"),
                "auschwitz": (50.0345, 19.1783, "Poland"),
                "warsaw": (52.2297, 21.0122, "Poland"),
                "berlin": (52.52, 13.405, "Germany"),
                "paris": (48.8566, 2.3522, "France"),
            }
            if low in fallback:
                lat, lon, country_name = fallback[low]
                return self._result(name, lat, lon, "CITY", "offline:fallback", 0.8, False, [{"country": country_name}])
            return {
                "resolution_status": "unresolved",
                "lat": None,
                "lon": None,
                "geo_source": "none",
                "geo_confidence": 0.0,
                "ambiguous": False,
                "candidates_count": 0,
                "candidates": [],
            }

        if self.prefer_country:
            pc = _norm(self.prefer_country)
            preferred = [c for c in candidates if _norm(c.get("countrycode", "")) == pc or _norm(c.get("country", "")) == pc]
            if preferred:
                candidates = preferred + [c for c in candidates if c not in preferred]

        candidates = sorted(candidates, key=lambda c: int(c.get("population") or 0), reverse=True)
        best = candidates[0]
        cand_meta = [
            {
                "name": c.get("name"),
                "countrycode": c.get("countrycode"),
                "population": c.get("population"),
                "lat": c.get("latitude"),
                "lon": c.get("longitude"),
                "geonameid": c.get("geonameid"),
            }
            for c in candidates[: self.max_candidates]
        ]
        ambiguous = len(candidates) > 1
        confidence = 0.88 if not ambiguous else 0.62
        return self._result(
            best.get("name", name), float(best.get("latitude")), float(best.get("longitude")), "CITY",
            "geonamescache:city", confidence, ambiguous, cand_meta, geonameid=best.get("geonameid"), countrycode=best.get("countrycode")
        )

    def _result(self, name: str, lat: float, lon: float, place_type: str, source: str, confidence: float, ambiguous: bool, candidates: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        return {
            "resolved_name": name,
            "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "place_type_resolved": place_type,
            "resolution_status": "resolved_ambiguous" if ambiguous else "resolved",
            "geo_source": source,
            "geo_confidence": confidence,
            "ambiguous": ambiguous,
            "candidates_count": len(candidates),
            "candidates": candidates,
            **extra,
        }
