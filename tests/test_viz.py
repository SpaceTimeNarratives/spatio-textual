from spatio_textual.viz import journeys_to_geojson


class DeterministicResolver:
    """Tiny test resolver so unit tests do not depend on gazetteer population/ambiguity data."""

    COORDS = {
        "London": (51.5072, -0.1276),
        "Paris": (48.8566, 2.3522),
    }

    def resolve(self, name, label=None, context=None):
        if name == "Czechoslovakia":
            return {
                "resolved_name": name,
                "lat": None,
                "lon": None,
                "resolution_status": "unresolved",
                "place_type_resolved": "HISTORICAL_POLITY",
                "geo_source": "test:historical",
                "geo_confidence": 0.0,
                "ambiguous": True,
            }
        coord = self.COORDS.get(name)
        if coord is None:
            return {
                "resolved_name": name,
                "lat": None,
                "lon": None,
                "resolution_status": "unresolved",
                "geo_source": "test:none",
                "geo_confidence": 0.0,
                "ambiguous": False,
            }
        lat, lon = coord
        return {
            "resolved_name": name,
            "lat": lat,
            "lon": lon,
            "resolution_status": "resolved",
            "geo_source": "test:fixed",
            "geo_confidence": 1.0,
            "ambiguous": False,
        }


def test_journeys_to_geojson_maps_resolved_route():
    journeys = [{
        "journeyId": "j1",
        "fileId": "f1",
        "segId": 0,
        "start_location": "London",
        "end_location": "Paris",
        "transport_mode": "train",
        "date": None,
        "journey_reason": None,
        "evidence_quote": "I travelled from London to Paris.",
        "explicit_or_inferred": {
            "start_location": "explicit",
            "end_location": "explicit",
            "transport_mode": "missing",
            "date": "missing",
            "journey_reason": "missing",
        },
        "confidence": 0.9,
        "requires_review": False,
    }]
    geo = journeys_to_geojson(journeys, resolver=DeterministicResolver())
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) == 1
    feature = geo["features"][0]
    assert feature["geometry"]["type"] == "LineString"
    assert len(feature["geometry"]["coordinates"]) == 2
    assert feature["properties"]["journeyId"] == "j1"
    assert geo["audit"][0]["mapped"] is True


def test_journeys_to_geojson_preserves_unresolved_historical_route_in_audit():
    journeys = [{
        "journeyId": "j-historical",
        "start_location": "Czechoslovakia",
        "end_location": "London",
        "evidence_quote": "I travelled from Czechoslovakia to London.",
        "requires_review": True,
    }]
    geo = journeys_to_geojson(journeys, resolver=DeterministicResolver())
    assert geo["features"] == []
    assert geo["audit"][0]["journeyId"] == "j-historical"
    assert geo["audit"][0]["reason"] == "unresolved_endpoint"


def test_journeys_to_geojson_skips_missing_endpoint():
    geo = journeys_to_geojson(
        [{"journeyId": "j2", "start_location": "London", "end_location": None}],
        resolver=DeterministicResolver(),
    )
    assert geo["features"] == []
    assert geo["audit"][0]["reason"] == "missing_endpoint"
