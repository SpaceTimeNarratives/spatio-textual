from spatio_textual.geocode import GeoResolver


def test_historical_polity_is_preserved_not_mapped_to_modern_country():
    result = GeoResolver().resolve("Czechoslovakia", label="GPE")
    assert result is not None
    assert result["resolved_name"] == "Czechoslovakia"
    assert result["place_type_resolved"] == "HISTORICAL_POLITY"
    assert result["resolution_status"] == "unresolved"
    assert result["geo_source"] == "historical_name:preserved"
    assert result["lat"] is None
    assert result["lon"] is None
    assert result["historical_name"] is True
    assert result["ambiguous"] is True


def test_known_fallback_place_remains_resolvable():
    # This assertion is intentionally modest: the resolver may use geonamescache
    # or the built-in fallback, but should still return usable coordinates.
    result = GeoResolver().resolve("London", label="GPE")
    assert result is not None
    assert result["resolution_status"] in {"resolved", "resolved_ambiguous"}
    assert result["lat"] is not None
    assert result["lon"] is not None
