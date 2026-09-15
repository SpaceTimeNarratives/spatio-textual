from spatio_textual.review import apply_human_review, human_correction_burden


def _record():
    return {
        "journeyId": "ex-j-1",
        "end_location": "London",
        "requires_review": True,
        "human_status": "unreviewed",
        "human_edits": [],
    }


def test_apply_accept_preserves_record_and_audit_event():
    original = _record()
    reviewed = apply_human_review(
        original,
        action="accept",
        reason="human_flag",
        timestamp="2026-09-08T12:00:00+00:00",
    )
    assert original["human_status"] == "unreviewed"
    assert reviewed["human_status"] == "accepted"
    assert reviewed["requires_review"] is False
    assert reviewed["human_edits"][0]["action"] == "accept"


def test_apply_edit_records_old_and_new_values():
    reviewed = apply_human_review(
        _record(),
        action="edit",
        field="end_location",
        new_value="London, England",
        reason="disambiguation",
        timestamp="2026-09-08T12:00:00+00:00",
    )
    assert reviewed["end_location"] == "London, England"
    assert reviewed["human_status"] == "edited"
    event = reviewed["human_edits"][0]
    assert event["old_value"] == "London"
    assert event["new_value"] == "London, England"
    assert event["field"] == "end_location"


def test_apply_reject_sets_status():
    reviewed = apply_human_review(
        _record(),
        action="reject",
        reason="unsupported_llm_field",
        timestamp="2026-09-08T12:00:00+00:00",
    )
    assert reviewed["human_status"] == "rejected"
    assert reviewed["requires_review"] is False


def test_correction_burden_distinguishes_review_from_correction():
    accepted = apply_human_review(
        _record(), action="accept", reason="human_flag", timestamp="2026-09-08T12:00:00+00:00"
    )
    edited = apply_human_review(
        _record(), action="edit", field="end_location", new_value="London, England",
        reason="disambiguation", timestamp="2026-09-08T12:00:00+00:00"
    )
    rejected = apply_human_review(
        _record(), action="reject", reason="unsupported_llm_field", timestamp="2026-09-08T12:00:00+00:00"
    )
    unreviewed = _record()

    summary = human_correction_burden([accepted, edited, rejected, unreviewed])
    assert summary["records_total"] == 4
    assert summary["records_reviewed"] == 3
    assert summary["records_corrected"] == 2
    assert summary["review_rate"] == 0.75
    assert summary["correction_rate"] == 0.5
    assert summary["human_edit_events"] == 3
    assert summary["edited_fields"] == ["end_location"]


def test_place_name_correction_clears_geometry_and_preserves_audit():
    from spatio_textual.review import apply_place_review
    from spatio_textual.viz import to_geojson

    entity = {
        'text': 'Cambridge', 'label': 'GPE', 'resolved_name': 'Cambridge',
        'lat': 52.2, 'lon': 0.12, 'latitude': 52.2, 'longitude': 0.12,
        'geo_source': 'machine', 'geo_confidence': 0.8,
        'geonameid': 123, 'countrycode': 'GB', 'place_type_resolved': 'CITY',
        'candidates': [{'name': 'Cambridge', 'lat': 52.2, 'lon': 0.12}],
        'candidates_count': 1, 'ambiguous': True,
        'resolution_status': 'resolved_ambiguous', 'requires_review': True,
        'human_edits': [{'action': 'accept', 'reason': 'human_flag'}],
    }
    result = apply_place_review(entity, action='edit', field='resolved_name',
                               new_value='Cambridge, Massachusetts')
    assert result['resolved_name'] == 'Cambridge, Massachusetts'
    assert result['requires_review'] is True
    assert result['resolution_status'] == 'unresolved'
    assert result['candidates'] == []
    assert result['geo_source'] is None
    assert result['geonameid'] is None
    assert result['countrycode'] is None
    assert to_geojson([{'entities': [result]}])['features'] == []
    assert result['human_edits'][0] == entity['human_edits'][0]
    assert any(e.get('field') == 'lat' and e['old_value'] == 52.2
               for e in result['human_edits'])
    assert entity['lat'] == 52.2  # the original prediction is untouched
    accepted = apply_place_review(entity, action='accept')
    assert accepted['lat'] == 52.2
    assert accepted['resolved_name'] == 'Cambridge'


def test_place_name_correction_normalizes_action_before_clearing_geometry():
    from spatio_textual.review import apply_place_review

    entity = {
        "resolved_name": "Cambridge", "lat": 52.2, "lon": 0.12,
        "resolution_status": "resolved", "requires_review": True,
    }
    result = apply_place_review(
        entity, action=" Edit ", field="resolved_name",
        new_value="Cambridge, Massachusetts",
    )
    assert result["resolved_name"] == "Cambridge, Massachusetts"
    assert result["lat"] is None
    assert result["lon"] is None
    assert result["resolution_status"] == "unresolved"
    assert result["requires_review"] is True
