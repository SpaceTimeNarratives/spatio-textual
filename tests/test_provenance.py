from spatio_textual.provenance import build_run_manifest, redact_secret_like_keys, sha256_text


def test_sha256_text_is_stable():
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abcd")
    assert sha256_text(None) is None


def test_redact_secret_like_keys_is_recursive_and_non_mutating():
    config = {
        "provider": "example",
        "api_key": "secret-value",
        "nested": {"access_token": "token-value", "model": "demo"},
    }
    redacted = redact_secret_like_keys(config)
    assert redacted["api_key"] == "<redacted>"
    assert redacted["nested"]["access_token"] == "<redacted>"
    assert redacted["nested"]["model"] == "demo"
    assert config["api_key"] == "secret-value"


def test_build_run_manifest_hashes_input_and_redacts_config():
    manifest = build_run_manifest(
        input_text="Spatial humanities",
        config={"model": "example-model", "api_key": "do-not-export"},
        git_commit="abc123",
        timestamp="2026-09-08T12:00:00+00:00",
    )
    assert manifest["git_commit"] == "abc123"
    assert manifest["input_chars"] == len("Spatial humanities")
    assert len(manifest["input_sha256"]) == 64
    assert manifest["config"]["api_key"] == "<redacted>"
    assert manifest["config"]["model"] == "example-model"
