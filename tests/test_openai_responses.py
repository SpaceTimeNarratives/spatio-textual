import sys
import types

from spatio_textual.openai_responses import OpenAIResponsesJSONClient


class _Usage:
    input_tokens = 17
    output_tokens = 9


class _Response:
    id = "resp_test_123"
    model = "gpt-test-resolved"
    output_text = '{"spans": []}'
    usage = _Usage()


class _Responses:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _Response()


class _FakeOpenAI:
    last_instance = None

    def __init__(self, **kwargs):
        self.responses = _Responses()
        self.init_kwargs = kwargs
        _FakeOpenAI.last_instance = self


def test_responses_client_uses_strict_schema_and_preserves_metadata(monkeypatch):
    fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"spans": {"type": "array", "items": {"type": "object"}}},
        "required": ["spans"],
    }
    client = OpenAIResponsesJSONClient(
        model="gpt-test-requested",
        response_schema=schema,
        schema_name="spatio_textual_test",
        reasoning_effort="none",
        api_key="test-key",
    )
    result = client.complete_json("llm_toponym_extraction", "prompt", input_text="London")

    instance = _FakeOpenAI.last_instance
    assert instance is not None
    sent = instance.responses.kwargs
    assert sent["model"] == "gpt-test-requested"
    assert sent["input"] == "prompt"
    assert sent["reasoning"] == {"effort": "none"}
    assert sent["store"] is False
    assert sent["text"]["format"]["type"] == "json_schema"
    assert sent["text"]["format"]["name"] == "spatio_textual_test"
    assert sent["text"]["format"]["schema"] == schema
    assert sent["text"]["format"]["strict"] is True

    assert result["spans"] == []
    assert result["_raw_response_text"] == '{"spans": []}'
    assert result["_response_metadata"]["response_id"] == "resp_test_123"
    assert result["_response_metadata"]["requested_model"] == "gpt-test-requested"
    assert result["_response_metadata"]["resolved_model"] == "gpt-test-resolved"
    assert result["telemetry"]["input_tokens"] == 17
    assert result["telemetry"]["output_tokens"] == 9
    assert result["telemetry"]["success"] is True


def test_responses_client_records_provider_failure_without_fabricating_output(monkeypatch):
    class BrokenResponses:
        def create(self, **kwargs):
            raise RuntimeError("simulated provider failure")

    class BrokenOpenAI:
        def __init__(self, **kwargs):
            self.responses = BrokenResponses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=BrokenOpenAI))
    client = OpenAIResponsesJSONClient(
        model="gpt-test",
        response_schema={"type": "object"},
        schema_name="test",
    )
    result = client.complete_json("task", "prompt")

    assert result["telemetry"]["success"] is False
    assert "simulated provider failure" in result["telemetry"]["error"]
    assert result["_raw_response_text"] == ""
    assert result["_response_metadata"]["response_id"] is None
