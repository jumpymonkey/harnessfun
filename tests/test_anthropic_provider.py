"""Unit tests for VertexAnthropicProvider and provider dispatching."""

import json
from unittest.mock import MagicMock, patch
import pytest
import httpx

from harnessfun.models import Message, ProviderResponse, ToolCall, ToolResult
from harnessfun.providers import (
    GCPGeminiProvider,
    VertexAnthropicProvider,
    get_provider,
    is_anthropic_model,
)


def test_is_anthropic_model():
    """Verifies model identification logic."""
    assert is_anthropic_model("claude-3-5-sonnet") is True
    assert is_anthropic_model("claude-3-7-sonnet@20250219") is True
    assert is_anthropic_model("claude-3-haiku") is True
    assert is_anthropic_model("anthropic/claude-3-opus") is True
    assert is_anthropic_model("gemini-2.5-flash") is False
    assert is_anthropic_model("gemini-1.5-pro") is False


def test_get_provider_dispatch(monkeypatch):
    """Verifies get_provider routes to the correct provider class."""
    mock_creds = MagicMock()
    monkeypatch.setattr("harnessfun.providers.vertex_anthropic.verify_gcp_adc", lambda: (mock_creds, "test-proj"))
    monkeypatch.setattr("google.genai.Client", MagicMock())

    p_claude = get_provider("claude-3-5-sonnet", project_id="test-proj", location="us-east5")
    assert isinstance(p_claude, VertexAnthropicProvider)

    p_gemini = get_provider("gemini-2.5-flash", project_id="test-proj", location="global")
    assert isinstance(p_gemini, GCPGeminiProvider)


def test_model_normalization(monkeypatch):
    """Verifies that common aliases map to Vertex AI model identifiers."""
    mock_creds = MagicMock()
    monkeypatch.setattr("harnessfun.providers.vertex_anthropic.verify_gcp_adc", lambda: (mock_creds, "test-proj"))
    provider = VertexAnthropicProvider(project_id="test-proj")

    assert provider._normalize_model("claude-3-5-sonnet") == "claude-3-5-sonnet-v2@20241022"
    assert provider._normalize_model("claude-3-7-sonnet") == "claude-3-7-sonnet@20250219"
    assert provider._normalize_model("claude-3-5-haiku") == "claude-3-5-haiku@20241022"
    assert provider._normalize_model("claude-3-opus") == "claude-3-opus@20240229"
    assert provider._normalize_model("claude-custom@123") == "claude-custom@123"


def test_location_default_us_east5(monkeypatch):
    """Verifies that 'global' or empty location defaults to 'us-east5' for Vertex Anthropic."""
    mock_creds = MagicMock()
    monkeypatch.setattr("harnessfun.providers.vertex_anthropic.verify_gcp_adc", lambda: (mock_creds, "test-proj"))

    p1 = VertexAnthropicProvider(project_id="test-proj", location="global")
    assert p1.location == "us-east5"

    p2 = VertexAnthropicProvider(project_id="test-proj", location="")
    assert p2.location == "us-east5"

    p3 = VertexAnthropicProvider(project_id="test-proj", location="europe-west1")
    assert p3.location == "europe-west1"


def test_tool_schema_conversion(monkeypatch):
    """Verifies Python functions are converted to Anthropic tool schema format."""
    mock_creds = MagicMock()
    monkeypatch.setattr("harnessfun.providers.vertex_anthropic.verify_gcp_adc", lambda: (mock_creds, "test-proj"))
    provider = VertexAnthropicProvider(project_id="test-proj")

    def sample_tool(location: str, unit: str = "celsius") -> str:
        """Get the weather for a location."""
        return "sunny"

    schema = provider._convert_tool_to_anthropic_schema(sample_tool)
    assert schema["name"] == "sample_tool"
    assert schema["description"] == "Get the weather for a location."
    assert "input_schema" in schema
    assert schema["input_schema"]["type"] == "object"
    assert "location" in schema["input_schema"]["properties"]
    assert "unit" in schema["input_schema"]["properties"]
    assert "location" in schema["input_schema"]["required"]


def test_message_conversion(monkeypatch):
    """Verifies Message objects convert into Anthropic messages format."""
    mock_creds = MagicMock()
    monkeypatch.setattr("harnessfun.providers.vertex_anthropic.verify_gcp_adc", lambda: (mock_creds, "test-proj"))
    provider = VertexAnthropicProvider(project_id="test-proj")

    messages = [
        Message(role="user", content="What is the weather in Seattle?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call_123", name="get_weather", args={"location": "Seattle"})],
        ),
        Message(
            role="tool",
            content="",
            tool_results=[ToolResult(call_id="call_123", name="get_weather", output={"temp": 68})],
        ),
    ]

    anthropic_msgs = provider._convert_messages(messages)
    assert len(anthropic_msgs) == 3

    # Check user prompt
    assert anthropic_msgs[0]["role"] == "user"
    assert anthropic_msgs[0]["content"] == "What is the weather in Seattle?"

    # Check assistant tool_use
    assert anthropic_msgs[1]["role"] == "assistant"
    assert anthropic_msgs[1]["content"][0]["type"] == "tool_use"
    assert anthropic_msgs[1]["content"][0]["name"] == "get_weather"
    assert anthropic_msgs[1]["content"][0]["id"] == "call_123"

    # Check tool_result
    assert anthropic_msgs[2]["role"] == "user"
    assert anthropic_msgs[2]["content"][0]["type"] == "tool_result"
    assert anthropic_msgs[2]["content"][0]["tool_use_id"] == "call_123"


def test_raw_predict_text_response(monkeypatch):
    """Verifies generate() parses text responses correctly via rawPredict."""
    mock_creds = MagicMock()
    mock_creds.token = "fake-token"
    mock_creds.valid = True
    monkeypatch.setattr("harnessfun.providers.vertex_anthropic.verify_gcp_adc", lambda: (mock_creds, "test-proj"))

    provider = VertexAnthropicProvider(project_id="test-proj", location="us-east5")
    provider._client = None

    fake_response_data = {
        "content": [
            {"type": "text", "text": "Hello from Claude on Vertex AI!"}
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_response_data

    with patch.object(provider, "_get_access_token", return_value="fake-token"):
        with patch("httpx.Client.post", return_value=mock_resp):
            response = provider.generate(
                messages=[Message(role="user", content="Hi")],
                tools=[],
                model="claude-3-5-sonnet",
                system_instruction="Be concise."
            )

    assert isinstance(response, ProviderResponse)
    assert response.text == "Hello from Claude on Vertex AI!"
    assert response.tool_calls == []


def test_raw_predict_tool_call_response(monkeypatch):
    """Verifies generate() parses tool_use response blocks correctly via rawPredict."""
    mock_creds = MagicMock()
    mock_creds.token = "fake-token"
    mock_creds.valid = True
    monkeypatch.setattr("harnessfun.providers.vertex_anthropic.verify_gcp_adc", lambda: (mock_creds, "test-proj"))

    provider = VertexAnthropicProvider(project_id="test-proj", location="us-east5")
    provider._client = None

    fake_response_data = {
        "content": [
            {"type": "text", "text": "Let me look up the weather."},
            {
                "type": "tool_use",
                "id": "toolu_01Abc",
                "name": "get_weather",
                "input": {"location": "San Francisco"}
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_response_data

    with patch.object(provider, "_get_access_token", return_value="fake-token"):
        with patch("httpx.Client.post", return_value=mock_resp):
            response = provider.generate(
                messages=[Message(role="user", content="Weather in SF?")],
                tools=[],
                model="claude-3-5-sonnet",
                system_instruction=""
            )

    assert response.text == "Let me look up the weather."
    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert tc.name == "get_weather"
    assert tc.args == {"location": "San Francisco"}
    assert tc.id == "toolu_01Abc"


def test_model_enablement_error_guidance(monkeypatch):
    """Verifies that 403/404 API errors produce actionable Model Garden enablement instructions."""
    mock_creds = MagicMock()
    monkeypatch.setattr("harnessfun.providers.vertex_anthropic.verify_gcp_adc", lambda: (mock_creds, "test-proj"))

    provider = VertexAnthropicProvider(project_id="test-proj", location="us-east5")
    provider._client = None

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = json.dumps({
        "error": {
            "code": 403,
            "message": "Permission denied on resource. Model has not been enabled in Model Garden.",
            "status": "PERMISSION_DENIED"
        }
    })

    with patch.object(provider, "_get_access_token", return_value="fake-token"):
        with patch("httpx.Client.post", return_value=mock_resp):
            with pytest.raises(RuntimeError) as exc_info:
                provider.generate(
                    messages=[Message(role="user", content="Hi")],
                    tools=[],
                    model="claude-3-5-sonnet",
                    system_instruction=""
                )

    error_msg = str(exc_info.value)
    assert "Model Garden" in error_msg
    assert "https://console.cloud.google.com/vertex-ai/model-garden" in error_msg
