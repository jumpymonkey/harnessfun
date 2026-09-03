"""Unit tests for GCPGeminiProvider and Gemini tool-calling compatibility."""

from unittest.mock import MagicMock
import pytest
from google.genai import types

from harnessfun.harness import UniversalHarness
from harnessfun.models import Message, ProviderResponse, SessionConfig, ToolCall, ToolResult
from harnessfun.providers.gcp_gemini import GCPGeminiProvider
from harnessfun.tools import ToolRegistry, get_weather


def test_gemini_provider_extracts_thought_signature_and_parts(monkeypatch):
    """Verifies that GCPGeminiProvider extracts thought_signature from candidate parts."""
    mock_client = MagicMock()
    monkeypatch.setattr("google.genai.Client", lambda **kw: mock_client)

    provider = GCPGeminiProvider(project_id="test-proj")

    # Simulate Gemini API response with thought and function call with thought_signature
    p_thought = types.Part(text="Analyzing weather request...", thought=True)
    p_call = types.Part(
        function_call=types.FunctionCall(name="default_api:get_weather", args={"location": "Paris"}),
        thought_signature=b"test_thought_signature_bytes_123"
    )
    mock_resp = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[p_thought, p_call])
            )
        ]
    )
    mock_client.models.generate_content.return_value = mock_resp

    resp = provider.generate(
        messages=[Message(role="user", content="What's the weather in Paris?")],
        tools=[get_weather],
        model="gemini-2.5-flash",
        system_instruction="You are a helpful assistant."
    )

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "default_api:get_weather"
    assert resp.tool_calls[0].args == {"location": "Paris"}
    assert resp.tool_calls[0].thought_signature == b"test_thought_signature_bytes_123"
    assert resp.raw_parts is not None
    assert len(resp.raw_parts) == 2
    assert resp.text == "Analyzing weather request..."


def test_gemini_provider_reconstructs_assistant_content_with_thought_signature(monkeypatch):
    """Verifies that subsequent generate() calls pass thought_signature back in types.Content."""
    mock_client = MagicMock()
    monkeypatch.setattr("google.genai.Client", lambda **kw: mock_client)

    provider = GCPGeminiProvider(project_id="test-proj")

    # Mock response for step 2 (answering after tool response)
    final_part = types.Part.from_text(text="The weather in Paris is Sunny and 22°C.")
    mock_client.models.generate_content.return_value = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[final_part])
            )
        ]
    )

    # Reconstruct history: user -> assistant (tool call with thought_signature) -> tool result
    tc = ToolCall(
        id="call_0",
        name="default_api:get_weather",
        args={"location": "Paris"},
        thought_signature=b"signature_token_abc"
    )
    messages = [
        Message(role="user", content="Weather in Paris?"),
        Message(role="assistant", tool_calls=[tc]),
        Message(role="tool", tool_results=[ToolResult(call_id="call_0", name="default_api:get_weather", output={"temp": "22C"})])
    ]

    provider.generate(
        messages=messages,
        tools=[get_weather],
        model="gemini-2.5-flash",
        system_instruction="Be helpful."
    )

    # Verify contents sent to mock_client.models.generate_content
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    contents = call_kwargs["contents"]
    assert len(contents) == 3

    # Check turn 2 (model assistant turn)
    model_content = contents[1]
    assert model_content.role == "model"
    assert len(model_content.parts) == 1
    assert model_content.parts[0].function_call.name == "default_api:get_weather"
    assert model_content.parts[0].thought_signature == b"signature_token_abc"


def test_gemini_provider_reconstructs_assistant_content_with_raw_parts(monkeypatch):
    """Verifies that exact raw parts are preserved and sent if present on Message."""
    mock_client = MagicMock()
    monkeypatch.setattr("google.genai.Client", lambda **kw: mock_client)

    provider = GCPGeminiProvider(project_id="test-proj")

    mock_client.models.generate_content.return_value = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part.from_text(text="Done")])
            )
        ]
    )

    raw_part_1 = types.Part(text="Let me check...", thought=True)
    raw_part_2 = types.Part(
        function_call=types.FunctionCall(name="get_weather", args={"location": "Tokyo"}),
        thought_signature=b"tokyo_signature"
    )

    messages = [
        Message(role="user", content="Weather in Tokyo?"),
        Message(role="assistant", tool_calls=[ToolCall(id="0", name="get_weather", args={})], raw_parts=[raw_part_1, raw_part_2]),
        Message(role="tool", tool_results=[ToolResult(call_id="0", name="get_weather", output={"condition": "Rain"})])
    ]

    provider.generate(
        messages=messages,
        tools=[get_weather],
        model="gemini-2.5-flash",
        system_instruction="Be helpful."
    )

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    contents = call_kwargs["contents"]
    model_content = contents[1]

    # Verify that raw_parts were reused directly
    assert model_content.parts == [raw_part_1, raw_part_2]
    assert model_content.parts[1].thought_signature == b"tokyo_signature"


def test_gemini_provider_wraps_non_dict_tool_output(monkeypatch):
    """Verifies that primitive tool outputs (strings, ints) are wrapped in dicts for Gemini types."""
    mock_client = MagicMock()
    monkeypatch.setattr("google.genai.Client", lambda **kw: mock_client)

    provider = GCPGeminiProvider(project_id="test-proj")
    mock_client.models.generate_content.return_value = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part.from_text(text="Ok")]))]
    )

    messages = [
        Message(role="tool", tool_results=[ToolResult(call_id="0", name="calc", output="42")])
    ]

    provider.generate(
        messages=messages,
        tools=[],
        model="gemini-2.5-flash",
        system_instruction="Be helpful."
    )

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    tool_content = call_kwargs["contents"][0]
    assert tool_content.role == "user"
    assert tool_content.parts[0].function_response.response == {"result": "42"}


def test_namespaced_tool_execution():
    """Verifies that ToolRegistry executes tools prefixed with default_api: or similar namespaces."""
    registry = ToolRegistry()

    @registry.register
    def get_weather(location: str):
        return {"temp": "25C", "location": location}

    # Test exact call
    res1 = registry.execute("get_weather", {"location": "London"})
    assert res1["temp"] == "25C"

    # Test namespaced call as returned by Gemini API
    res2 = registry.execute("default_api:get_weather", {"location": "London"})
    assert res2["temp"] == "25C"

    # Test non-existent tool
    res3 = registry.execute("default_api:unknown_tool", {})
    assert "error" in res3


def test_end_to_end_gemini_multi_turn_with_thought_signature(monkeypatch):
    """Simulates a full 2-step tool calling loop with UniversalHarness and GCPGeminiProvider."""
    mock_client = MagicMock()
    monkeypatch.setattr("google.genai.Client", lambda **kw: mock_client)

    provider = GCPGeminiProvider(project_id="test-proj")

    step1_part = types.Part(
        function_call=types.FunctionCall(name="default_api:get_weather", args={"location": "Paris"}),
        thought_signature=b"valid_thought_signature_step1"
    )
    step1_resp = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[step1_part]))]
    )

    step2_part = types.Part.from_text(text="It is currently 22°C and Sunny in Paris.")
    step2_resp = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[step2_part]))]
    )

    # Configure mock responses for step 1 and step 2
    mock_client.models.generate_content.side_effect = [step1_resp, step2_resp]

    registry = ToolRegistry()
    registry.register(get_weather)

    config = SessionConfig(project_id="test-proj", active_model="gemini-2.5-flash")
    harness = UniversalHarness(provider=provider, config=config, registry=registry)

    final_answer = harness.run_turn("What is the weather in Paris?")

    assert final_answer == "It is currently 22°C and Sunny in Paris."
    assert mock_client.models.generate_content.call_count == 2

    # Verify that in step 2, the assistant's previous message sent to Gemini had the thought_signature intact!
    second_call_contents = mock_client.models.generate_content.call_args_list[1].kwargs["contents"]
    assistant_history_content = second_call_contents[1]
    assert assistant_history_content.role == "model"
    fc_part = assistant_history_content.parts[0]
    assert fc_part.thought_signature == b"valid_thought_signature_step1"
