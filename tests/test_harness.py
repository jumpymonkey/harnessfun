"""Unit tests for UniversalHarness execution loop, event pipeline, and tool calling."""

import json
from typing import Callable, List
from harnessfun.harness import UniversalHarness
from harnessfun.models import HarnessEvent, Message, ProviderResponse, SessionConfig, ToolCall
from harnessfun.providers.base import BaseLLMProvider
from harnessfun.tools import ToolRegistry


class MockProvider(BaseLLMProvider):
    """Mock LLM Provider for unit testing."""

    def __init__(self):
        self.call_count = 0

    def list_models(self) -> List[str]:
        return ["mock-model-1", "mock-model-2"]

    def generate(
        self,
        messages: List[Message],
        tools: List[Callable],
        model: str,
        system_instruction: str,
    ) -> ProviderResponse:
        self.call_count += 1

        # First turn: Request tool call 'add'
        if self.call_count == 1:
            return ProviderResponse(
                tool_calls=[ToolCall(id="call_1", name="add", args={"a": 5, "b": 10})]
            )
        # Second turn: Answer with final text after receiving tool output
        elif self.call_count == 2:
            return ProviderResponse(text="The result of 5 + 10 is 15.")

        return ProviderResponse(text="Fallback")


class MockThoughtAndToolProvider(BaseLLMProvider):
    """Mock provider that returns both thought text and tool calls."""

    def __init__(self):
        self.call_count = 0

    def list_models(self) -> List[str]:
        return ["mock-model-thought"]

    def generate(
        self,
        messages: List[Message],
        tools: List[Callable],
        model: str,
        system_instruction: str,
    ) -> ProviderResponse:
        self.call_count += 1
        if self.call_count == 1:
            return ProviderResponse(
                text="I should add 3 and 7 using the add tool.",
                tool_calls=[ToolCall(id="call_t1", name="add", args={"a": 3, "b": 7})],
            )
        return ProviderResponse(text="3 + 7 equals 10.")


def test_harness_tool_execution_loop():
    """Tests that the harness executes tools and feeds output back into the loop."""
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        return a + b

    mock_provider = MockProvider()
    config = SessionConfig(project_id="test-project", active_model="mock-model-1")
    harness = UniversalHarness(provider=mock_provider, config=config, registry=registry)

    result = harness.run_turn("Add 5 and 10")

    assert result == "The result of 5 + 10 is 15."
    assert mock_provider.call_count == 2
    assert len(harness.messages) == 4  # User -> Assistant ToolCall -> Tool Result -> Assistant Text
    # Check that events were recorded
    assert len(harness.events) >= 4
    event_types = [e.type for e in harness.events]
    assert event_types == ["step_start", "tool_call", "tool_result", "step_start", "turn_complete"]


def test_harness_run_turn_stream_events():
    """Tests yielding events in real-time through run_turn_stream."""
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        return a + b

    mock_provider = MockProvider()
    config = SessionConfig(project_id="test-project", active_model="mock-model-1")
    harness = UniversalHarness(provider=mock_provider, config=config, registry=registry)

    streamed_events: List[HarnessEvent] = list(harness.run_turn_stream("Calculate 5 + 10"))

    assert len(streamed_events) == 5
    assert streamed_events[0].type == "step_start"
    assert streamed_events[0].data["step"] == 1

    assert streamed_events[1].type == "tool_call"
    assert streamed_events[1].data["tool"] == "add"
    assert streamed_events[1].data["args"] == {"a": 5, "b": 10}

    assert streamed_events[2].type == "tool_result"
    assert streamed_events[2].data["output"] == {"result": 15}
    assert streamed_events[2].data["is_error"] is False

    assert streamed_events[3].type == "step_start"
    assert streamed_events[3].data["step"] == 2

    assert streamed_events[4].type == "turn_complete"
    assert streamed_events[4].data["content"] == "The result of 5 + 10 is 15."
    assert streamed_events[4].data["total_steps"] == 2


def test_harness_model_thought_preservation():
    """Tests that model thoughts alongside tool calls emit a model_thought event and preserve text."""
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        return a + b

    mock_provider = MockThoughtAndToolProvider()
    config = SessionConfig(project_id="test-project", active_model="mock-thought-model")
    harness = UniversalHarness(provider=mock_provider, config=config, registry=registry)

    events = list(harness.run_turn_stream("Add 3 and 7"))
    event_types = [e.type for e in events]

    assert "model_thought" in event_types
    thought_event = next(e for e in events if e.type == "model_thought")
    assert "add 3 and 7" in thought_event.data["thought"]

    # Verify that the assistant message preserved the content text
    assistant_msg = harness.messages[1]
    assert assistant_msg.role == "assistant"
    assert assistant_msg.content == "I should add 3 and 7 using the add tool."
    assert len(assistant_msg.tool_calls) == 1


def test_harness_tool_error_event():
    """Tests that tool execution errors are captured and emitted as tool_result with is_error=True."""
    registry = ToolRegistry()

    @registry.register
    def failing_tool(x: int) -> int:
        raise ValueError("Invalid parameter x")

    class ToolFailProvider(BaseLLMProvider):
        def __init__(self):
            self.calls = 0

        def list_models(self) -> List[str]:
            return ["fail-model"]

        def generate(self, messages, tools, model, system_instruction):
            self.calls += 1
            if self.calls == 1:
                return ProviderResponse(
                    tool_calls=[ToolCall(id="call_err", name="failing_tool", args={"x": 10})]
                )
            return ProviderResponse(text="Encountered an error with the tool.")

    harness = UniversalHarness(
        provider=ToolFailProvider(),
        config=SessionConfig(project_id="test-project"),
        registry=registry,
    )

    events = list(harness.run_turn_stream("Run tool"))
    result_event = next(e for e in events if e.type == "tool_result")

    assert result_event.data["is_error"] is True
    assert "error" in result_event.data["output"]


def test_harness_max_steps_exceeded():
    """Tests that infinite tool loops cleanly terminate and emit an error event."""
    registry = ToolRegistry()

    @registry.register
    def dummy() -> str:
        return "loop"

    class InfiniteToolProvider(BaseLLMProvider):
        def list_models(self) -> List[str]:
            return ["loop-model"]

        def generate(self, messages, tools, model, system_instruction):
            return ProviderResponse(
                tool_calls=[ToolCall(id="c", name="dummy", args={})]
            )

    config = SessionConfig(project_id="test-project", max_steps=3)
    harness = UniversalHarness(
        provider=InfiniteToolProvider(),
        config=config,
        registry=registry,
    )

    result = harness.run_turn("Loop forever")

    assert "Exceeded maximum allowed execution steps" in result
    error_event = harness.events[-1]
    assert error_event.type == "error"
    assert error_event.data["max_steps"] == 3


def test_harness_provider_exception():
    """Tests graceful handling when provider raises an exception."""
    class CrashingProvider(BaseLLMProvider):
        def list_models(self) -> List[str]:
            return ["crash-model"]

        def generate(self, messages, tools, model, system_instruction):
            raise ConnectionError("GCP API Network Timeout")

    harness = UniversalHarness(
        provider=CrashingProvider(),
        config=SessionConfig(project_id="test-project"),
    )

    result = harness.run_turn("Hello")

    assert "Provider Error during step 1" in result
    assert "GCP API Network Timeout" in result
    assert harness.events[-1].type == "error"


def test_harness_event_listeners():
    """Tests registering on_event and add_event_listener callbacks."""
    registry = ToolRegistry()
    mock_provider = MockProvider()
    config = SessionConfig(project_id="test-project")

    collected_via_init: List[HarnessEvent] = []
    collected_via_add: List[HarnessEvent] = []

    harness = UniversalHarness(
        provider=mock_provider,
        config=config,
        registry=registry,
        on_event=collected_via_init.append,
    )
    harness.add_event_listener(collected_via_add.append)

    harness.run_turn("Add 5 and 10")

    assert len(collected_via_init) > 0
    assert len(collected_via_init) == len(collected_via_add)
    assert collected_via_init[0].type == "step_start"


def test_harness_trajectory_export(tmp_path):
    """Tests exporting trajectory to structured JSONL."""
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        return a + b

    mock_provider = MockProvider()
    config = SessionConfig(project_id="test-project")
    harness = UniversalHarness(provider=mock_provider, config=config, registry=registry)

    harness.run_turn("Add 5 and 10")

    trajectory = harness.get_trajectory()
    assert len(trajectory) == 5
    assert trajectory[0]["type"] == "step_start"
    assert "timestamp" in trajectory[0]

    export_file = tmp_path / "trajectory.jsonl"
    harness.export_trajectory_jsonl(str(export_file))

    assert export_file.exists()
    lines = export_file.read_text().strip().split("\n")
    assert len(lines) == 5
    parsed_first = json.loads(lines[0])
    assert parsed_first["type"] == "step_start"


def test_harness_clear():
    """Tests that clear() resets messages and events."""
    mock_provider = MockProvider()
    config = SessionConfig(project_id="test-project")
    harness = UniversalHarness(provider=mock_provider, config=config)

    harness.run_turn("Test")
    assert len(harness.messages) > 0
    assert len(harness.events) > 0

    harness.clear()
    assert len(harness.messages) == 0
    assert len(harness.events) == 0
