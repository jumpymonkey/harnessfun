"""Unit tests for UniversalHarness execution loop and tool calling."""

from typing import Callable, List
from harnessfun.harness import UniversalHarness
from harnessfun.models import Message, ProviderResponse, SessionConfig, ToolCall
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
        system_instruction: str
    ) -> ProviderResponse:
        self.call_count += 1
        last_msg = messages[-1]

        # First turn: Request tool call 'add'
        if self.call_count == 1:
            return ProviderResponse(
                tool_calls=[ToolCall(id="call_1", name="add", args={"a": 5, "b": 10})]
            )
        # Second turn: Answer with final text after receiving tool output
        elif self.call_count == 2:
            return ProviderResponse(text="The result of 5 + 10 is 15.")

        return ProviderResponse(text="Fallback")


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
