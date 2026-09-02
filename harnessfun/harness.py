"""Universal Execution Harness Engine for harnessfun."""

from typing import List, Optional
from harnessfun.models import Message, ProviderResponse, SessionConfig, ToolResult
from harnessfun.providers.base import BaseLLMProvider
from harnessfun.tools import ToolRegistry, default_registry


class UniversalHarness:
    """Core execution harness managing state, provider calls, and tool loops."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        config: SessionConfig,
        registry: Optional[ToolRegistry] = None
    ):
        self.provider = provider
        self.config = config
        self.registry = registry or default_registry
        self.messages: List[Message] = []

    def clear(self) -> None:
        """Resets the conversation history."""
        self.messages = []

    def set_model(self, model_id: str) -> None:
        """Switches the active model ID dynamically."""
        self.config.active_model = model_id

    def set_system_instruction(self, instruction: str) -> None:
        """Updates system instructions for subsequent turns."""
        self.config.system_instruction = instruction

    def run_turn(self, user_prompt: str) -> str:
        """Executes a multi-step turn for a user prompt."""
        self.messages.append(Message(role="user", content=user_prompt))
        tools_list = self.registry.get_functions_list()

        for step in range(self.config.max_steps):
            response: ProviderResponse = self.provider.generate(
                messages=self.messages,
                tools=tools_list,
                model=self.config.active_model,
                system_instruction=self.config.system_instruction
            )

            # Case A: Provider returned tool calls
            if response.tool_calls:
                self.messages.append(
                    Message(role="assistant", tool_calls=response.tool_calls)
                )

                tool_results: List[ToolResult] = []
                for tc in response.tool_calls:
                    output = self.registry.execute(tc.name, tc.args)
                    tool_results.append(
                        ToolResult(call_id=tc.id, name=tc.name, output=output)
                    )

                self.messages.append(
                    Message(role="tool", tool_results=tool_results)
                )
                # Loop continues so LLM ingests tool results!

            # Case B: Provider returned text answer
            elif response.text:
                self.messages.append(
                    Message(role="assistant", content=response.text)
                )
                return response.text

        return "Harness Error: Exceeded maximum allowed execution steps."
