"""Universal Execution Harness Engine for harnessfun with Event Pipeline."""

import json
from typing import Any, Callable, Dict, Generator, List, Optional
from harnessfun.models import (
    HarnessEvent,
    Message,
    ProviderResponse,
    SessionConfig,
    ToolResult,
)
from harnessfun.providers.base import BaseLLMProvider
from harnessfun.tools import ToolRegistry, default_registry


class UniversalHarness:
    """Core execution harness managing state, provider calls, tool loops, and event streams."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        config: SessionConfig,
        registry: Optional[ToolRegistry] = None,
        on_event: Optional[Callable[[HarnessEvent], None]] = None,
    ):
        self.provider = provider
        self.config = config
        self.registry = registry or default_registry
        self.messages: List[Message] = []
        self.events: List[HarnessEvent] = []
        self.on_event = on_event
        self._listeners: List[Callable[[HarnessEvent], None]] = []
        if on_event:
            self._listeners.append(on_event)

    def add_event_listener(self, listener: Callable[[HarnessEvent], None]) -> None:
        """Registers a callback listener for harness events."""
        self._listeners.append(listener)

    def clear(self) -> None:
        """Resets the conversation history and trajectory events."""
        self.messages = []
        self.events = []

    def set_model(self, model_id: str) -> None:
        """Switches the active model ID dynamically."""
        self.config.active_model = model_id

    def set_system_instruction(self, instruction: str) -> None:
        """Updates system instructions for subsequent turns."""
        self.config.system_instruction = instruction

    def _emit(self, event: HarnessEvent) -> HarnessEvent:
        """Appends event to internal history and notifies all listeners."""
        self.events.append(event)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass
        return event

    def run_turn_stream(self, user_prompt: str) -> Generator[HarnessEvent, None, str]:
        """Executes a multi-step turn, yielding HarnessEvents in real time."""
        self.messages.append(Message(role="user", content=user_prompt))
        tools_list = self.registry.get_functions_list()

        for step in range(self.config.max_steps):
            yield self._emit(
                HarnessEvent(
                    type="step_start",
                    step=step,
                    data={
                        "step": step + 1,
                        "max_steps": self.config.max_steps,
                        "model": self.config.active_model,
                    },
                )
            )

            try:
                response: ProviderResponse = self.provider.generate(
                    messages=self.messages,
                    tools=tools_list,
                    model=self.config.active_model,
                    system_instruction=self.config.system_instruction,
                )
            except Exception as e:
                err_msg = f"Provider Error during step {step + 1}: {str(e)}"
                yield self._emit(
                    HarnessEvent(
                        type="error",
                        step=step,
                        data={"error": err_msg, "exception": str(e)},
                    )
                )
                return err_msg

            # Emit thought event if model produced rationale alongside tool calls
            if response.text and response.tool_calls:
                yield self._emit(
                    HarnessEvent(
                        type="model_thought",
                        step=step,
                        data={"thought": response.text},
                    )
                )

            # Case A: Provider returned tool calls
            if response.tool_calls:
                self.messages.append(
                    Message(
                        role="assistant",
                        content=response.text,
                        tool_calls=response.tool_calls,
                    )
                )

                tool_results: List[ToolResult] = []
                for tc in response.tool_calls:
                    yield self._emit(
                        HarnessEvent(
                            type="tool_call",
                            step=step,
                            data={"call_id": tc.id, "tool": tc.name, "args": tc.args},
                        )
                    )

                    output = self.registry.execute(tc.name, tc.args)
                    is_error = isinstance(output, dict) and "error" in output

                    yield self._emit(
                        HarnessEvent(
                            type="tool_result",
                            step=step,
                            data={
                                "call_id": tc.id,
                                "tool": tc.name,
                                "output": output,
                                "is_error": is_error,
                            },
                        )
                    )

                    tool_results.append(
                        ToolResult(call_id=tc.id, name=tc.name, output=output)
                    )

                self.messages.append(
                    Message(role="tool", tool_results=tool_results)
                )
                # Loop continues so LLM ingests tool results

            # Case B: Provider returned text answer (final completion)
            elif response.text:
                self.messages.append(
                    Message(role="assistant", content=response.text)
                )
                yield self._emit(
                    HarnessEvent(
                        type="turn_complete",
                        step=step,
                        data={
                            "content": response.text,
                            "total_steps": step + 1,
                        },
                    )
                )
                return response.text

            else:
                err_msg = (
                    f"Provider returned an empty response with no content or tool calls at step {step + 1}."
                )
                yield self._emit(
                    HarnessEvent(
                        type="error",
                        step=step,
                        data={"error": err_msg},
                    )
                )
                return err_msg

        # Exceeded step limit
        err_msg = "Harness Error: Exceeded maximum allowed execution steps."
        yield self._emit(
            HarnessEvent(
                type="error",
                step=self.config.max_steps,
                data={
                    "error": err_msg,
                    "max_steps": self.config.max_steps,
                },
            )
        )
        return err_msg

    def run_turn(self, user_prompt: str) -> str:
        """Executes a multi-step turn for a user prompt and returns the final string response."""
        final_output = ""
        for event in self.run_turn_stream(user_prompt):
            if event.type == "turn_complete":
                final_output = event.data.get("content", "")
            elif event.type == "error":
                final_output = event.data.get("error", "Harness Error: Execution failed.")
        return final_output

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Returns serialized trajectory events for analysis or export."""
        return [
            {
                "type": e.type,
                "step": e.step,
                "data": e.data,
                "timestamp": e.timestamp,
            }
            for e in self.events
        ]

    def export_trajectory_jsonl(self, filepath: str) -> None:
        """Exports all session events as a JSONL trajectory log."""
        with open(filepath, "w", encoding="utf-8") as f:
            for event in self.events:
                record = {
                    "type": event.type,
                    "step": event.step,
                    "data": event.data,
                    "timestamp": event.timestamp,
                }
                f.write(json.dumps(record) + "\n")
