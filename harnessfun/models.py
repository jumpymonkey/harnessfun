import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HarnessEvent:
    """Represents an execution event emitted by UniversalHarness."""
    type: str  # e.g., "step_start", "model_thought", "tool_call", "tool_result", "turn_complete", "error"
    step: int
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolCall:
    """Represents a tool call requested by the model."""
    id: str
    name: str
    args: Dict[str, Any]
    thought_signature: Optional[Any] = None


@dataclass
class ToolResult:
    """Represents the output from executing a local tool."""
    call_id: str
    name: str
    output: Dict[str, Any]


@dataclass
class Message:
    """Normalized message representation for conversation history."""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    raw_parts: Optional[List[Any]] = None


@dataclass
class ProviderResponse:
    """Normalized response returned by LLM providers."""
    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_parts: Optional[List[Any]] = None


@dataclass
class SessionConfig:
    """Configuration settings for a harness execution session."""
    project_id: str
    location: str = "us-central1"
    model_location: str = "global"
    active_model: str = "gemini-2.5-flash"
    system_instruction: str = (
        "You are a helpful, general-purpose AI assistant. Answer general knowledge "
        "questions using your built-in knowledge. When a prompt requires tools "
        "(such as weather, time, or math calculations), invoke the appropriate tool."
    )
    max_steps: int = 10
