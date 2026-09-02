"""Normalized Data Models for harnessfun."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """Represents a tool call requested by the model."""
    id: str
    name: str
    args: Dict[str, Any]


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


@dataclass
class ProviderResponse:
    """Normalized response returned by LLM providers."""
    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


@dataclass
class SessionConfig:
    """Configuration settings for a harness execution session."""
    project_id: str
    location: str = "us-central1"
    active_model: str = "gemini-2.5-flash"
    system_instruction: str = "You are a helpful assistant with access to local tools."
    max_steps: int = 10
