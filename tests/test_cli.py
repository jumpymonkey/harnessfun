"""Tests for harnessfun CLI and streaming renderer."""

import json
from unittest.mock import MagicMock
from click.testing import CliRunner

from harnessfun.cli import cli, stream_turn_to_console
from harnessfun.models import HarnessEvent, SessionConfig


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "harnessfun" in result.output
    assert "auth-check" in result.output
    assert "models" in result.output
    assert "run" in result.output
    assert "chat" in result.output


def test_run_help_has_trace_option():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--trace" in result.output


def test_chat_help_has_trace_option():
    runner = CliRunner()
    result = runner.invoke(cli, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--trace" in result.output


def test_stream_turn_to_console_renders_all_events():
    mock_harness = MagicMock()
    mock_harness.config = SessionConfig(
        project_id="test-proj",
        location="us-central1",
        model_location="global",
        active_model="gemini-2.5-flash",
    )

    events = [
        HarnessEvent(type="step_start", step=1, data={"step": 2, "max_steps": 10, "model": "gemini-2.5-flash"}),
        HarnessEvent(type="model_thought", step=1, data={"thought": "Thinking about computing 2+2..."}),
        HarnessEvent(type="tool_call", step=1, data={"call_id": "call_1", "tool": "calculate", "args": {"expression": "2+2"}}),
        HarnessEvent(type="tool_result", step=1, data={"call_id": "call_1", "tool": "calculate", "output": {"result": 4}, "is_error": False}),
        HarnessEvent(type="tool_result", step=1, data={"call_id": "call_2", "tool": "broken_tool", "output": "Tool error occurred", "is_error": True}),
        HarnessEvent(type="turn_complete", step=1, data={"content": "2 + 2 = 4", "total_steps": 2}),
        HarnessEvent(type="error", step=1, data={"error": "Test error"}),
    ]

    mock_harness.run_turn_stream.return_value = iter(events)

    # Calling stream_turn_to_console should execute cleanly through all event types
    stream_turn_to_console(mock_harness, "calculate 2+2")
    mock_harness.run_turn_stream.assert_called_once_with("calculate 2+2")
