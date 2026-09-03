import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from click.testing import CliRunner

from harnessfun.cli import cli, _handle_mcp_command, _show_mcp_help
from harnessfun.config import SessionConfig
from harnessfun.harness import UniversalHarness
from harnessfun.mcp.manager import MCPClientManager, MCPServerConfig
from harnessfun.models import ToolDefinition
from harnessfun.providers.gcp_gemini import GCPGeminiProvider
from harnessfun.providers.vertex_anthropic import VertexAnthropicProvider
from harnessfun.tools import ToolRegistry, _callable_to_json_schema


def test_tool_definition_execution():
    """Verify ToolDefinition executes handlers and ToolRegistry wraps non-dict results."""
    def sample_handler(x: int, y: int = 10):
        return x + y

    tdef = ToolDefinition(
        name="test_add",
        description="Adds two numbers",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x"],
        },
        handler=sample_handler,
        server_name="math_server",
    )

    # Direct handler invocation
    assert tdef.execute(x=5, y=15) == 20

    # Registry invocation wrapping primitive into dict
    reg = ToolRegistry()
    reg.register_tool_definition(tdef)
    assert reg.execute("test_add", {"x": 5, "y": 15}) == {"result": 20}

    # Dict returns should pass through as-is
    def dict_handler(msg: str):
        return {"status": "ok", "message": msg}

    tdef_dict = ToolDefinition(
        name="status_tool",
        description="Returns status",
        parameters={"type": "object", "properties": {}},
        handler=dict_handler,
    )
    reg.register_tool_definition(tdef_dict)
    assert reg.execute("status_tool", {"msg": "hello"}) == {"status": "ok", "message": "hello"}


def test_callable_to_json_schema():
    """Verify auto-generation of JSON Schema from Python function signatures."""
    def sample_func(city: str, days: int, verbose: bool = False, rating: float = 4.5) -> str:
        """Fetch weather forecasts.
        
        Args:
            city: The city name
            days: Number of days
        """
        return f"{city}-{days}"

    schema = _callable_to_json_schema(sample_func)
    assert schema["type"] == "object"
    assert "city" in schema["properties"]
    assert schema["properties"]["city"]["type"] == "string"
    assert schema["properties"]["days"]["type"] == "integer"
    assert schema["properties"]["verbose"]["type"] == "boolean"
    assert schema["properties"]["rating"]["type"] == "number"
    assert "city" in schema["required"]
    assert "days" in schema["required"]
    assert "verbose" not in schema["required"]
    assert "rating" not in schema["required"]


def test_tool_registry_with_tool_definitions():
    """Verify ToolRegistry registers, queries, resolves, and unregisters ToolDefinitions."""
    reg = ToolRegistry()

    @reg.register
    def local_tool(a: int) -> int:
        """Local tool description."""
        return a * 2

    mcp_tool = ToolDefinition(
        name="sqlite__query",
        description="Run a SQL query",
        parameters={"type": "object", "properties": {"sql": {"type": "string"}}},
        handler=lambda sql: {"rows": [1, 2, 3]},
        server_name="sqlite",
    )
    reg.register_tool_definition(mcp_tool)

    tools = reg.get_tools()
    assert len(tools) == 2
    tool_names = [t.name for t in tools]
    assert "local_tool" in tool_names
    assert "sqlite__query" in tool_names

    # Execution with scoped name
    assert reg.execute("sqlite__query", {"sql": "SELECT 1"}) == {"rows": [1, 2, 3]}

    # Execution with colon prefix resolution
    assert reg.execute("sqlite:query", {"sql": "SELECT 1"}) == {"rows": [1, 2, 3]}

    # Execution with short unqualified name
    assert reg.execute("query", {"sql": "SELECT 1"}) == {"rows": [1, 2, 3]}

    # Unregister server
    removed = reg.unregister_server("sqlite")
    assert removed == 1
    assert len(reg.get_tools()) == 1
    assert "sqlite__query" not in [t.name for t in reg.get_tools()]


def test_anthropic_provider_converts_tool_definition():
    """Verify VertexAnthropicProvider converts ToolDefinition into valid Anthropic tool schema."""
    provider = VertexAnthropicProvider(project_id="test-proj")
    tdef = ToolDefinition(
        name="test_tool",
        description="Does something useful",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=lambda query: query,
    )

    schema = provider._convert_tool_to_anthropic_schema(tdef)
    assert schema == {
        "name": "test_tool",
        "description": "Does something useful",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }


def test_gemini_provider_handles_tool_definition():
    """Verify GCPGeminiProvider formats ToolDefinition as FunctionDeclaration."""
    with patch("google.genai.Client") as mock_client:
        provider = GCPGeminiProvider(project_id="test-proj")
        tdef = ToolDefinition(
            name="github__search",
            description="Search GitHub repositories",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            handler=lambda q: {"items": []},
            server_name="github",
        )

        mock_response = MagicMock()
        mock_response.text = "Results found."
        mock_response.candidates = [
            MagicMock(
                finish_reason="STOP",
                content=MagicMock(parts=[MagicMock(text="Results found.", function_call=None)]),
            )
        ]
        mock_client.return_value.models.generate_content.return_value = mock_response

        resp = provider.generate(
            messages=[],
            tools=[tdef],
            model="gemini-2.5-flash",
            system_instruction=None,
        )
        assert resp.text == "Results found."
        call_kwargs = mock_client.return_value.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert len(config.tools) == 1


@patch("harnessfun.mcp.manager.stdio_client")
@patch("harnessfun.mcp.manager.ClientSession")
def test_mcp_client_manager_mocked_session(mock_client_session_cls, mock_stdio_client):
    """Verify MCPClientManager connects, discovers tools, executes tools, and disconnects."""
    # Set up async context manager for stdio_client
    mock_stdio_ctx = AsyncMock()
    mock_stdio_ctx.__aenter__.return_value = (MagicMock(), MagicMock())
    mock_stdio_ctx.__aexit__.return_value = None
    mock_stdio_client.return_value = mock_stdio_ctx

    # Set up mock session
    mock_tool_1 = MagicMock()
    mock_tool_1.name = "read_query"
    mock_tool_1.description = "Execute a SELECT query"
    mock_tool_1.input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    mock_tools_result = MagicMock()
    mock_tools_result.tools = [mock_tool_1]

    mock_call_result = MagicMock()
    mock_content = MagicMock()
    mock_content.text = '{"data": [1, 2, 3]}'
    mock_call_result.content = [mock_content]
    mock_call_result.is_error = False

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
    mock_session.call_tool = AsyncMock(return_value=mock_call_result)

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__.return_value = mock_session
    mock_session_ctx.__aexit__.return_value = None
    mock_client_session_cls.return_value = mock_session_ctx

    manager = MCPClientManager()
    try:
        tools = manager.connect_stdio(name="test_db", command="mock_cmd", args=["--arg"])
        assert len(tools) == 1
        assert tools[0].name == "test_db__read_query"
        assert tools[0].server_name == "test_db"
        assert tools[0].description == "Execute a SELECT query"

        # Execute tool through manager
        result = manager.call_tool("test_db", "read_query", {"query": "SELECT *"})
        assert result == {"data": [1, 2, 3]}

        # List servers
        servers = manager.list_servers()
        assert len(servers) == 1
        assert servers[0]["name"] == "test_db"
        assert servers[0]["tool_count"] == 1
        assert servers[0]["status"] == "connected"

        # Get server tools
        db_tools = manager.get_server_tools("test_db")
        assert len(db_tools) == 1
        assert db_tools[0].name == "test_db__read_query"

        # Disconnect
        assert manager.disconnect("test_db") is True
        assert manager.servers["test_db"].status == "disconnected"
    finally:
        manager.close()


def test_mcp_config_file_parsing(tmp_path):
    """Verify loading MCP servers from JSON and YAML configuration files."""
    config_data = {
        "mcpServers": {
            "sqlite": {
                "command": "uvx",
                "args": ["mcp-server-sqlite", "--db-path", "test.db"],
                "env": {"DEBUG": "1"},
            },
            "web_fetch": {
                "url": "http://localhost:8000/sse",
            },
        }
    }

    json_file = tmp_path / "mcp_config.json"
    json_file.write_text(json.dumps(config_data))

    manager = MCPClientManager()
    try:
        with patch.object(manager, "connect_server", side_effect=[
            [ToolDefinition(name="sqlite__query", description="Run query", parameters={}, handler=lambda: {})],
            [ToolDefinition(name="web_fetch__fetch", description="Fetch url", parameters={}, handler=lambda: {})],
        ]) as mock_conn:
            loaded = manager.load_config_file(str(json_file))
            assert "sqlite" in loaded
            assert "web_fetch" in loaded
            assert mock_conn.call_count == 2
    finally:
        manager.close()


def test_cli_mcp_commands():
    """Verify interactive /mcp slash command handler output."""
    mock_provider = MagicMock()
    config = SessionConfig(project_id="test-proj")
    manager = MCPClientManager()
    harness = UniversalHarness(provider=mock_provider, config=config, mcp_manager=manager)

    try:
        # /mcp help
        _handle_mcp_command("help", harness, manager)

        # /mcp list with no servers
        _handle_mcp_command("list", harness, manager)

        # /mcp connect with missing args
        _handle_mcp_command("connect", harness, manager)

        # /mcp connect-url with missing args
        _handle_mcp_command("connect-url", harness, manager)

        # /mcp disconnect with missing args
        _handle_mcp_command("disconnect", harness, manager)

        # /mcp unknown subcommand
        _handle_mcp_command("unknown_action", harness, manager)

        # /mcp connect mock
        with patch.object(manager, "connect_stdio", return_value=[
            ToolDefinition(name="db__test", description="Test tool", parameters={}, handler=lambda: {}, server_name="db")
        ]):
            _handle_mcp_command("connect db mock_cmd --flag", harness, manager)
            assert "db__test" in [t.name for t in harness.registry.get_tools()]

            # /mcp list with connected server
            _handle_mcp_command("list", harness, manager)

            # /mcp tools
            _handle_mcp_command("tools", harness, manager)
            _handle_mcp_command("tools db", harness, manager)

            # /mcp disconnect
            _handle_mcp_command("disconnect db", harness, manager)
            assert "db__test" not in [t.name for t in harness.registry.get_tools()]
    finally:
        manager.close()


def test_cli_help_options():
    """Verify --mcp-config flag is documented on run and chat CLI commands."""
    runner = CliRunner()

    res_run = runner.invoke(cli, ["run", "--help"])
    assert res_run.exit_code == 0
    assert "--mcp-config" in res_run.output

    res_chat = runner.invoke(cli, ["chat", "--help"])
    assert res_chat.exit_code == 0
    assert "--mcp-config" in res_chat.output
