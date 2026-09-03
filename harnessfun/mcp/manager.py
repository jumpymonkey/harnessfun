"""MCP Client Manager for establishing and orchestrating MCP server sessions."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional
import yaml

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    HAS_MCP_SDK = True
except ImportError:
    HAS_MCP_SDK = False

from harnessfun.models import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    transport: str = "stdio"  # "stdio" or "sse"
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


@dataclass
class MCPServerSession:
    """Maintains active connection and discovered tools for an MCP server."""
    config: MCPServerConfig
    status: str = "disconnected"  # "connected", "disconnected", "error"
    error: Optional[str] = None
    tools: List[ToolDefinition] = field(default_factory=list)
    _session: Optional[Any] = None
    _exit_stack: Optional[AsyncExitStack] = None


class MCPClientManager:
    """Manages connections to external Model Context Protocol (MCP) servers."""

    def __init__(self):
        if not HAS_MCP_SDK:
            raise ImportError(
                "The 'mcp' Python package is required for MCP server integration. "
                "Install it via 'pip install mcp>=1.0.0'."
            )

        self.servers: Dict[str, MCPServerSession] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="HarnessMCPEventLoop")
        self._thread.start()
        self._closed = False

    def _run_loop(self) -> None:
        """Runs the background asyncio event loop dedicated to MCP streams."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coroutine(self, coro: Any, timeout: float = 30.0) -> Any:
        """Executes an async coroutine synchronously on the background MCP loop."""
        if self._closed or not self._loop.is_running():
            raise RuntimeError("MCPClientManager background event loop is not running.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def connect_server(self, config: MCPServerConfig) -> List[ToolDefinition]:
        """Connects to an MCP server according to its configuration and returns discovered tools."""
        return self._run_coroutine(self._connect_server_async(config))

    def connect_stdio(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> List[ToolDefinition]:
        """Connects to a local stdio-based MCP server."""
        cfg = MCPServerConfig(
            name=name,
            transport="stdio",
            command=command,
            args=args or [],
            env=env,
            cwd=cwd,
        )
        return self.connect_server(cfg)

    def connect_sse(
        self,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[ToolDefinition]:
        """Connects to a remote SSE/HTTP-based MCP server."""
        cfg = MCPServerConfig(
            name=name,
            transport="sse",
            url=url,
            headers=headers,
        )
        return self.connect_server(cfg)

    async def _connect_server_async(self, config: MCPServerConfig) -> List[ToolDefinition]:
        """Async implementation of MCP server connection handshake and tool discovery."""
        name = config.name

        # If already connected under this name, disconnect first
        if name in self.servers and self.servers[name].status == "connected":
            await self._disconnect_async(name)

        server_session = MCPServerSession(config=config)
        self.servers[name] = server_session
        exit_stack = AsyncExitStack()
        server_session._exit_stack = exit_stack

        try:
            if config.transport == "sse":
                if not config.url:
                    raise ValueError(f"MCP server '{name}' missing 'url' for SSE transport.")
                streams = await exit_stack.enter_async_context(
                    sse_client(url=config.url, headers=config.headers)
                )
            else:
                # Default: stdio transport
                if not config.command:
                    raise ValueError(f"MCP server '{name}' missing 'command' for stdio transport.")
                merged_env = os.environ.copy()
                if config.env:
                    merged_env.update(config.env)

                params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env=merged_env,
                    cwd=config.cwd,
                )
                streams = await exit_stack.enter_async_context(stdio_client(params))

            read_stream, write_stream = streams
            client_session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            server_session._session = client_session

            # Perform MCP handshake
            await client_session.initialize()

            # Query list of tools exposed by the server
            tools_result = await client_session.list_tools()
            discovered_tools: List[ToolDefinition] = []

            for t in tools_result.tools:
                scoped_name = f"{name}__{t.name}"
                desc = t.description or f"MCP tool '{t.name}' provided by '{name}'"
                
                # Extract input schema
                if isinstance(t.input_schema, dict):
                    schema = t.input_schema
                elif hasattr(t.input_schema, "model_dump"):
                    schema = t.input_schema.model_dump()
                else:
                    schema = {"type": "object", "properties": {}}

                # Construct handler proxy
                tool_def = ToolDefinition(
                    name=scoped_name,
                    description=desc,
                    parameters=schema,
                    handler=self._build_tool_handler(name, t.name),
                    server_name=name,
                )
                discovered_tools.append(tool_def)

            server_session.tools = discovered_tools
            server_session.status = "connected"
            server_session.error = None
            return list(discovered_tools)

        except Exception as e:
            server_session.status = "error"
            server_session.error = str(e)
            try:
                await exit_stack.aclose()
            except Exception:
                pass
            raise RuntimeError(f"Failed to connect to MCP server '{name}': {str(e)}") from e

    def _build_tool_handler(self, server_name: str, original_tool_name: str) -> Callable[..., Dict[str, Any]]:
        """Creates a synchronous execution proxy callable for a specific MCP tool."""
        def handler(**kwargs: Any) -> Dict[str, Any]:
            return self.call_tool(server_name, original_tool_name, kwargs)
        return handler

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Executes a tool on an active MCP server session."""
        server = self.servers.get(server_name)
        if not server or server.status != "connected" or not server._session:
            return {"error": f"MCP server '{server_name}' is not currently connected."}

        try:
            return self._run_coroutine(
                self._call_tool_async(server._session, tool_name, arguments),
                timeout=timeout,
            )
        except Exception as e:
            return {"error": f"MCP tool execution failed ({server_name}__{tool_name}): {str(e)}"}

    async def _call_tool_async(
        self,
        session: Any,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Async implementation of session.call_tool with response normalization."""
        result = await session.call_tool(name=tool_name, arguments=arguments)

        # Check for error flags
        if getattr(result, "is_error", False):
            err_texts = [c.text for c in getattr(result, "content", []) if hasattr(c, "text")]
            err_msg = "\n".join(err_texts) if err_texts else "MCP tool returned an error."
            return {"error": err_msg}

        # Extract content blocks (text, image, json)
        content_items = getattr(result, "content", [])
        text_outputs = []
        for item in content_items:
            if hasattr(item, "text"):
                text_outputs.append(item.text)
            elif hasattr(item, "data"):
                mime = getattr(item, "mimeType", "application/octet-stream")
                text_outputs.append(f"[Binary Content: {mime}]")
            else:
                text_outputs.append(str(item))

        raw_output = "\n".join(text_outputs)

        # If the output text is valid JSON, parse to dictionary or list
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, dict):
                return parsed
            elif isinstance(parsed, list):
                return {"result": parsed}
        except Exception:
            pass

        return {"result": raw_output}

    def disconnect(self, name: str) -> bool:
        """Disconnects a named MCP server and releases all underlying resources."""
        if name not in self.servers:
            return False
        return self._run_coroutine(self._disconnect_async(name))

    async def _disconnect_async(self, name: str) -> bool:
        """Async implementation of session and process termination."""
        server = self.servers.get(name)
        if not server:
            return False

        if server._exit_stack:
            try:
                await server._exit_stack.aclose()
            except Exception as e:
                logger.warning("Error closing MCP server '%s' exit stack: %s", name, e)

        server.status = "disconnected"
        server._session = None
        server._exit_stack = None
        server.tools.clear()
        return True

    def disconnect_all(self) -> None:
        """Disconnects all active MCP servers."""
        if self._closed:
            return
        for name in list(self.servers.keys()):
            try:
                self.disconnect(name)
            except Exception:
                pass

    def list_servers(self) -> List[Dict[str, Any]]:
        """Returns status and metadata for all configured MCP servers."""
        res = []
        for name, srv in self.servers.items():
            res.append({
                "name": name,
                "transport": srv.config.transport,
                "target": srv.config.url if srv.config.transport == "sse" else f"{srv.config.command} {' '.join(srv.config.args)}".strip(),
                "status": srv.status,
                "error": srv.error,
                "tool_count": len(srv.tools),
                "tools": [t.name for t in srv.tools],
            })
        return res

    def get_server_tools(self, server_name: Optional[str] = None) -> List[ToolDefinition]:
        """Returns ToolDefinitions from a specified server or all connected servers."""
        if server_name:
            server = self.servers.get(server_name)
            return list(server.tools) if server else []

        all_tools: List[ToolDefinition] = []
        for server in self.servers.values():
            if server.status == "connected":
                all_tools.extend(server.tools)
        return all_tools

    def load_config_file(self, filepath: str) -> Dict[str, List[ToolDefinition]]:
        """Loads and connects MCP servers specified in a JSON or YAML config file."""
        path = Path(filepath).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MCP configuration file not found at: {filepath}")

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in (".yaml", ".yml"):
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)

        return self.load_config_dict(data)

    def load_config_dict(self, config_dict: Dict[str, Any]) -> Dict[str, List[ToolDefinition]]:
        """Parses standard MCP server dictionary and connects all declared servers."""
        # Support both 'mcpServers' (standard Claude/Cursor/Antigravity) and 'mcp_servers'
        servers_spec = config_dict.get("mcpServers") or config_dict.get("mcp_servers") or config_dict

        results: Dict[str, List[ToolDefinition]] = {}
        for srv_name, srv_body in servers_spec.items():
            if not isinstance(srv_body, dict):
                continue

            transport = srv_body.get("transport", "stdio")
            url = srv_body.get("url")
            command = srv_body.get("command")
            args = srv_body.get("args", [])
            env = srv_body.get("env")
            cwd = srv_body.get("cwd")
            headers = srv_body.get("headers")

            if url and (transport == "sse" or not command):
                cfg = MCPServerConfig(
                    name=srv_name,
                    transport="sse",
                    url=url,
                    headers=headers,
                )
            else:
                cfg = MCPServerConfig(
                    name=srv_name,
                    transport="stdio",
                    command=command,
                    args=args,
                    env=env,
                    cwd=cwd,
                )

            try:
                tools = self.connect_server(cfg)
                results[srv_name] = tools
            except Exception as e:
                logger.error("Failed to connect configured server '%s': %s", srv_name, e)
                results[srv_name] = []

        return results

    def close(self) -> None:
        """Closes all MCP sessions and shuts down the background event loop."""
        if self._closed:
            return
        self._closed = True
        try:
            self.disconnect_all()
        finally:
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread.is_alive():
                self._thread.join(timeout=2.0)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
