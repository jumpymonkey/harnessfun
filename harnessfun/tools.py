"""Decorator-based Tool Registry for harnessfun."""

import datetime
import inspect
import json
from typing import Any, Callable, Dict, List, Optional


from harnessfun.models import ToolDefinition


def _callable_to_json_schema(func: Callable) -> Dict[str, Any]:
    """Generates a JSON Schema object for a Python function's arguments."""
    sig = inspect.signature(func)
    properties: Dict[str, Any] = {}
    required: List[str] = []

    type_map = {
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        str: "string",
    }

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        p_type = type_map.get(param.annotation, "string")
        properties[param_name] = {
            "type": p_type,
            "description": f"Parameter {param_name}"
        }
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


class ToolRegistry:
    """Manages Python function tools, MCP tools, and handles execution."""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: Dict[str, ToolDefinition] = {}

    def register(self, func: Optional[Callable] = None, *, name: Optional[str] = None):
        """Decorator to register a Python function as a tool."""
        def decorator(f: Callable):
            fn_name = name or f.__name__
            self.tools[fn_name] = f
            self.tool_definitions[fn_name] = ToolDefinition(
                name=fn_name,
                description=(f.__doc__ or f"Tool function {fn_name}").strip(),
                parameters=_callable_to_json_schema(f),
                handler=f,
                server_name="local"
            )
            return f

        if func is None:
            return decorator
        return decorator(func)

    def register_tool_definition(self, tool_def: ToolDefinition) -> None:
        """Registers a ToolDefinition (e.g. discovered from an MCP server)."""
        self.tool_definitions[tool_def.name] = tool_def
        self.tools[tool_def.name] = tool_def.handler or tool_def.execute

    def unregister(self, name: str) -> bool:
        """Removes a tool by name."""
        removed = False
        if name in self.tool_definitions:
            del self.tool_definitions[name]
            removed = True
        if name in self.tools:
            del self.tools[name]
            removed = True
        return removed

    def unregister_server(self, server_name: str) -> int:
        """Unregisters all tools associated with a specific MCP server name."""
        to_remove = [
            t_name for t_name, t_def in self.tool_definitions.items()
            if t_def.server_name == server_name or t_name.startswith(f"{server_name}__")
        ]
        for t_name in to_remove:
            self.unregister(t_name)
        return len(to_remove)

    def get_tools(self) -> List[ToolDefinition]:
        """Returns all registered tool definitions."""
        return list(self.tool_definitions.values())

    def get_functions_list(self) -> List[Callable]:
        """Returns the list of raw Python functions or execution handlers registered as tools."""
        return list(self.tools.values())

    def _resolve_tool_name(self, name: str) -> Optional[str]:
        """Resolves tool name across exact matches and namespace prefixes."""
        if name in self.tools:
            return name

        # Handle colon prefix (e.g., default_api:get_weather or sqlite:query)
        if ":" in name:
            candidate = name.replace(":", "__")
            if candidate in self.tools:
                return candidate
            stripped = name.split(":", 1)[-1]
            if stripped in self.tools:
                return stripped

        # Handle double underscore prefix (e.g., sqlite__query -> query fallback)
        if "__" in name:
            stripped = name.split("__", 1)[-1]
            if stripped in self.tools:
                return stripped

        # Reverse check: if called with stripped name but only one namespaced tool exists
        matching = [k for k in self.tools if k.endswith(f"__{name}") or k.endswith(f":{name}")]
        if len(matching) == 1:
            return matching[0]

        return None

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a registered tool safely and returns a dictionary output."""
        resolved = self._resolve_tool_name(name)
        if not resolved:
            return {"error": f"Tool '{name}' not found in registry."}

        try:
            func = self.tools[resolved]
            result = func(**args)
            if isinstance(result, dict):
                return result
            return {"result": result}
        except Exception as e:
            return {"error": f"Tool Execution Error ({name}): {str(e)}"}

    def get_info(self) -> List[Dict[str, Any]]:
        """Returns metadata for all registered tools."""
        info = []
        for name, t_def in self.tool_definitions.items():
            info.append({
                "name": name,
                "description": t_def.description,
                "parameters": t_def.parameters,
                "server": t_def.server_name or "local",
            })
        return info


# --- Default Built-in Tools ---
default_registry = ToolRegistry()


@default_registry.register
def get_weather(location: str) -> dict:
    """Fetches the current weather report for a given location."""
    return {
        "location": location,
        "temperature": "22°C",
        "condition": "Sunny",
        "humidity": "45%"
    }


@default_registry.register
def calculate(expression: str) -> float:
    """Evaluates a mathematical expression."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression).issubset(allowed_chars):
        raise ValueError("Invalid characters in mathematical expression.")
    return float(eval(expression, {"__builtins__": None}, {}))


@default_registry.register
def get_current_time(timezone: str = "UTC") -> str:
    """Gets the current date and time formatted in UTC."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")
