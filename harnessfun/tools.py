"""Decorator-based Tool Registry for harnessfun."""

import datetime
import inspect
import json
from typing import Any, Callable, Dict, List, Optional


class ToolRegistry:
    """Manages Python function tools and handles execution."""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, func: Optional[Callable] = None, *, name: Optional[str] = None):
        """Decorator to register a Python function as a tool."""
        def decorator(f: Callable):
            fn_name = name or f.__name__
            self.tools[fn_name] = f
            return f

        if func is None:
            return decorator
        return decorator(func)

    def get_functions_list(self) -> List[Callable]:
        """Returns the list of raw Python functions registered as tools."""
        return list(self.tools.values())

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a registered tool safely and returns a dictionary output."""
        actual_name = name
        if actual_name not in self.tools and ":" in actual_name:
            stripped = actual_name.split(":", 1)[-1]
            if stripped in self.tools:
                actual_name = stripped

        if actual_name not in self.tools:
            return {"error": f"Tool '{name}' not found in registry."}

        try:
            func = self.tools[actual_name]
            result = func(**args)
            if isinstance(result, dict):
                return result
            return {"result": result}
        except Exception as e:
            return {"error": f"Tool Execution Error ({name}): {str(e)}"}

    def get_info(self) -> List[Dict[str, Any]]:
        """Returns metadata for all registered tools."""
        info = []
        for name, func in self.tools.items():
            sig = inspect.signature(func)
            doc = (func.__doc__ or "No description provided.").strip()
            params = {
                p: str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
                for p, param in sig.parameters.items()
            }
            info.append({
                "name": name,
                "description": doc,
                "parameters": params
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
