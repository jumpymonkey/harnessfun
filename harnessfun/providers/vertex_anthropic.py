"""Anthropic Claude / Google Cloud Vertex AI Provider Adapter for harnessfun."""

import inspect
import json
from typing import Any, Callable, Dict, List, Optional
import google.auth
from google.auth.transport.requests import Request
import httpx

from harnessfun.config import verify_gcp_adc
from harnessfun.models import Message, ProviderResponse, ToolCall
from harnessfun.providers.base import BaseLLMProvider

# Try to import AnthropicVertex if the official SDK is installed
try:
    from anthropic import AnthropicVertex
    HAS_ANTHROPIC_SDK = True
except ImportError:
    HAS_ANTHROPIC_SDK = False


# Known model aliases to full Vertex AI Model Garden tags
MODEL_ALIASES: Dict[str, str] = {
    "claude-3-7-sonnet": "claude-3-7-sonnet@20250219",
    "claude-3-5-sonnet": "claude-3-5-sonnet-v2@20241022",
    "claude-3-5-sonnet-v2": "claude-3-5-sonnet-v2@20241022",
    "claude-3-5-sonnet-v1": "claude-3-5-sonnet@20240620",
    "claude-3-5-haiku": "claude-3-5-haiku@20241022",
    "claude-3-opus": "claude-3-opus@20240229",
    "claude-3-haiku": "claude-3-haiku@20240307",
}


def normalize_anthropic_model_id(model_id: str) -> str:
    """Normalizes short aliases to canonical Vertex AI Model Garden IDs."""
    clean = model_id.lower().strip()
    return MODEL_ALIASES.get(clean, clean)


class VertexAnthropicProvider(BaseLLMProvider):
    """Adapter for Anthropic Claude models accessed via Google Cloud Vertex AI and ADC."""

    SUPPORTED_MODELS = [
        "claude-3-7-sonnet@20250219",
        "claude-3-5-sonnet-v2@20241022",
        "claude-3-5-sonnet@20240620",
        "claude-3-5-haiku@20241022",
        "claude-3-opus@20240229",
        "claude-3-haiku@20240307",
    ]

    def __init__(
        self,
        project_id: str,
        location: str = "us-east5",
        credentials: Optional[Any] = None,
    ):
        self.project_id = project_id
        # Anthropic on Vertex AI requires regional endpoints (defaulting to us-east5 if global)
        self.location = "us-east5" if location in ("global", "", None) else location
        
        # Initialize Google ADC credentials
        if credentials:
            self.credentials = credentials
        else:
            self.credentials, detected_proj = verify_gcp_adc()
            if not self.project_id:
                self.project_id = detected_proj
        self.auth_request = Request()

        # Initialize official SDK client if available
        self.sdk_client = None
        if HAS_ANTHROPIC_SDK:
            try:
                self.sdk_client = AnthropicVertex(
                    project_id=self.project_id,
                    region=self.location
                )
            except Exception:
                self.sdk_client = None

    def list_models(self) -> List[str]:
        """Returns the list of available Anthropic Claude models on Vertex AI."""
        return list(self.SUPPORTED_MODELS)

    def _normalize_model(self, model_id: str) -> str:
        """Normalizes model ID using MODEL_ALIASES."""
        return normalize_anthropic_model_id(model_id)

    def _get_access_token(self) -> str:
        """Refreshes and returns the valid Google Cloud OAuth bearer token."""
        if not self.credentials.valid or not self.credentials.token:
            self.credentials.refresh(self.auth_request)
        return self.credentials.token

    @staticmethod
    def _python_type_to_json_type(py_type: Any) -> str:
        """Maps Python types to JSON Schema types."""
        if py_type is int:
            return "integer"
        elif py_type is float:
            return "number"
        elif py_type is bool:
            return "boolean"
        elif py_type is list or py_type is List:
            return "array"
        elif py_type is dict or py_type is Dict:
            return "object"
        return "string"

    def _convert_tool_to_anthropic_schema(self, func: Callable) -> Dict[str, Any]:
        """Converts a Python callable into Anthropic's tool JSON schema."""
        sig = inspect.signature(func)
        doc = (func.__doc__ or f"Tool function {func.__name__}").strip()
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_type = self._python_type_to_json_type(param.annotation)
            properties[param_name] = {
                "type": param_type,
                "description": f"Argument {param_name}"
            }
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": func.__name__,
            "description": doc,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Converts normalized Message list into Anthropic's Messages API format."""
        anthropic_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == "user" and msg.content:
                anthropic_messages.append({
                    "role": "user",
                    "content": msg.content
                })
            elif msg.role == "assistant":
                content_blocks = []
                if msg.content:
                    content_blocks.append({
                        "type": "text",
                        "text": msg.content
                    })
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.args
                    })
                if content_blocks:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content_blocks
                    })
            elif msg.role == "tool":
                tool_results_blocks = []
                for tr in msg.tool_results:
                    output_str = json.dumps(tr.output) if isinstance(tr.output, (dict, list)) else str(tr.output)
                    tool_results_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tr.call_id,
                        "content": output_str
                    })
                if tool_results_blocks:
                    anthropic_messages.append({
                        "role": "user",
                        "content": tool_results_blocks
                    })

        # Anthropic requires strict alternation between user and assistant roles.
        # Merge adjacent turns with identical roles if any exist.
        merged_messages: List[Dict[str, Any]] = []
        for msg in anthropic_messages:
            if merged_messages and merged_messages[-1]["role"] == msg["role"]:
                last_content = merged_messages[-1]["content"]
                new_content = msg["content"]

                if isinstance(last_content, str) and isinstance(new_content, str):
                    merged_messages[-1]["content"] = f"{last_content}\n\n{new_content}"
                else:
                    if isinstance(last_content, str):
                        last_content = [{"type": "text", "text": last_content}]
                    if isinstance(new_content, str):
                        new_content = [{"type": "text", "text": new_content}]
                    merged_messages[-1]["content"] = last_content + new_content
            else:
                merged_messages.append(msg)

        return merged_messages

    def generate(
        self,
        messages: List[Message],
        tools: List[Callable],
        model: str,
        system_instruction: str
    ) -> ProviderResponse:
        """Executes content generation via Vertex AI Anthropic endpoint using ADC."""
        canonical_model = normalize_anthropic_model_id(model)
        anthropic_messages = self._convert_messages(messages)
        anthropic_tools = [self._convert_tool_to_anthropic_schema(t) for t in tools] if tools else None

        # 1. Use official SDK if available
        if self.sdk_client:
            try:
                kwargs = {
                    "model": canonical_model,
                    "max_tokens": 4096,
                    "messages": anthropic_messages,
                    "temperature": 0.0
                }
                if system_instruction:
                    kwargs["system"] = system_instruction
                if anthropic_tools:
                    kwargs["tools"] = anthropic_tools

                response = self.sdk_client.messages.create(**kwargs)
                return self._parse_sdk_response(response)
            except Exception as e:
                # If SDK fails due to access, raise clear message; otherwise fallback to direct REST
                if "does not have access" in str(e) or "was not found" in str(e):
                    raise e

        # 2. Direct Vertex AI REST API via google-auth + httpx
        return self._generate_via_rest(
            canonical_model=canonical_model,
            anthropic_messages=anthropic_messages,
            anthropic_tools=anthropic_tools,
            system_instruction=system_instruction
        )

    def _generate_via_rest(
        self,
        canonical_model: str,
        anthropic_messages: List[Dict[str, Any]],
        anthropic_tools: Optional[List[Dict[str, Any]]],
        system_instruction: str
    ) -> ProviderResponse:
        """Calls the Vertex AI rawPredict endpoint directly."""
        token = self._get_access_token()
        url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/"
            f"locations/{self.location}/publishers/anthropic/models/{canonical_model}:rawPredict"
        )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }

        payload: Dict[str, Any] = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": anthropic_messages,
            "max_tokens": 4096,
            "temperature": 0.0
        }
        if system_instruction:
            payload["system"] = system_instruction
        if anthropic_tools:
            payload["tools"] = anthropic_tools

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)

        if resp.status_code != 200:
            error_msg = resp.text
            try:
                data = resp.json()
                if isinstance(data, dict):
                    err = data.get("error")
                    if isinstance(err, dict):
                        error_msg = err.get("message", resp.text)
                    elif isinstance(err, str):
                        error_msg = err
            except Exception:
                pass

            if resp.status_code in (403, 404):
                console_url = f"https://console.cloud.google.com/vertex-ai/model-garden"
                raise RuntimeError(
                    f"Anthropic model access required on Vertex AI.\n"
                    f"Model '{canonical_model}' is not yet enabled or accessible in project '{self.project_id}' ({self.location}).\n"
                    f"Details: {error_msg}\n\n"
                    f"To enable Claude on Vertex AI, visit Model Garden in the Google Cloud Console:\n  {console_url}\n"
                    f"(Supported regions include 'us-east5' and 'europe-west1'. Use --model-location to change region)."
                )
            raise RuntimeError(f"Vertex AI Anthropic API Error ({resp.status_code}): {error_msg}")

        data = resp.json()
        return self._parse_json_response(data)

    def _parse_sdk_response(self, response: Any) -> ProviderResponse:
        """Parses response from Anthropic SDK object."""
        tool_calls: List[ToolCall] = []
        text_parts: List[str] = []

        for block in getattr(response, "content", []):
            b_type = getattr(block, "type", "")
            if b_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif b_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        args=getattr(block, "input", {})
                    )
                )

        return ProviderResponse(
            text="\n".join(text_parts).strip() if text_parts else None,
            tool_calls=tool_calls
        )

    def _parse_json_response(self, data: Dict[str, Any]) -> ProviderResponse:
        """Parses response from raw Vertex AI REST JSON."""
        tool_calls: List[ToolCall] = []
        text_parts: List[str] = []

        for item in data.get("content", []):
            item_type = item.get("type")
            if item_type == "text":
                text_parts.append(item.get("text", ""))
            elif item_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=item.get("id", ""),
                        name=item.get("name", ""),
                        args=item.get("input", {})
                    )
                )

        return ProviderResponse(
            text="\n".join(text_parts).strip() if text_parts else None,
            tool_calls=tool_calls
        )
