"""GCP Gemini / Vertex AI Provider Adapter for harnessfun."""

from typing import Any, Callable, Dict, List
from google import genai
from google.genai import types

from harnessfun.models import Message, ProviderResponse, ToolCall
from harnessfun.providers.base import BaseLLMProvider


class GCPGeminiProvider(BaseLLMProvider):
    """Adapter for Google Gemini models via GCP Vertex AI & Application Default Credentials."""

    def __init__(self, project_id: str, location: str = "global"):
        self.project_id = project_id
        self.location = location
        # Initialize Google GenAI client in Vertex AI mode using ADC
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location
        )

    def list_models(self) -> List[str]:
        """Queries GCP Vertex AI API and returns list of available Gemini model IDs."""
        try:
            models_pager = self.client.models.list()
            model_ids = []
            for m in models_pager:
                name = m.name or ""
                # Strip publisher prefix if present
                clean_name = name.split("/")[-1]
                if "gemini" in clean_name.lower():
                    model_ids.append(clean_name)
            return sorted(list(set(model_ids)))
        except Exception as e:
            # Fallback default model list if query fails or offline
            return [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash"
            ]

    def generate(
        self,
        messages: List[Message],
        tools: List[Callable],
        model: str,
        system_instruction: str
    ) -> ProviderResponse:
        """Executes content generation via Vertex AI SDK and maps responses to ProviderResponse."""
        contents: List[types.Content] = []

        # 1. Convert normalized Message list into Gemini types.Content objects
        for msg in messages:
            if msg.role == "user" and msg.content:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg.content)]
                    )
                )
            elif msg.role == "assistant":
                parts = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                for tc in msg.tool_calls:
                    parts.append(
                        types.Part.from_function_call(
                            name=tc.name,
                            args=tc.args
                        )
                    )
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            elif msg.role == "tool":
                parts = []
                for tr in msg.tool_results:
                    parts.append(
                        types.Part.from_function_response(
                            name=tr.name,
                            response=tr.output
                        )
                    )
                if parts:
                    contents.append(types.Content(role="user", parts=parts))

        # 2. Convert Python callables into FunctionDeclarations to avoid SDK AFC interference
        gemini_tools = None
        if tools:
            func_decls = []
            for t in tools:
                if callable(t):
                    func_decls.append(
                        types.FunctionDeclaration.from_callable(callable=t, client=self.client)
                    )
                elif isinstance(t, types.FunctionDeclaration):
                    func_decls.append(t)
            if func_decls:
                gemini_tools = [types.Tool(function_declarations=func_decls)]

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=gemini_tools,
            temperature=0.0
        )

        # 3. Call Vertex AI API
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        # 4. Extract function calls if any
        tool_calls: List[ToolCall] = []
        if response.function_calls:
            for idx, call in enumerate(response.function_calls):
                args_dict = dict(call.args) if call.args else {}
                tool_calls.append(
                    ToolCall(
                        id=f"call_{idx}_{call.name}",
                        name=call.name,
                        args=args_dict
                    )
                )

        text_content = None
        if not tool_calls:
            try:
                text_content = response.text
            except Exception:
                text_content = None

        return ProviderResponse(
            text=text_content,
            tool_calls=tool_calls
        )
