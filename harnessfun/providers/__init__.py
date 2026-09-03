"""LLM Provider Adapters and Factory for harnessfun."""

from typing import Optional
from harnessfun.providers.base import BaseLLMProvider
from harnessfun.providers.gcp_gemini import GCPGeminiProvider
from harnessfun.providers.vertex_anthropic import VertexAnthropicProvider, normalize_anthropic_model_id


def is_anthropic_model(model: str) -> bool:
    """Checks whether a given model string corresponds to an Anthropic model."""
    clean = model.lower().strip()
    return clean.startswith("claude") or "anthropic" in clean


def get_provider(
    model: str,
    project_id: str,
    location: Optional[str] = "global"
) -> BaseLLMProvider:
    """Provider factory returning the appropriate adapter based on the model ID."""
    if is_anthropic_model(model):
        # Anthropic models on Vertex AI require regional endpoints (defaulting to us-east5 if global)
        anthropic_loc = "us-east5" if location in ("global", "", None) else location
        return VertexAnthropicProvider(project_id=project_id, location=anthropic_loc)
    return GCPGeminiProvider(project_id=project_id, location=location or "global")


__all__ = [
    "BaseLLMProvider",
    "GCPGeminiProvider",
    "VertexAnthropicProvider",
    "get_provider",
    "is_anthropic_model",
    "normalize_anthropic_model_id",
]
