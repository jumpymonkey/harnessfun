"""GCP Authentication & Configuration Loader for harnessfun."""

import os
import yaml
from pathlib import Path
from typing import Any, Optional, Tuple
from harnessfun.models import SessionConfig


class SecurityValidationError(Exception):
    """Raised when insecure authentication (e.g. API keys) is detected."""
    pass


class AuthError(Exception):
    """Raised when GCP Application Default Credentials are invalid or missing."""
    pass


def validate_no_api_keys() -> None:
    """Enforces prohibition of static API keys."""
    if "GEMINI_API_KEY" in os.environ:
        raise SecurityValidationError(
            "Security Policy Error: GEMINI_API_KEY detected in environment variables. "
            "harnessfun strictly prohibits API key usage and requires Google Cloud "
            "Application Default Credentials (ADC) or Service Account authentication."
        )


def verify_gcp_adc() -> Tuple[Any, str]:
    """Verifies that Google Cloud ADC is valid and returns (credentials, project_id)."""
    validate_no_api_keys()
    
    try:
        import google.auth
        credentials, project_id = google.auth.default()
        
        # Check if project ID is available from env if not inferred
        env_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        final_project = env_project or project_id
        
        if not final_project:
            raise AuthError(
                "GCP Project ID could not be determined. Please set the GOOGLE_CLOUD_PROJECT "
                "environment variable or configure your active gcloud project via:\n"
                "  gcloud config set project YOUR_PROJECT_ID"
            )
            
        return credentials, final_project
    except Exception as e:
        if isinstance(e, (AuthError, SecurityValidationError)):
            raise e
        raise AuthError(
            f"Failed to authenticate with Google Cloud ADC: {e}\n"
            "Please run 'gcloud auth application-default login' to authenticate."
        )


def load_config(
    config_path: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    model_location: Optional[str] = None,
    model: Optional[str] = None
) -> SessionConfig:
    """Loads session configuration from YAML config file, CLI args, and env vars."""
    validate_no_api_keys()

    cfg_dict = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}

    gcp_cfg = cfg_dict.get("gcp", {})
    harness_cfg = cfg_dict.get("harness", {})

    # Determine Project ID
    final_project = (
        project_id or 
        os.environ.get("GOOGLE_CLOUD_PROJECT") or 
        gcp_cfg.get("project_id")
    )

    # Validate ADC & Project ID
    if not final_project:
        _, final_project = verify_gcp_adc()
    else:
        verify_gcp_adc()

    final_location = (
        location or 
        os.environ.get("GOOGLE_CLOUD_LOCATION") or 
        gcp_cfg.get("location", "us-central1")
    )

    final_model_location = (
        model_location or
        os.environ.get("GEMINI_LOCATION") or
        os.environ.get("GOOGLE_CLOUD_MODEL_LOCATION") or
        harness_cfg.get("model_location") or
        gcp_cfg.get("model_location", "global")
    )

    final_model = (
        model or 
        os.environ.get("HARNESS_MODEL") or 
        harness_cfg.get("default_model", "gemini-2.5-flash")
    )

    return SessionConfig(
        project_id=final_project,
        location=final_location,
        model_location=final_model_location,
        active_model=final_model,
        system_instruction=harness_cfg.get(
            "system_instruction", 
            "You are a helpful, general-purpose AI assistant. Answer general knowledge questions using your built-in knowledge. When a prompt requires tools (such as weather, time, or math calculations), invoke the appropriate tool."
        ),
        max_steps=harness_cfg.get("max_steps", 10)
    )
