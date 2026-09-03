"""Unit tests for harnessfun configuration and ADC security validation."""

import os
import pytest
from harnessfun.config import SecurityValidationError, validate_no_api_keys


def test_api_key_prohibition(monkeypatch):
    """Ensures GEMINI_API_KEY presence raises SecurityValidationError."""
    monkeypatch.setenv("GEMINI_API_KEY", "insecure-test-key")
    with pytest.raises(SecurityValidationError) as exc_info:
        validate_no_api_keys()
    assert "GEMINI_API_KEY detected" in str(exc_info.value)


def test_valid_env_when_no_api_key(monkeypatch):
    """Ensures validate_no_api_keys passes when GEMINI_API_KEY is not set."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Should not raise exception
    validate_no_api_keys()


def test_default_locations(monkeypatch):
    """Ensures load_config defaults GCP location to 'us-central1' and model_location to 'global'."""
    from harnessfun.config import load_config
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GEMINI_LOCATION", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_MODEL_LOCATION", raising=False)
    monkeypatch.setattr("harnessfun.config.verify_gcp_adc", lambda: (None, "test-proj"))
    cfg = load_config(project_id="test-proj")
    assert cfg.location == "us-central1"
    assert cfg.model_location == "global"


def test_custom_model_location(monkeypatch):
    """Ensures load_config respects custom model_location."""
    from harnessfun.config import load_config
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.setattr("harnessfun.config.verify_gcp_adc", lambda: (None, "test-proj"))
    cfg = load_config(project_id="test-proj", model_location="europe-west1")
    assert cfg.location == "us-central1"
    assert cfg.model_location == "europe-west1"


def test_session_config_default_locations():
    """Ensures SessionConfig defaults location to 'us-central1' and model_location to 'global'."""
    from harnessfun.models import SessionConfig
    cfg = SessionConfig(project_id="test-proj")
    assert cfg.location == "us-central1"
    assert cfg.model_location == "global"


def test_provider_default_location():
    """Ensures GCPGeminiProvider defaults location to 'global' when reaching models."""
    import inspect
    from harnessfun.providers.gcp_gemini import GCPGeminiProvider
    sig = inspect.signature(GCPGeminiProvider.__init__)
    assert sig.parameters["location"].default == "global"

