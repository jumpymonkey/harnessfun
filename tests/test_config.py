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
