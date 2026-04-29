"""Pytest fixtures and shared setup for tests."""

import os

# Set fake Azure env vars before any test module is imported, so modules that
# read settings at import time (e.g. azure_ai_eval.model_config) succeed.
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_DEPLOYED_MODEL_NAME", "gpt-4.1")
os.environ.setdefault("AZURE_OPENAI_API_VERSION", "2024-10-21")
os.environ.setdefault(
    "AZURE_AI_PROJECT_ENDPOINT",
    "https://example.foundry.azure.com/api/projects/p",
)
