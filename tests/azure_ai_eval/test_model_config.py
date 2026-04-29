import importlib

import pytest


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYED_MODEL_NAME", "gpt-4.1")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    # Prevent a real .env from leaking values into the test.
    monkeypatch.chdir(tmp_path)


def test_model_config(mock_env):
    from azure_ai_eval import model_config as module

    importlib.reload(module)

    assert module.model_config["azure_endpoint"] == "https://example.openai.azure.com/"
    assert module.model_config["azure_deployment"] == "gpt-4.1"
    assert module.model_config["api_version"] == "2024-10-21"  # type: ignore
    assert "api_key" not in module.model_config
