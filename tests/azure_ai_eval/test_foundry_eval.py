import json

import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv(
        "AZURE_AI_PROJECT_ENDPOINT",
        "https://example.foundry.azure.com/api/projects/p",
    )
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYED_MODEL_NAME", "gpt-4.1")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def test_build_dataset(tmp_path):
    import importlib

    from azure_ai_eval import foundry_eval

    importlib.reload(foundry_eval)

    ground_truth = [
        {"query": "What is the capital of France?", "expected_response": "Paris"},
        {"query": "What is the capital of Germany?", "expected_response": "Berlin"},
    ]
    predicted = ["Paris.", "Berlin."]

    out = tmp_path / "data.jsonl"
    result_path = foundry_eval.build_dataset(ground_truth, predicted, out)

    assert result_path == str(out)
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert rows[0] == {
        "query": "What is the capital of France?",
        "response": "Paris.",
        "ground_truth": "Paris",
    }
    assert rows[1]["response"] == "Berlin."


def test_missing_endpoint_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)

    import importlib

    from azure_ai_eval import foundry_eval

    with pytest.raises(KeyError):
        importlib.reload(foundry_eval)
