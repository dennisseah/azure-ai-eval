from unittest.mock import MagicMock, patch

from azure_ai_eval.similarity import evaluate


@patch("azure_ai_eval.similarity.DefaultAzureCredential")
@patch("azure_ai_eval.similarity.SimilarityEvaluator")
def test_evaluate(mock_evaluator_cls, mock_credential):
    model_config = {
        "azure_endpoint": "https://example.openai.azure.com/",
        "azure_deployment": "gpt-4.1",
        "api_version": "2024-10-21",
    }
    ground_truth = [
        {"query": "What is the capital of France?", "expected_response": "Paris"},
        {"query": "What is the capital of Germany?", "expected_response": "Berlin"},
        {"query": "What is the capital of China?", "expected_response": "Beijing"},
    ]
    predicted = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "It is Beijing",
    ]

    mock_eval = MagicMock(
        side_effect=lambda query, response, ground_truth: {
            "similarity": 5.0,
            "similarity_result": "pass",
            "query": query,
            "response": response,
            "ground_truth": ground_truth,
        }
    )
    mock_evaluator_cls.return_value = mock_eval

    results = evaluate(model_config, ground_truth, predicted)  # type: ignore

    mock_evaluator_cls.assert_called_once_with(
        model_config, credential=mock_credential.return_value, threshold=4
    )
    assert mock_eval.call_count == len(ground_truth)
    assert isinstance(results, list)
    assert len(results) == len(ground_truth)
    for i, r in enumerate(results):
        assert r["query"] == ground_truth[i]["query"]
        assert r["response"] == predicted[i]
        assert r["ground_truth"] == ground_truth[i]["expected_response"]
