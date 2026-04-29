import json
from typing import Any

from azure.ai.evaluation import AzureOpenAIModelConfiguration, SimilarityEvaluator
from azure.identity import DefaultAzureCredential

from azure_ai_eval.model_config import model_config


def evaluate(
    model_config: AzureOpenAIModelConfiguration,
    ground_truth: list[dict],
    predicted: list[str],
) -> list[dict[str, Any]]:
    eval = SimilarityEvaluator(
        model_config, credential=DefaultAzureCredential(), threshold=4
    )

    return [
        eval(
            query=gt["query"],
            response=predicted[i],
            ground_truth=gt["expected_response"],
        )
        for i, gt in enumerate(ground_truth)
    ]


if __name__ == "__main__":
    gt = json.load(open("ground-truth.json", "r"))
    predicted = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "It is Beijing",
    ]

    results = evaluate(model_config, gt, predicted)
    print(json.dumps(results, indent=2))
