import json
from typing import Any

from azure.ai.evaluation import AzureOpenAIModelConfiguration, SimilarityEvaluator
from azure.identity import DefaultAzureCredential
from pydantic import BaseModel

from azure_ai_eval.model_config import model_config


class Result(BaseModel):
    similarity: float
    gpt_similarity: float
    similarity_result: str
    similarity_threshold: int
    similarity_prompt_tokens: int
    similarity_completion_tokens: int
    similarity_total_tokens: int
    similarity_finish_reason: str
    similarity_model: str
    similarity_sample_input: str
    similarity_sample_output: str


def evaluate(
    model_config: AzureOpenAIModelConfiguration,
    ground_truth: list[dict],
    predicted: list[str],
) -> list[Result]:
    eval = SimilarityEvaluator(
        model_config, credential=DefaultAzureCredential(), threshold=4
    )

    results: list[dict[str, Any]] = [
        eval(
            query=gt["query"],
            response=predicted[i],
            ground_truth=gt["expected_response"],
        )
        for i, gt in enumerate(ground_truth)
    ]

    return [Result(**r) for r in results]


if __name__ == "__main__":
    gt = json.load(open("ground-truth.json", "r"))
    predicted = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "It is Beijing",
    ]

    result = evaluate(model_config, gt, predicted)
    print(json.dumps([r.model_dump() for r in result], indent=2))
