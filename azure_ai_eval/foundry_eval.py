"""Run evaluation and upload results to Azure AI Foundry project."""

import json
import os
import tempfile
from pathlib import Path

from azure.ai.evaluation import SimilarityEvaluator, evaluate
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from azure_ai_eval.model_config import model_config
from azure_ai_eval.status_evaluator import StatusEvaluator

load_dotenv()
AZURE_AI_PROJECT_ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]


def build_dataset(
    ground_truth: list[dict],
    predicted: list[str],
    predicted_status: list[int],
    path: str | os.PathLike,
) -> str:
    # Build a JSONL dataset from ground truth and predicted responses
    # Each row in the JSONL file will contain the original query, the model's predicted
    # response,
    # and the expected ground truth response for evaluation purposes
    out = Path(path)
    data = [
        {
            "query": gt["query"],
            "response": predicted[i],
            "status": predicted_status[i],
            "ground_truth": gt["expected_response"],
        }
        for i, gt in enumerate(ground_truth)
    ]

    with out.open("w") as f:
        f.write("\n".join(json.dumps(row) for row in data))

    return str(out)


if __name__ == "__main__":
    gt = json.load(open("ground-truth.json", "r"))
    predicted = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "It is Beijing",
    ]
    predicted_status = [0, 0, 1]  # 0 means correct, 1 means incorrect

    similarity = SimilarityEvaluator(
        model_config, credential=DefaultAzureCredential(), threshold=4
    )
    status_eval = StatusEvaluator()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        data_path = build_dataset(
            gt, predicted, predicted_status, tmp_dir / "eval-data.jsonl"
        )

        result = evaluate(
            evaluation_name="nathan-eval",
            data=data_path,
            evaluators={"similarity": similarity, "status": status_eval},
            evaluator_config={
                "status": {"column_mapping": {"status": "${data.status}"}},
            },
            azure_ai_project=AZURE_AI_PROJECT_ENDPOINT,
            output_path=str("eval-results.json"),
        )

    print("Studio URL:", result.get("studio_url"))
    print(json.dumps(result["metrics"], indent=2))
