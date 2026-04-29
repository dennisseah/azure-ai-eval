"""Run evaluation and upload results to Azure AI Foundry project."""

import json
import os
import tempfile
from pathlib import Path

from azure.ai.evaluation import SimilarityEvaluator, evaluate
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from azure_ai_eval.model_config import model_config

load_dotenv()
AZURE_AI_PROJECT_ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]


def build_dataset(
    ground_truth: list[dict], predicted: list[str], path: str | os.PathLike
) -> str:
    out = Path(path)
    data = [
        {
            "query": gt["query"],
            "response": predicted[i],
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

    similarity = SimilarityEvaluator(
        model_config, credential=DefaultAzureCredential(), threshold=4
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        data_path = build_dataset(gt, predicted, tmp_dir / "eval-data.jsonl")

        result = evaluate(
            evaluation_name="similarity-eval",
            data=data_path,
            evaluators={"similarity": similarity},
            azure_ai_project=AZURE_AI_PROJECT_ENDPOINT,
            output_path=str(tmp_dir / "eval-results.json"),
        )

    print("Studio URL:", result.get("studio_url"))
    print(json.dumps(result["metrics"], indent=2))
