"""Custom Azure AI evaluator that checks the dataset's `status` field equals 0."""

from typing import Any


class StatusEvaluator:
    """Evaluator that checks whether the row's `status` field equals 0.

    This evaluator does not require an LLM. It reads the `status` value from the
    dataset row (mapped via `evaluator_config` column mapping) and reports a
    pass/fail outcome.

    Output schema follows the Azure AI Foundry evaluation convention so the UI
    renders it correctly:
        - `status`: numeric score (1.0 pass, 0.0 fail)
        - `status_result`: "pass" | "fail"
        - `status_threshold`: pass threshold (1.0)
        - `status_reason`: human-readable explanation
        - `status_value`: the raw status value from the row
    """

    id = "status_evaluator"
    THRESHOLD = 1.0

    def _result(
        self, passed: bool, reason: str, status_value: Any = None
    ) -> dict[str, Any]:
        score = 1.0 if passed else 0.0
        return {
            "status": score,
            "status_result": "pass" if passed else "fail",
            "status_threshold": self.THRESHOLD,
            "status_reason": reason,
            "status_value": status_value,
        }

    def __call__(self, *, status: Any = None, **kwargs: Any) -> dict[str, Any]:
        if status is None:
            return self._result(False, "Missing 'status' field.")

        passed = status == 0
        reason = "status is 0" if passed else f"Expected status 0, got {status!r}"
        return self._result(passed, reason, status)
