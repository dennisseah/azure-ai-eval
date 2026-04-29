from azure_ai_eval.status_evaluator import StatusEvaluator


def test_status_zero_passes():
    evaluator = StatusEvaluator()
    result = evaluator(status=0)
    assert result["status_value"] == 0
    assert result["status_result"] == "pass"
    assert result["status"] == 1.0
    assert result["status_threshold"] == 1.0


def test_status_nonzero_fails():
    evaluator = StatusEvaluator()
    result = evaluator(status=1)
    assert result["status_value"] == 1
    assert result["status_result"] == "fail"
    assert result["status"] == 0.0


def test_missing_status_fails():
    evaluator = StatusEvaluator()
    result = evaluator()
    assert result["status_result"] == "fail"
    assert result["status"] == 0.0
    assert "Missing 'status' field" in result["status_reason"]


def test_extra_kwargs_are_ignored():
    evaluator = StatusEvaluator()
    result = evaluator(status=0, query="q", response="r", ground_truth="g")
    assert result["status_result"] == "pass"
    assert result["status"] == 1.0
