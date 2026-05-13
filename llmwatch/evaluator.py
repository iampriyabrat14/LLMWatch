import json
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from .db import MetricsDB


def run_ragas_eval(
    dataset_path: str = "eval/dataset.json",
    project: str = "default",
    db_path: str = "llmwatch.db",
    threshold: float = 0.80,
) -> Dict[str, Any]:
    """Run RAGAS evaluation and persist aggregate score to the DB."""
    with open(dataset_path) as f:
        raw = json.load(f)

    hf_dataset = Dataset.from_list(raw)

    result = evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    scores = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
    }
    overall = round(sum(scores.values()) / len(scores), 4)
    scores["overall"] = overall
    scores["passed"] = overall >= threshold
    scores["threshold"] = threshold

    db = MetricsDB(db_path)
    db.insert_metric(
        project=project,
        model="ragas-eval",
        latency_ms=0.0,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        ragas_score=overall,
    )

    return scores


def format_pr_comment(scores: Dict[str, Any]) -> str:
    """Format RAGAS scores as a GitHub PR comment."""
    status = lambda s: "✅" if s >= scores["threshold"] else "❌"
    lines = [
        "## LLMWatch Evaluation Results",
        "",
        f"| Metric | Score | Status |",
        f"|--------|-------|--------|",
        f"| Faithfulness      | {scores['faithfulness']:.2f} | {status(scores['faithfulness'])} |",
        f"| Answer Relevancy  | {scores['answer_relevancy']:.2f} | {status(scores['answer_relevancy'])} |",
        f"| Context Precision | {scores['context_precision']:.2f} | {status(scores['context_precision'])} |",
        f"| **Overall**       | **{scores['overall']:.2f}** | {status(scores['overall'])} |",
        "",
        f"Threshold: `{scores['threshold']}`  |  Result: {'**PASSED** ✅' if scores['passed'] else '**FAILED** ❌'}",
    ]
    return "\n".join(lines)
