"""
CLI evaluation script — used by GitHub Actions and local runs.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --project my-agent --threshold 0.85
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmwatch.evaluator import run_ragas_eval, format_pr_comment


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for LLMWatch")
    parser.add_argument("--dataset", default="eval/dataset.json")
    parser.add_argument("--project", default=os.getenv("LLMWATCH_PROJECT", "default"))
    parser.add_argument("--db", default=os.getenv("LLMWATCH_DB", "llmwatch.db"))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("EVAL_SCORE_THRESHOLD", "0.80")))
    parser.add_argument("--output-json", default=None, help="Write scores to this JSON file")
    parser.add_argument("--output-comment", default=None, help="Write PR comment markdown to this file")
    args = parser.parse_args()

    print(f"Running RAGAS evaluation on: {args.dataset}")
    print(f"Project: {args.project} | Threshold: {args.threshold}")
    print("-" * 50)

    scores = run_ragas_eval(
        dataset_path=args.dataset,
        project=args.project,
        db_path=args.db,
        threshold=args.threshold,
    )

    print(f"Faithfulness:      {scores['faithfulness']:.4f}")
    print(f"Answer Relevancy:  {scores['answer_relevancy']:.4f}")
    print(f"Context Precision: {scores['context_precision']:.4f}")
    print(f"Overall Score:     {scores['overall']:.4f}")
    print(f"Threshold:         {scores['threshold']}")
    print(f"Result:            {'PASSED ✅' if scores['passed'] else 'FAILED ❌'}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"\nScores saved to: {args.output_json}")

    comment = format_pr_comment(scores)
    if args.output_comment:
        with open(args.output_comment, "w") as f:
            f.write(comment)
        print(f"PR comment saved to: {args.output_comment}")
    else:
        print("\n" + comment)

    sys.exit(0 if scores["passed"] else 1)


if __name__ == "__main__":
    main()
