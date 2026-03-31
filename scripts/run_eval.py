from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from eval.evaluator import Evaluator


def main() -> None:
    paper_path = ROOT / "data" / "sample_paper2.txt"
    evaluator = Evaluator(settings.benchmark_path)
    results = evaluator.run(paper_path)

    print("=== Evaluation Results ===")
    total_score = 0.0
    for item in results:
        total_score += item.score
        print(f"Case type: {item.case_type}")
        print(f"Query: {item.query}")
        print(f"Keyword score: {item.keyword_score:.2f} ({item.hit_count}/{item.total_required})")
        print(f"LLM score: {item.llm_score:.2f}")
        print(f"Criteria score: {item.criteria_score:.2f}")
        print("Criteria detail:")
        if item.criteria_detail:
            for criterion, passed in item.criteria_detail.items():
                mark = "✓" if passed else "✗"
                print(f"  - {criterion}: {mark}")
        else:
            print("  - (none)")
        print(f"Final score: {item.score:.2f}")
        print(f"LLM reason: {item.llm_reason}")
        print("-" * 50)

    avg = total_score / len(results) if results else 0.0
    print(f"Average score: {avg:.2f}")


if __name__ == "__main__":
    main()
