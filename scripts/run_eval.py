from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from eval.evaluator import Evaluator


def main() -> None:
    paper_path = ROOT / "data" / "sample_paper.txt"
    evaluator = Evaluator(settings.benchmark_path)
    results = evaluator.run(paper_path)

    print("=== Evaluation Results ===")
    total_score = 0.0
    for item in results:
        total_score += item.score
        print(f"Query: {item.query}")
        print(f"Score: {item.score:.2f} ({item.hit_count}/{item.total_required})")
        print("-" * 50)

    avg = total_score / len(results) if results else 0.0
    print(f"Average score: {avg:.2f}")


if __name__ == "__main__":
    main()
