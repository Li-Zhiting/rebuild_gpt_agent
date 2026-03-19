from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.app import ResearchPaperAgent
from tools.parser import load_text_file


@dataclass(slots=True)
class EvalResult:
    query: str
    hit_count: int
    total_required: int
    score: float


class Evaluator:
    def __init__(self, benchmark_path: str | Path) -> None:
        self.benchmark_path = Path(benchmark_path)

    def load_benchmark(self) -> list[dict[str, Any]]:
        return json.loads(self.benchmark_path.read_text(encoding="utf-8"))

    def run(self, paper_path: str | Path) -> list[EvalResult]:
        text = load_text_file(paper_path)
        agent = ResearchPaperAgent()
        agent.load_document(text)
        data = self.load_benchmark()
        results: list[EvalResult] = []
        has_paper_b = bool(agent.doc_memory.source_texts.get("paper_b", "").strip())

        for row in data:
            requires_two_papers = bool(row.get("requires_two_papers", False))
            if requires_two_papers and not has_paper_b:
                print(f"[DEBUG] skip case (requires two papers): {row['query']}")
                continue

            output = agent.ask(row["query"])
            print(f"[DEBUG] answer: {output.answer}")
            answer_lower = output.answer.lower()
            hits = sum(1 for kw in row["must_include"] if kw.lower() in answer_lower)
            total = len(row["must_include"])
            results.append(
                EvalResult(
                    query=row["query"],
                    hit_count=hits,
                    total_required=total,
                    score=hits / total if total else 0.0,
                )
            )
        return results
