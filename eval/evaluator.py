from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.app import ResearchPaperAgent
from tools.parser import load_text_file


@dataclass(slots=True)
class EvalResult:
    case_type: str
    query: str
    hit_count: int
    total_required: int
    keyword_score: float
    llm_score: float
    criteria_score: float
    criteria_detail: dict[str, bool]
    llm_reason: str
    score: float


class Evaluator:
    def __init__(
        self,
        benchmark_path: str | Path,
        model: str = "qwen-plus",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.benchmark_path = Path(benchmark_path)
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url or os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            ),
        )

    def load_benchmark(self) -> list[dict[str, Any]]:
        return json.loads(self.benchmark_path.read_text(encoding="utf-8"))

    def _build_eval_system_prompt(self) -> str:
        return (
            "你是一个回答质量评估器。\n"
            "你的唯一任务是：基于用户 query、reference_answer 和 agent answer，"
            "评估回答是否真正回答了问题、质量如何。\n"
            "你必须重点对照 reference_answer 的核心要点，判断 agent answer 的覆盖度与准确性。\n\n"
            "你必须严格输出 JSON，且只能是如下格式：\n"
            "{\n"
            '  "score": 0.0,\n'
            '  "reason": "一句中文理由"\n'
            "}\n\n"
            "规则：\n"
            "1. score 必须是 0 到 1 之间的浮点数（越高表示越好）。\n"
            "2. reason 必须是一句简短中文理由。\n"
            "3. 仅输出 JSON，禁止输出任何额外文本。"
        )

    def _llm_score(
        self, query: str, answer: str, reference_answer: str = ""
    ) -> tuple[float, str]:
        ref_text = reference_answer.strip() if isinstance(reference_answer, str) else ""
        if not ref_text:
            ref_text = "（未提供）"
        user_prompt = (
            f"query:\n{query}\n\n"
            f"reference_answer:\n{ref_text}\n\n"
            f"answer:\n{answer}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": self._build_eval_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)

            score_raw = data.get("score", 0.0)
            score = float(score_raw)
            score = max(0.0, min(1.0, score))

            reason_raw = data.get("reason", "")
            reason = reason_raw if isinstance(reason_raw, str) else ""
            if not reason.strip():
                reason = "LLM未提供有效理由。"

            return score, reason
        except Exception as exc:
            return 0.0, f"LLM评估失败：{exc}"

    def _criteria_score(
        self, answer: str, criteria_list: list[str] | tuple[str, ...]
    ) -> tuple[float, dict[str, bool]]:
        criteria = [str(item).strip() for item in criteria_list if str(item).strip()]
        if not criteria:
            return 0.0, {}

        system_prompt = (
            "你是一个回答标准检查器。\n"
            "任务：根据给定的 answer，逐条判断 criteria 是否满足。\n"
            "输出必须是严格 JSON 对象，key 必须与输入 criteria 原文完全一致，value 必须是 true 或 false。\n"
            "禁止输出任何额外文本。"
        )
        user_prompt = (
            f"answer:\n{answer}\n\n"
            f"criteria:\n{json.dumps(criteria, ensure_ascii=False)}\n\n"
            "请输出 JSON 对象。"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("criteria 响应不是 JSON 对象")

            detail: dict[str, bool] = {}
            for criterion in criteria:
                value = parsed.get(criterion, False)
                if isinstance(value, bool):
                    detail[criterion] = value
                elif isinstance(value, str):
                    detail[criterion] = value.strip().lower() in {"true", "yes", "1"}
                elif isinstance(value, (int, float)):
                    detail[criterion] = bool(value)
                else:
                    detail[criterion] = False

            true_count = sum(1 for passed in detail.values() if passed)
            score = true_count / len(criteria) if criteria else 0.0
            return score, detail
        except Exception:
            return 0.0, {criterion: False for criterion in criteria}

    def _resolve_weights(self, row: dict[str, Any]) -> tuple[float, float, float]:
        weights = row.get("weights", {})
        try:
            keyword_weight = float(weights.get("keyword", 0.5))
            llm_weight = float(weights.get("llm", 0.5))
            criteria_weight = float(weights.get("criteria", 0.0))
        except Exception:
            return 0.5, 0.5, 0.0

        if keyword_weight < 0 or llm_weight < 0 or criteria_weight < 0:
            return 0.5, 0.5, 0.0

        total = keyword_weight + llm_weight + criteria_weight
        if total <= 0:
            return 0.5, 0.5, 0.0

        return (
            keyword_weight / total,
            llm_weight / total,
            criteria_weight / total,
        )

    def run(self, paper_path: str | Path) -> list[EvalResult]:
        text = load_text_file(paper_path)
        agent = ResearchPaperAgent()
        agent.load_document(text)
        data = self.load_benchmark()
        results: list[EvalResult] = []
        has_paper_b = bool(agent.doc_memory.source_texts.get("paper_b", "").strip())

        for row in data:
            # Only enforce paper-count constraints when the field is explicitly provided.
            if "requires_two_papers" in row:
                requires_two_papers = bool(row.get("requires_two_papers"))
                if requires_two_papers and not has_paper_b:
                    print(f"[DEBUG] skip case (requires two papers): {row['query']}")
                    continue
                if (not requires_two_papers) and has_paper_b:
                    print(f"[DEBUG] skip case (requires single paper): {row['query']}")
                    continue

            output = agent.ask(row["query"])
            print(f"[DEBUG] answer: {output.answer}")
            answer_lower = output.answer.lower()
            hits = sum(1 for kw in row["must_include"] if kw.lower() in answer_lower)
            total = len(row["must_include"])
            keyword_score = hits / total if total else 0.0
            llm_score, llm_reason = self._llm_score(
                query=row["query"],
                answer=output.answer,
                reference_answer=row.get("reference_answer", ""),
            )
            criteria_raw = row.get("eval_criteria", [])
            criteria_list = criteria_raw if isinstance(criteria_raw, list) else []
            criteria_score, criteria_detail = self._criteria_score(
                answer=output.answer,
                criteria_list=criteria_list,
            )
            keyword_weight, llm_weight, criteria_weight = self._resolve_weights(row)
            final_score = (
                keyword_weight * keyword_score
                + llm_weight * llm_score
                + criteria_weight * criteria_score
            )
            results.append(
                EvalResult(
                    case_type=str(row.get("case_type", "unknown")),
                    query=row["query"],
                    hit_count=hits,
                    total_required=total,
                    keyword_score=keyword_score,
                    llm_score=llm_score,
                    criteria_score=criteria_score,
                    criteria_detail=criteria_detail,
                    llm_reason=llm_reason,
                    score=final_score,
                )
            )
        return results
