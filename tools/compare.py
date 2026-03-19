from __future__ import annotations

import json
import os
from typing import Dict

from openai import OpenAI

from agent.memory import ConversationMemory, DocumentMemory
from tools.base import ToolResult


class CompareTool:
    name = "compare"
    required_keys = ("原理", "方法", "效果")

    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url or os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            ),
        )

    def _build_system_prompt(self) -> str:
        return (
            "你是一个论文对比分析专家。\n"
            "你的唯一任务是：基于输入的 Paper A 和 Paper B 证据片段，对两篇论文进行客观对比。\n\n"
            "你必须严格输出 JSON，且只能是如下格式：\n"
            "{\n"
            '  "原理": "...",\n'
            '  "方法": "...",\n'
            '  "效果": "..."\n'
            "}\n\n"
            "规则：\n"
            "1. 仅输出 JSON，禁止输出任何额外文本。\n"
            '2. 键必须完整且固定为：\"原理\"、\"方法\"、\"效果\"。\n'
            "3. 仅可基于输入证据进行对比，不允许编造。\n"
            "4. 当某维度证据不足时，该字段输出空字符串 \"\"。"
        )

    def _build_user_prompt(self, query: str, chunks_a: list[str], chunks_b: list[str]) -> str:
        paper_a = "\n".join(f"chunk{i + 1}内容: {chunk}" for i, chunk in enumerate(chunks_a))
        paper_b = "\n".join(f"chunk{i + 1}内容: {chunk}" for i, chunk in enumerate(chunks_b))
        return (
            f"用户请求：{query}\n\n"
            "[Paper A]\n"
            f"{paper_a}\n\n"
            "[Paper B]\n"
            f"{paper_b}\n\n"
            "请从以下三个维度对比两篇论文：原理、方法、效果。"
        )

    def _normalize_analysis(self, data: Dict[str, object]) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for key in self.required_keys:
            value = data.get(key, "")
            normalized[key] = value if isinstance(value, str) else ""
        return normalized

    def _analyze_with_llm(self, query: str, chunks_a: list[str], chunks_b: list[str]) -> Dict[str, str]:
        user_prompt = self._build_user_prompt(query=query, chunks_a=chunks_a, chunks_b=chunks_b)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content.strip()
        return self._normalize_analysis(json.loads(content))

    def _fallback_analysis(self, chunks_a: list[str], chunks_b: list[str]) -> Dict[str, str]:
        a0 = chunks_a[0][:120] if chunks_a else ""
        b0 = chunks_b[0][:120] if chunks_b else ""
        return {
            "原理": f"Paper A 与 Paper B 在问题建模和理论假设上可能存在差异（A: {a0}；B: {b0}）。",
            "方法": f"两者方法路径可能不同，需结合具体算法流程进一步核对（A: {a0}；B: {b0}）。",
            "效果": "当前证据片段不足以稳定判断整体效果优劣，建议结合完整实验表格对比。",
        }

    def _render_content(self, analysis: Dict[str, str], chunks_a: list[str], chunks_b: list[str]) -> str:
        evidence_a = "\n---\n".join(chunk[:180] for chunk in chunks_a) if chunks_a else "无"
        evidence_b = "\n---\n".join(chunk[:180] for chunk in chunks_b) if chunks_b else "无"
        return (
            "【对比分析】\n"
            f"- 原理：{analysis['原理']}\n"
            f"- 方法：{analysis['方法']}\n"
            f"- 效果：{analysis['效果']}\n\n"
            "【证据片段】\n"
            "[Paper A]\n"
            f"{evidence_a}\n\n"
            "[Paper B]\n"
            f"{evidence_b}"
        )

    def run(
        self,
        query: str,
        doc_memory: DocumentMemory,
        conv_memory: ConversationMemory,
    ) -> ToolResult:
        chunks = doc_memory.top_k(query, k=6, source="both")
        chunks_a = [item["chunk"] for item in chunks if item["source"] == "paper_a"]
        chunks_b = [item["chunk"] for item in chunks if item["source"] == "paper_b"]

        if not chunks_a and not chunks_b:
            return ToolResult(tool_name=self.name, content="【对比分析】\n未检索到相关内容。")

        if not chunks_a or not chunks_b:
            return ToolResult(
                tool_name=self.name,
                content="【对比分析】\n当前仅检测到一篇论文内容；请提供 paper_a 和 paper_b 两篇论文后再进行对比。",
            )

        try:
            analysis = self._analyze_with_llm(query=query, chunks_a=chunks_a, chunks_b=chunks_b)
        except Exception:
            analysis = self._fallback_analysis(chunks_a=chunks_a, chunks_b=chunks_b)

        content = self._render_content(analysis=analysis, chunks_a=chunks_a, chunks_b=chunks_b)
        return ToolResult(tool_name=self.name, content=content)
