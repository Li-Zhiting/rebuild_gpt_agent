from __future__ import annotations

import json
import os
from typing import Dict

from openai import OpenAI

from agent.memory import ConversationMemory, DocumentMemory
from tools.base import ToolResult


class SummarizeTool:
    name = "summary"
    required_keys = ("研究问题", "方法概要", "实验与结论")

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
            "你是一个论文总结专家。\n"
            "你的唯一任务是：根据输入文本进行结构化总结，围绕“研究问题”“方法概要”“实验与结论”三个维度作答。\n\n"
            "你必须严格输出 JSON，且只能是如下格式：\n"
            "{\n"
            '  "研究问题": "...",\n'
            '  "方法概要": "...",\n'
            '  "实验与结论": "..."\n'
            "}\n\n"
            "规则：\n"
            "1. 仅输出 JSON，禁止输出任何额外文本。\n"
            '2. 键必须完整且固定为：\"研究问题\"、\"方法概要\"、\"实验与结论\"。\n'
            "3. 仅可基于输入证据进行总结，不允许编造。\n"
            "4. 当证据不足时，对应字段输出空字符串 \"\"。"
        )

    def _build_user_prompt(self, query: str, chunks: list[str]) -> str:
        context = "\n\n".join(
            f"[chunk_{idx + 1}]\n{chunk}" for idx, chunk in enumerate(chunks)
        )
        return (
            f"用户请求：{query}\n\n"
            "请仅基于以下证据片段进行结构化总结：\n"
            f"{context}"
        )

    def _normalize_summary(self, data: Dict[str, object]) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for key in self.required_keys:
            value = data.get(key, "")
            normalized[key] = value if isinstance(value, str) else ""
        return normalized

    def _summarize_with_llm(self, query: str, chunks: list[str]) -> Dict[str, str]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(query=query, chunks=chunks)},
            ],
        )
        content = response.choices[0].message.content.strip()
        return self._normalize_summary(json.loads(content))

    def _fallback_summary(self, chunks: list[str]) -> Dict[str, str]:
        c0 = chunks[0][:250] if chunks else ""
        c1 = chunks[1][:250] if len(chunks) > 1 else c0
        c2 = chunks[2][:250] if len(chunks) > 2 else c0
        return {
            "研究问题": c0,
            "方法概要": c1,
            "实验与结论": c2,
        }

    def _render_content(self, summary: Dict[str, str], chunks: list[str]) -> str:
        evidence = "\n---\n".join(chunk[:220] for chunk in chunks) if chunks else "无"
        return (
            "【结构化总结】\n"
            "1. 研究问题：\n"
            f"{summary['研究问题']}\n\n"
            "2. 方法概要：\n"
            f"{summary['方法概要']}\n\n"
            "3. 实验与结论：\n"
            f"{summary['实验与结论']}\n\n"
            "【证据片段】\n"
            f"{evidence}"
        )

    def run(
        self,
        query: str,
        doc_memory: DocumentMemory,
        conv_memory: ConversationMemory,
    ) -> ToolResult:
        retrieved = doc_memory.top_k(query, k=3)
        chunks = [item["chunk"] for item in retrieved]

        if not chunks:
            return ToolResult(tool_name=self.name, content="【结构化总结】\n未检索到相关内容。")

        try:
            summary = self._summarize_with_llm(query=query, chunks=chunks)
        except Exception:
            summary = self._fallback_summary(chunks=chunks)

        content = self._render_content(summary=summary, chunks=chunks)
        return ToolResult(tool_name=self.name, content=content)
