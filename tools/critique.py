from __future__ import annotations

import json
import os
from typing import Dict

from openai import OpenAI

from agent.memory import ConversationMemory, DocumentMemory
from tools.base import ToolResult


class CritiqueTool:
    name = "critique"
    required_keys = ("优点", "局限1", "局限2", "局限3")

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
            "你是一个对输入文本进行理性客观评判的专家。\n"
            "你的唯一任务是：根据输入文本，对其缺点、局限性、问题、不足、可改进点进行评价，"
            "并将评价内容根据“优点”“局限1”“局限2”“局限3”这四个点分别作答。\n\n"
            "你必须严格输出 JSON，且只能是如下格式：\n"
            "{\n"
            '  "优点": "...",\n'
            '  "局限1": "...",\n'
            '  "局限2": "...",\n'
            '  "局限3": "..."\n'
            "}\n\n"
            "规则：\n"
            "1. 仅输出 JSON，禁止输出任何额外文本。\n"
            '2. 键必须完整且固定为：\"优点\"、\"局限1\"、\"局限2\"、\"局限3\"。\n'
            "3. 当证据不足以支撑某一字段时，该字段输出空字符串 \"\"。\n"
            "4. 不允许编造论文中不存在的信息。"
        )

    def _normalize_analysis(self, data: Dict[str, object]) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for key in self.required_keys:
            value = data.get(key, "")
            normalized[key] = value if isinstance(value, str) else ""
        return normalized

    def _analyze_with_llm(self, query: str, chunks: list[str]) -> Dict[str, str]:
        context = "\n\n".join(
            f"[chunk_{idx + 1}]\n{chunk}" for idx, chunk in enumerate(chunks)
        )
        user_prompt = (
            f"用户请求：{query}\n\n"
            "请仅基于以下证据片段进行评判：\n"
            f"{context}"
        )
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

    def _fallback_analysis(self) -> Dict[str, str]:
        return {
            "优点": "方法目标明确，具备一定系统性。",
            "局限1": "是否存在数据集或任务范围过窄的问题？",
            "局限2": "是否缺少更强 baseline 或消融实验？",
            "局限3": "是否缺少真实场景部署或鲁棒性验证？",
        }

    def _render_content(self, analysis: Dict[str, str], chunks: list[str]) -> str:
        evidence = "\n---\n".join(chunk[:220] for chunk in chunks) if chunks else "无"
        return (
            "【批判性分析】\n"
            f"- 优点：{analysis['优点']}\n"
            f"- 局限1：{analysis['局限1']}\n"
            f"- 局限2：{analysis['局限2']}\n"
            f"- 局限3：{analysis['局限3']}\n\n"
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

        try:
            analysis = self._analyze_with_llm(query=query, chunks=chunks)
        except Exception:
            analysis = self._fallback_analysis()

        content = self._render_content(analysis=analysis, chunks=chunks)
        return ToolResult(tool_name=self.name, content=content)
