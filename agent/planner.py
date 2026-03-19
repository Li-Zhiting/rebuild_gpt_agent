import json
import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI


ToolName = Literal["summary", "compare", "critique"]


@dataclass
class Plan:
    tool_name: ToolName
    reason: str


class Planner:
    """
    LLM-based planner:
    - 输入用户 query
    - 输出一个 plan，只选择 3 个 tool 中的 1 个
    """

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

    def make_plan(self, query: str) -> Plan:
        """
        使用真实 Qwen API 决策应该调用哪个 tool。
        只允许返回:
        - summary
        - compare
        - critique
        """
        system_prompt = """
你是一个论文分析 Agent 的 Planner。
你的唯一任务是：根据用户请求，从以下三个工具中选择一个最合适的工具。

可选工具：
1. summary
   - 用于总结论文核心内容、贡献、方法、实验结果
   - 典型请求：总结、概述、介绍这篇论文讲了什么

2. compare
   - 用于比较两篇论文、两个方法、两个模型
   - 典型请求：对比、比较、异同、谁更好

3. critique
   - 用于分析论文缺点、局限性、问题、不足、可改进点
   - 典型请求：评价不足、局限、问题、批判性分析

你必须严格输出 JSON，且只能是如下格式：
{
  "tool_name": "summary" | "compare" | "critique",
  "reason": "一句简短中文解释"
}

禁止输出任何额外文本。
""".strip()

        user_prompt = f"用户请求：{query}"

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
            data = json.loads(content)

            tool_name = data["tool_name"]
            reason = data.get("reason", "")

            if tool_name not in {"summary", "compare", "critique"}:
                raise ValueError(f"Invalid tool_name: {tool_name}")

            return Plan(tool_name=tool_name, reason=reason)

        except Exception:
            # 兜底策略：当 LLM 输出不稳定、JSON 解析失败、API 异常时，回退到简单规则
            return self._fallback_plan(query)

    def _fallback_plan(self, query: str) -> Plan:
        q = query.lower()

        compare_keywords = ["对比", "比较", "异同", "区别", "哪个更好", "versus", "compare"]
        critique_keywords = ["缺点", "不足", "局限", "问题", "批判", "评价一下问题", "weakness", "limitation"]
        summary_keywords = ["总结", "概述", "介绍", "讲了什么", "summary", "summarize"]

        if any(k in q for k in compare_keywords):
            return Plan(tool_name="compare", reason="fallback: 命中 compare 关键词")
        if any(k in q for k in critique_keywords):
            return Plan(tool_name="critique", reason="fallback: 命中 critique 关键词")
        if any(k in q for k in summary_keywords):
            return Plan(tool_name="summary", reason="fallback: 命中 summary 关键词")

        return Plan(tool_name="summary", reason="fallback: 默认走 summary")
    
    
if __name__ == "__main__":
    planner = Planner(
        model="qwen-plus",  # 也可以换成你实际可用的 qwen 模型
    )

    test_queries = [
        "请总结这篇论文的核心贡献",
        "请比较这两篇论文的方法差异",
        "这篇论文有哪些局限性？",
    ]

    for q in test_queries:
        plan = planner.make_plan(q)
        print(f"query={q}")
        print(f"tool={plan.tool_name}, reason={plan.reason}")
        print("-" * 50)
