from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class Plan:
    intent: str # 意图标识
    steps: List[str] # 执行步骤
    tool_name: str # 工具名

# ========== 2. 规划器（Planner）只负责"决策" ==========
class Planner:
    """
    先用规则路由，后续再替换为 LLM Planner。
    你当前要的是结构，不是炫技。
    """

    def make_plan(self, query: str) -> Plan:
        lowered = query.lower() # 将用户输入统一转换为小写。消除大小写差异对关键词匹配的影响
        if any(word in lowered for word in ["compare", "对比", "区别"]): # 意图识别，只要出现一个相关关键词就触发
            return Plan( # 生成一个结构化计划
                intent="compare_papers_or_methods",
                steps=["识别比较对象", "抽取关键段落", "生成对比结论"],
                tool_name="compare",
            )
        if any(word in lowered for word in ["weakness", "缺点", "不足", "批判"]):
            return Plan(
                intent="critique_paper",
                steps=["抽取方法与实验描述", "识别局限", "生成批判意见"],
                tool_name="critique",
            )
        return Plan( # 默认兜底策略，如果以上规则都不匹配，就归类为 总结论文 
            intent="summarize_paper",
            steps=["读取论文", "抽取核心信息", "生成结构化总结"],
            tool_name="summarize",
        )
