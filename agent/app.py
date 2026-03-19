from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from config import settings
from agent.memory import ConversationMemory, DocumentMemory
from agent.planner import Planner
from tools.compare import CompareTool
from tools.critique import CritiqueTool
from tools.summarize import SummarizeTool


@dataclass(slots=True)
class AgentOutput:
    tool_name: str
    answer: str
    plan_reason: str


class ResearchPaperAgent:
    def __init__(self) -> None:
        self.planner = Planner()
        self.conv_memory = ConversationMemory(turn_limit=settings.history_turn_limit)
        self.doc_memory = DocumentMemory()
        self.tools: Dict[str, object] = {
            "summarize": SummarizeTool(),
            "critique": CritiqueTool(),
            "compare": CompareTool(),
        }

    def load_document(self, text: str) -> None:
        self.doc_memory.load(
            text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def ask(self, query: str) -> AgentOutput:
        plan = self.planner.make_plan(query)
        print(f"[DEBUG] planner selected: {plan.tool_name}, reason={plan.reason}")

        tool = self.tools.get(plan.tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {plan.tool_name}")

        tool_result = tool.run(query, self.doc_memory, self.conv_memory)

        self.conv_memory.add("assistant", tool_result.content)

        return AgentOutput(
            tool_name=plan.tool_name,
            answer=tool_result.content,
            plan_reason=plan.reason,
            )
