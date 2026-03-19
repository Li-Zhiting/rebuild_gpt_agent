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
    plan_intent: str
    plan_steps: list[str]
    tool_name: str
    answer: str


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
        self.conv_memory.add("user", query)
        plan = self.planner.make_plan(query)
        print(f"DEBUG plan: {plan}")
        tool = self.tools[plan.tool_name]
        tool_result = tool.run(query, self.doc_memory, self.conv_memory)
        self.conv_memory.add("assistant", tool_result.content)
        return AgentOutput(
            plan_intent=plan.intent,
            plan_steps=plan.steps,
            tool_name=tool_result.tool_name,
            answer=tool_result.content,
        )
