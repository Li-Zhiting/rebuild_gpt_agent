from __future__ import annotations

from agent.memory import ConversationMemory, DocumentMemory
from tools.base import ToolResult


class SummarizeTool:
    name = "summarize"

    def run(
        self,
        query: str,
        doc_memory: DocumentMemory,
        conv_memory: ConversationMemory,
    ) -> ToolResult:
        chunks = doc_memory.top_k(query, k=3)
        content = (
            "【结构化总结】\n"
            "1. 研究问题：\n"
            f"{chunks[0][:250]}\n\n"
            "2. 方法概要：\n"
            f"{chunks[1][:250] if len(chunks) > 1 else chunks[0][:250]}\n\n"
            "3. 实验与结论：\n"
            f"{chunks[2][:250] if len(chunks) > 2 else chunks[0][:250]}\n"
        )
        return ToolResult(tool_name=self.name, content=content)
