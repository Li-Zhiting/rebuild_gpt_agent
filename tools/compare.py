from __future__ import annotations

from agent.memory import ConversationMemory, DocumentMemory
from tools.base import ToolResult


class CompareTool:
    name = "compare"

    def run(
        self,
        query: str,
        doc_memory: DocumentMemory,
        conv_memory: ConversationMemory,
    ) -> ToolResult:
        chunks = doc_memory.top_k(query, k=4)
        content = (
            "【对比分析】\n"
            "维度1：任务定义\n"
            f"{chunks[0][:180]}\n\n"
            "维度2：方法差异\n"
            f"{chunks[1][:180] if len(chunks) > 1 else chunks[0][:180]}\n\n"
            "维度3：实验表现\n"
            f"{chunks[2][:180] if len(chunks) > 2 else chunks[0][:180]}\n\n"
            "维度4：适用场景\n"
            f"{chunks[3][:180] if len(chunks) > 3 else chunks[0][:180]}\n"
        )
        return ToolResult(tool_name=self.name, content=content)
