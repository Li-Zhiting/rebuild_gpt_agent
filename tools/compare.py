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
        chunks = doc_memory.top_k(query, k=6, source="both")
        chunks_a = [item["chunk"] for item in chunks if item["source"] == "paper_a"]
        chunks_b = [item["chunk"] for item in chunks if item["source"] == "paper_b"]

        if not chunks_a and not chunks_b:
            return ToolResult(tool_name=self.name, content="【对比分析】\n未检索到相关内容。")

        fallback_a = chunks_a[0] if chunks_a else ""
        fallback_b = chunks_b[0] if chunks_b else ""

        def pick(items: list[str], idx: int, fallback: str) -> str:
            if items and len(items) > idx:
                return items[idx]
            if items:
                return items[0]
            return fallback

        content = (
            "【对比分析】\n"
            "维度1：任务定义\n"
            f"paper_a: {pick(chunks_a, 0, fallback_b)[:180]}\n"
            f"paper_b: {pick(chunks_b, 0, fallback_a)[:180]}\n\n"
            "维度2：方法差异\n"
            f"paper_a: {pick(chunks_a, 1, fallback_b)[:180]}\n"
            f"paper_b: {pick(chunks_b, 1, fallback_a)[:180]}\n\n"
            "维度3：实验表现\n"
            f"paper_a: {pick(chunks_a, 2, fallback_b)[:180]}\n"
            f"paper_b: {pick(chunks_b, 2, fallback_a)[:180]}\n\n"
            "维度4：适用场景\n"
            f"paper_a: {pick(chunks_a, 3, fallback_b)[:180]}\n"
            f"paper_b: {pick(chunks_b, 3, fallback_a)[:180]}\n"
        )
        return ToolResult(tool_name=self.name, content=content)
