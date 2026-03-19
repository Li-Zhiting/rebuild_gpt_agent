from __future__ import annotations

from agent.memory import ConversationMemory, DocumentMemory
from tools.base import ToolResult


class CritiqueTool:
    name = "critique"

    def run(
        self,
        query: str,
        doc_memory: DocumentMemory,
        conv_memory: ConversationMemory,
    ) -> ToolResult:
        retrieved = doc_memory.top_k(query, k=3)
        chunks = [item["chunk"] for item in retrieved]
        content = (
            "【批判性分析】\n"
            "- 可能的优点：方法目标明确，具备一定系统性。\n"
            "- 可能的局限1：是否存在数据集或任务范围过窄的问题？\n"
            "- 可能的局限2：是否缺少更强 baseline 或消融实验？\n"
            "- 可能的局限3：是否缺少真实场景部署或鲁棒性验证？\n\n"
            "【证据片段】\n"
            + "\n---\n".join(chunk[:220] for chunk in chunks)
        )
        return ToolResult(tool_name=self.name, content=content)
