from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent.memory import ConversationMemory, DocumentMemory


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    content: str


class Tool(Protocol):
    name: str

    def run(
        self,
        query: str,
        doc_memory: DocumentMemory,
        conv_memory: ConversationMemory,
    ) -> ToolResult:
        ...
