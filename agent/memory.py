from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from typing import Dict, List, Literal, Optional, TypedDict


class RetrievedChunk(TypedDict):
    source: Literal["paper_a", "paper_b"]
    chunk: str


@dataclass(slots=True)
class Turn: # 封装对话的基本单元
    role: str  # 谁 user/assistant / system 
    content: str # 说了什么 


@dataclass
class ConversationMemory: # 对话记忆容器
    turn_limit: int = 6 # 只保留最近6轮对话
    turns: List[Turn] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content)) # 添加新对话
        if len(self.turns) > self.turn_limit: # 只保留列表最后 N 个元素
            self.turns = self.turns[-self.turn_limit :]

    def render(self) -> str: # 将记忆格式 渲染成 llm 可以理解的字符串，这里是简单实现
        return "\n".join(f"[{t.role}] {t.content}" for t in self.turns)


# @dataclass
# class DocumentMemory: # 文档记忆、检索
#     source_text: str = "" # 原始完整文本
#     chunks: List[str] = field(default_factory=list)

#     def load(self, text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> None:
#         self.source_text = text
#         self.chunks = []
#         start = 0
#         while start < len(text):
#             end = min(len(text), start + chunk_size) # 计算当前窗口终点
#             self.chunks.append(text[start:end])
#             if end >= len(text):
#                 break
#             start = end - chunk_overlap # 下一块从当前重点往回退150字符

#     def top_k(self, query: str, k: int = 3) -> List[str]: # 检索策略
#         """
#         极简检索：按词重叠排序。
#         后续替换成 embedding + FAISS。
#         """
#         query_terms = {x.strip().lower() for x in query.split() if x.strip()}
#         scored = []
#         for chunk in self.chunks:
#             chunk_terms = {x.strip().lower() for x in chunk.split() if x.strip()}
#             score = len(query_terms & chunk_terms) # 计算交集，共现词的数量
#             scored.append((score, chunk))
#         scored.sort(key=lambda x: x[0], reverse=True) # 按照得分降序排列
#         return [chunk for _, chunk in scored[:k]] if scored else self.chunks[:k]

@dataclass
class DocumentMemory:
    source_texts: Dict[str, str] = field(default_factory = dict)
    chunks: Dict[str, List[str]] = field(default_factory = dict)

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 1200,
        chunk_overlap: int = 150
    ) -> List[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        
        results = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            results.append(text[start:end])
            if end >= len(text):
                break
            start = end - chunk_overlap
        return results

    def load(
        self,
        text_a: str,
        text_b: Optional[str] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 150
    ) -> None:
        """
        支持：
        - load(text_a)             -> 只加载 paper_a
        - load(text_a, text_b)     -> 加载两篇论文
        """
        self.source_texts = {
            "paper_a": text_a or "",
            "paper_b": text_b or "",
        }

        self.chunks = {
            "paper_a": self._chunk_text(
                self.source_texts["paper_a"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ),
            "paper_b": self._chunk_text(
                self.source_texts["paper_b"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ) if text_b else [],
        }

    def top_k(
        self,
        query: str,
        k: int = 3,
        source: Literal["paper_a", "paper_b", "both"] = "both"
    ) -> List[RetrievedChunk]:
        if source not in {"paper_a", "paper_b", "both"}:
            raise ValueError("source must be one of: 'paper_a', 'paper_b', 'both'")

        query_terms = {x.strip().lower() for x in query.split() if x.strip()}

        if source == "paper_a":
            candidate_chunks = [
                {"source": "paper_a", "chunk": chunk}
                for chunk in self.chunks.get("paper_a", [])
            ]
        elif source == "paper_b":
            candidate_chunks = [
                {"source": "paper_b", "chunk": chunk}
                for chunk in self.chunks.get("paper_b", [])
            ]
        else:
            candidate_chunks = (
                [
                    {"source": "paper_a", "chunk": chunk}
                    for chunk in self.chunks.get("paper_a", [])
                ] +
                [
                    {"source": "paper_b", "chunk": chunk}
                    for chunk in self.chunks.get("paper_b", [])
                ]
            )

        if not candidate_chunks:
            return []

        scored = []
        for item in candidate_chunks:
            chunk_terms = {x.strip().lower() for x in item["chunk"].split() if x.strip()}
            score = len(query_terms & chunk_terms)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]


if __name__ == "__main__":
    memory = DocumentMemory()

    paper_a = (
        "This paper studies time series forecasting methods for retail demand. "
        "We compare classical baselines and neural forecasting models."
    )
    paper_b = (
        "This paper focuses on representation learning for sensor data. "
        "It includes transfer learning experiments and downstream tasks."
    )

    print("=== Single paper: source='paper_a' ===")
    memory.load(paper_a)
    print(memory.top_k("forecasting", source="paper_a"))

    print("\n=== Two papers: source='both' ===")
    memory.load(paper_a, paper_b)
    print(memory.top_k("forecasting", source="both"))
