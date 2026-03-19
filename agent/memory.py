from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


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


@dataclass
class DocumentMemory: # 文档记忆、检索
    source_text: str = "" # 原始完整文本
    chunks: List[str] = field(default_factory=list)

    def load(self, text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> None:
        self.source_text = text
        self.chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size) # 计算当前窗口终点
            self.chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - chunk_overlap # 下一块从当前重点往回退150字符

    def top_k(self, query: str, k: int = 3) -> List[str]: # 检索策略
        """
        极简检索：按词重叠排序。
        后续替换成 embedding + FAISS。
        """
        query_terms = {x.strip().lower() for x in query.split() if x.strip()}
        scored = []
        for chunk in self.chunks:
            chunk_terms = {x.strip().lower() for x in chunk.split() if x.strip()}
            score = len(query_terms & chunk_terms) # 计算交集，共现词的数量
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True) # 按照得分降序排列
        return [chunk for _, chunk in scored[:k]] if scored else self.chunks[:k]
