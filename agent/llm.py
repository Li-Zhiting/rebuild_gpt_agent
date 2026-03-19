from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str = "rule_based"


class BaseLLM:
    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError


class RuleBasedLLM(BaseLLM):
    """
    这是最小可运行占位实现。
    你后面可以把它替换成 OpenAI / Qwen API。
    """

    def generate(self, prompt: str) -> LLMResponse:
        text = (
            "[RuleBasedLLM 输出]\n"
            "当前为脚手架模式。你应尽快把这里替换为真实 LLM 调用。\n\n"
            f"收到的 prompt 前 500 字符：\n{prompt[:500]}"
        )
        return LLMResponse(text=text)


class LLMFactory:
    @staticmethod
    def build(provider: Optional[str] = None) -> BaseLLM:
        # 先返回可运行版本，避免你现在被 API 阻塞。
        return RuleBasedLLM()
