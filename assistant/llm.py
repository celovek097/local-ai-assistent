from __future__ import annotations

from ollama import chat

from .config import LLMConfig
from .history import History


class LLM:
    """Обёртка над Ollama: формирует контекст с системным промптом и историей."""

    def __init__(self, cfg: LLMConfig, history: History):
        self.cfg = cfg
        self.history = history

    def _build_messages(self, user_text: str) -> list[dict]:
        messages: list[dict] = []
        if self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.extend(self.history.tail(self.cfg.history_window))
        messages.append({"role": "user", "content": user_text})
        return messages

    def respond(self, user_text: str) -> str:
        messages = self._build_messages(user_text)
        response = chat(
            model=self.cfg.model,
            messages=messages,
            options=self.cfg.options or None,
        )
        answer = response.message.content or ""
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})
        return answer