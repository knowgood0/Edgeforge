"""
AI provider abstraction.

AIProvider is the interface every LLM backend must implement. GroqProvider
is the initial concrete implementation. Adding OpenAI/Anthropic/local
later means writing one new class here — the research agents in
ai/agents.py never call an SDK directly, only AIProvider.complete().

API keys are read from environment variables ONLY, never accepted as
function arguments from request bodies, and never sent to the frontend.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from typing import Optional


class AIProviderError(Exception):
    pass


@dataclass
class AIResponse:
    text: str
    provider: str
    model: str
    raw: Optional[dict] = None


class AIProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def complete(self, system_prompt: str, user_prompt: str,
                 model: Optional[str] = None, temperature: float = 0.3,
                 max_tokens: int = 2000) -> AIResponse:
        raise NotImplementedError


class GroqProvider(AIProvider):
    name = "groq"
    default_model = "llama-3.3-70b-versatile"

    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise AIProviderError(
                "GROQ_API_KEY is not set. Add it to your environment "
                "(see .env.example) — EdgeForge will not fabricate AI "
                "analysis without real API access."
            )

    def complete(self, system_prompt: str, user_prompt: str,
                 model: Optional[str] = None, temperature: float = 0.3,
                 max_tokens: int = 2000) -> AIResponse:
        try:
            from groq import Groq
        except ImportError as e:
            raise AIProviderError("groq package not installed. Run: pip install groq") from e

        client = Groq(api_key=self.api_key)
        model = model or self.default_model
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as e:
            raise AIProviderError(f"Groq API call failed: {e}") from e

        text = resp.choices[0].message.content
        return AIResponse(text=text, provider=self.name, model=model, raw=resp.model_dump() if hasattr(resp, "model_dump") else None)


# Registry so ai/agents.py (and future task-routing logic) can request a
# provider by name without importing every implementation directly.
_PROVIDERS = {
    "groq": GroqProvider,
    # "openai": OpenAIProvider,      # TODO when needed
    # "anthropic": AnthropicProvider,  # TODO when needed
    # "local": LocalModelProvider,     # TODO when needed
}


def get_provider(name: str = "groq") -> AIProvider:
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown AI provider: {name}. Available: {list(_PROVIDERS)}")
    return _PROVIDERS[name]()


# Task -> provider/model routing table. MVP routes everything to Groq;
# this is the seam where "hypothesis generation uses model A, skepticism
# uses model B" gets implemented later without touching agents.py logic.
TASK_ROUTING = {
    "researcher": {"provider": "groq", "model": None},
    "quant_analyst": {"provider": "groq", "model": None},
    "skeptic": {"provider": "groq", "model": None},
    "strategist": {"provider": "groq", "model": None},
    "code_engineer": {"provider": "groq", "model": None},
    "reviewer": {"provider": "groq", "model": None},
}


def get_provider_for_task(task: str) -> tuple[AIProvider, Optional[str]]:
    route = TASK_ROUTING.get(task, {"provider": "groq", "model": None})
    return get_provider(route["provider"]), route["model"]
