"""Task-based routing and factory for LLM providers."""

from typing import Any
from app.config import settings

from .base import BaseLLM
from .exceptions import LLMConfigError
from .gemini import GeminiLLM
from .mock import MockLLM
from .openai import OpenAILLM

# Single place where allowed tasks are defined.
# To add a new task, just add it here and optionally add override fields to app/config.py
ALLOWED_TASKS: tuple[str, ...] = ("script", "storyboard", "metadata", "factcheck")


def resolve_task_routing(task: str) -> tuple[str, str]:
    """Resolve the active provider and model for a given task.

    Precedence:
    1. Per-task override (e.g. LLM_PROVIDER_SCRIPT, LLM_MODEL_SCRIPT)
    2. Global default (LLM_PROVIDER, GEMINI_MODEL / OPENAI_MODEL)

    Returns:
        tuple of (provider_name, model_name)
    """
    clean_task = task.lower().strip()
    if clean_task not in ALLOWED_TASKS:
        allowed = ", ".join(ALLOWED_TASKS)
        raise LLMConfigError(f"Unknown task '{task}'. Allowed tasks are: {allowed}")

    # 1. Resolve Provider
    task_provider_attr = f"llm_provider_{clean_task}"
    provider = getattr(settings, task_provider_attr, None) or settings.llm_provider

    if not provider or not provider.strip():
        raise LLMConfigError(
            f"No provider configured for task '{clean_task}'. "
            f"Set LLM_PROVIDER or LLM_PROVIDER_{clean_task.upper()} in your environment."
        )

    provider = provider.lower().strip()

    # 2. Resolve Model
    task_model_attr = f"llm_model_{clean_task}"
    model = getattr(settings, task_model_attr, None)

    if not model or not model.strip():
        if provider == "gemini":
            model = settings.gemini_model
        elif provider == "openai":
            model = settings.openai_model
        elif provider == "mock":
            model = "mock-model"
        else:
            raise LLMConfigError(
                f"Unsupported LLM provider '{provider}' for task '{clean_task}'. "
                f"Supported providers: mock, gemini, openai"
            )

    return provider, model.strip()


def get_routing_table() -> dict[str, dict[str, Any]]:
    """Return the resolved provider and model for all allowed tasks."""
    table = {}
    for task in ALLOWED_TASKS:
        provider, model = resolve_task_routing(task)
        has_provider_override = getattr(settings, f"llm_provider_{task}", None) is not None
        has_model_override = getattr(settings, f"llm_model_{task}", None) is not None
        table[task] = {
            "provider": provider,
            "model": model,
            "has_provider_override": has_provider_override,
            "has_model_override": has_model_override,
        }
    return table


def create_llm_instance(provider: str, model: str | None = None) -> BaseLLM:
    """Create an LLM instance for a specific provider and optional model."""
    clean_provider = provider.lower().strip()

    if clean_provider == "mock":
        return MockLLM(
            model=model or "mock-model",
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
            temperature=settings.llm_temperature,
        )

    if clean_provider == "gemini":
        return GeminiLLM(
            model=model or settings.gemini_model,
            api_key=settings.gemini_api_key,
            thinking_level=settings.gemini_thinking_level,
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
            temperature=settings.llm_temperature,
        )

    if clean_provider == "openai":
        return OpenAILLM(
            model=model or settings.openai_model,
            api_key=settings.openai_api_key,
            reasoning_effort=settings.openai_reasoning_effort,
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
            temperature=settings.llm_temperature,
        )

    raise LLMConfigError(
        f"Unknown LLM provider '{provider}'. Supported providers are: mock, gemini, openai"
    )


def get_llm(task: str) -> BaseLLM:
    """Factory function: returns the configured LLM instance for a specific task.

    Args:
        task: Name of the task ('script', 'storyboard', 'metadata', 'factcheck').

    Raises:
        LLMConfigError: If the task name is unknown or configuration is missing.
    """
    provider, model = resolve_task_routing(task)
    return create_llm_instance(provider=provider, model=model)
