"""Unit tests for the LLM layer (no network required)."""

import pytest
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import (
    ALLOWED_TASKS,
    BaseLLM,
    LLMConfigError,
    create_llm_instance,
    get_llm,
    get_routing_table,
    resolve_task_routing,
)
from app.llm.gemini import GeminiLLM
from app.llm.mock import MockLLM
from app.llm.openai import OpenAILLM


class DemoModel(BaseModel):
    title: str
    scenes_count: int = 5
    tags: list[str] = Field(default_factory=list)


def test_factory_picks_default_provider(monkeypatch):
    """Test that get_llm picks the global default provider when no overrides are set."""
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_provider_script", None)
    monkeypatch.setattr(settings, "llm_model_script", None)

    llm = get_llm("script")
    assert isinstance(llm, MockLLM)
    assert llm.provider_name == "mock"
    assert llm.model == "mock-model"


def test_per_task_override_wins_over_default(monkeypatch):
    """Test that per-task provider and model overrides take precedence over global defaults."""
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_provider_script", "openai")
    monkeypatch.setattr(settings, "llm_model_script", "custom-gpt-test")

    provider, model = resolve_task_routing("script")
    assert provider == "openai"
    assert model == "custom-gpt-test"

    llm = get_llm("script")
    assert isinstance(llm, OpenAILLM)
    assert llm.provider_name == "openai"
    assert llm.model == "custom-gpt-test"


def test_missing_override_falls_back_to_default(monkeypatch):
    """Test that a task without an override falls back to global default provider and model."""
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_provider_storyboard", None)
    monkeypatch.setattr(settings, "llm_model_storyboard", None)

    provider, model = resolve_task_routing("storyboard")
    assert provider == "mock"
    assert model == "mock-model"

    llm = get_llm("storyboard")
    assert isinstance(llm, MockLLM)
    assert llm.provider_name == "mock"


def test_unknown_task_name_raises_clear_error():
    """Test that requesting an unknown task raises LLMConfigError naming the allowed tasks."""
    with pytest.raises(LLMConfigError) as exc_info:
        get_llm("non_existent_task")

    msg = str(exc_info.value)
    assert "Unknown task 'non_existent_task'" in msg
    for task in ALLOWED_TASKS:
        assert task in msg


def test_mock_works_for_text_generation():
    """Test deterministic text generation from MockLLM."""
    mock_llm = MockLLM(model="test-mock")
    prompt = "Create a hook about space"
    res = mock_llm.generate(prompt=prompt, system_prompt="You are an expert", task="script")

    assert res.provider == "mock"
    assert res.model == "test-mock"
    assert res.task == "script"
    assert "Response to: Create a hook about space" in res.output
    assert res.input_tokens > 0
    assert res.output_tokens > 0
    assert res.finish_reason == "stop"
    assert res.latency_seconds >= 0


def test_mock_works_for_structured_output():
    """Test structured output generation and Pydantic validation from MockLLM."""
    mock_llm = MockLLM()
    res = mock_llm.generate_structured(
        prompt="Generate video info",
        response_model=DemoModel,
        task="storyboard",
    )

    assert res.provider == "mock"
    assert res.task == "storyboard"
    assert isinstance(res.output, DemoModel)
    assert res.output.title == "mock_title"
    assert res.output.scenes_count == 5


def test_missing_api_key_raises_only_when_used():
    """Test that missing keys do not fail at instantiation, but raise clear errors on call."""
    gemini_llm = GeminiLLM(model="gemini-2.5-flash", api_key=None)
    openai_llm = OpenAILLM(model="gpt-4o-mini", api_key=None)

    # Calling Gemini without key must raise LLMConfigError naming GEMINI_API_KEY
    with pytest.raises(LLMConfigError) as exc_gemini:
        gemini_llm.generate("Hello Gemini")
    assert "GEMINI_API_KEY" in str(exc_gemini.value)

    # Calling OpenAI without key must raise LLMConfigError naming OPENAI_API_KEY
    with pytest.raises(LLMConfigError) as exc_openai:
        openai_llm.generate("Hello OpenAI")
    assert "OPENAI_API_KEY" in str(exc_openai.value)


def test_routing_table_inspection(monkeypatch):
    """Test that get_routing_table accurately lists all tasks and their override states."""
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_provider_script", "openai")
    monkeypatch.setattr(settings, "llm_model_script", "gpt-4o")

    table = get_routing_table()
    assert set(table.keys()) == set(ALLOWED_TASKS)
    assert table["script"]["provider"] == "openai"
    assert table["script"]["model"] == "gpt-4o"
    assert table["script"]["has_provider_override"] is True
    assert table["script"]["has_model_override"] is True

    assert table["storyboard"]["provider"] == "mock"
    assert table["storyboard"]["has_provider_override"] is False
