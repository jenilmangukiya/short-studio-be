"""Unit tests for the LLM layer (no network required)."""

import pytest
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import (
    ALLOWED_TASKS,
    BaseLLM,
    LLMConfigError,
    LLMProviderError,
    LLMTruncatedError,
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


class TruncatedDummyLLM(BaseLLM):
    """Test helper for simulating specific finish reasons."""

    def __init__(self, finish_reason: str):
        super().__init__(model="test-model", max_output_tokens=100)
        self._finish_reason = finish_reason

    @property
    def provider_name(self) -> str:
        return "dummy"

    def _execute_text(self, prompt: str, system_prompt: str | None):
        return "cut off text", 10, 100, None, self._finish_reason

    def _execute_structured(self, prompt: str, response_model, system_prompt: str | None):
        return {"title": "incomplete"}, 10, 100, None, self._finish_reason


def test_truncation_raises_llm_truncated_error():
    """Test that MAX_TOKENS and length finish reasons raise LLMTruncatedError naming LLM_MAX_OUTPUT_TOKENS."""
    # Verify inheritance
    assert issubclass(LLMTruncatedError, LLMProviderError)

    # Test text generation with MAX_TOKENS (Gemini style)
    gemini_truncated = TruncatedDummyLLM(finish_reason="MAX_TOKENS")
    with pytest.raises(LLMTruncatedError) as exc_max_tokens:
        gemini_truncated.generate("Write something long")
    err_str = str(exc_max_tokens.value)
    assert "LLM_MAX_OUTPUT_TOKENS" in err_str
    assert "token limit" in err_str.lower()
    assert "MAX_TOKENS" in err_str

    # Test text generation with length (OpenAI style)
    openai_truncated = TruncatedDummyLLM(finish_reason="length")
    with pytest.raises(LLMTruncatedError) as exc_length:
        openai_truncated.generate("Write something long")
    assert "LLM_MAX_OUTPUT_TOKENS" in str(exc_length.value)
    assert "token limit" in str(exc_length.value).lower()
    assert "length" in str(exc_length.value)

    # Test structured output truncation
    with pytest.raises(LLMTruncatedError) as exc_struct:
        gemini_truncated.generate_structured("Generate structured data", response_model=DemoModel)
    assert "LLM_MAX_OUTPUT_TOKENS" in str(exc_struct.value)


def test_missing_model_raises_clear_error_naming_setting(monkeypatch):
    """Test that missing model configurations raise LLMConfigError naming the setting."""
    # Gemini via router
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_model", None)
    monkeypatch.setattr(settings, "llm_provider_script", None)
    monkeypatch.setattr(settings, "llm_model_script", None)

    with pytest.raises(LLMConfigError) as exc_gemini:
        get_llm("script")
    assert "GEMINI_MODEL" in str(exc_gemini.value)

    # Gemini via create_llm_instance directly
    with pytest.raises(LLMConfigError) as exc_gemini_direct:
        create_llm_instance("gemini", model=None)
    assert "GEMINI_MODEL" in str(exc_gemini_direct.value)

    # OpenAI via router
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_model", None)
    monkeypatch.setattr(settings, "llm_provider_script", None)
    monkeypatch.setattr(settings, "llm_model_script", None)

    with pytest.raises(LLMConfigError) as exc_openai:
        get_llm("script")
    assert "OPENAI_MODEL" in str(exc_openai.value)

    # OpenAI via create_llm_instance directly
    with pytest.raises(LLMConfigError) as exc_openai_direct:
        create_llm_instance("openai", model=None)
    assert "OPENAI_MODEL" in str(exc_openai_direct.value)


def test_mock_works_with_no_model_set(monkeypatch):
    """Test that mock provider continues to work with no model configured."""
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_model_script", None)

    # Router factory
    llm = get_llm("script")
    assert isinstance(llm, MockLLM)
    assert llm.model == "mock-model"

    # Direct factory with model=None
    direct_llm = create_llm_instance("mock", model=None)
    assert isinstance(direct_llm, MockLLM)
    assert direct_llm.model == "mock-model"

    # Direct class instantiation
    instantiated_llm = MockLLM()
    assert instantiated_llm.model == "mock-model"

