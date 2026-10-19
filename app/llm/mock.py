"""Mock LLM provider for local development and testing with zero AI cost."""

from typing import Any, Type, TypeVar, get_args, get_origin
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from .base import BaseLLM

T = TypeVar("T", bound=BaseModel)


def _build_mock_instance(model_cls: Type[T]) -> T:
    """Recursively constructs a dummy valid instance of a Pydantic model."""
    data = {}
    for name, field in model_cls.model_fields.items():
        if field.default is not PydanticUndefined:
            data[name] = field.default
            continue
        if field.default_factory is not None:
            data[name] = field.default_factory()
            continue

        annotation = field.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)

        # Handle Optional/Union types where None might be allowed
        if origin is list:
            if args and hasattr(args[0], "model_fields"):
                data[name] = [_build_mock_instance(args[0])]
            elif args and args[0] is int:
                data[name] = [1]
            elif args and args[0] is float:
                data[name] = [1.0]
            else:
                data[name] = [f"mock_{name}_item"]
        elif origin is dict:
            data[name] = {}
        elif hasattr(annotation, "model_fields"):
            data[name] = _build_mock_instance(annotation)
        elif annotation is int:
            data[name] = 1
        elif annotation is float:
            data[name] = 1.0
        elif annotation is bool:
            data[name] = True
        else:
            data[name] = f"mock_{name}"

    return model_cls(**data)


class MockLLM(BaseLLM):
    """Mock LLM provider requiring no API keys and making no network requests."""

    def __init__(
        self,
        model: str = "mock-model",
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        super().__init__(
            model=model,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    @property
    def provider_name(self) -> str:
        return "mock"

    def _execute_text(
        self, prompt: str, system_prompt: str | None
    ) -> tuple[str, int, int, int | None, str | None]:
        sys_info = f" [System: {system_prompt}]" if system_prompt else ""
        text = f"[Mock LLM output ({self.model})]{sys_info} Response to: {prompt}"
        in_tokens = max(1, len(prompt.split()))
        out_tokens = max(1, len(text.split()))
        return text, in_tokens, out_tokens, None, "stop"

    def _execute_structured(
        self, prompt: str, response_model: Type[T], system_prompt: str | None
    ) -> tuple[Any, int, int, int | None, str | None]:
        instance = _build_mock_instance(response_model)
        in_tokens = max(1, len(prompt.split()))
        out_tokens = 25
        return instance, in_tokens, out_tokens, None, "stop"
