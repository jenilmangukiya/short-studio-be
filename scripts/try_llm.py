#!/usr/bin/env python3
"""CLI utility to test and inspect the LLM layer."""

import argparse
import sys
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from app.llm import (
    ALLOWED_TASKS,
    LLMConfigError,
    LLMError,
    LLMTruncatedError,
    create_llm_instance,
    get_llm,
    get_routing_table,
)


class SampleStructuredOutput(BaseModel):
    """Demo Pydantic model for testing structured output."""
    title: str = Field(description="Short video title")
    hook: str = Field(description="Attention-grabbing opening hook")
    hashtags: list[str] = Field(description="Two hashtags for the video")


def print_routing_table():
    """Print the resolved task routing table in a clean, human-readable format."""
    table = get_routing_table()
    print("\n" + "=" * 65)
    print("Resolved LLM Routing Table (Task -> Provider -> Model)")
    print("=" * 65)
    print(f"{'Task':<14} | {'Provider':<10} | {'Model':<22} | {'Override':<10}")
    print("-" * 65)
    for task, info in table.items():
        override_str = []
        if info["has_provider_override"]:
            override_str.append("provider")
        if info["has_model_override"]:
            override_str.append("model")
        override_display = "+".join(override_str) if override_str else "default"
        print(f"{task:<14} | {info['provider']:<10} | {info['model']:<22} | {override_display:<10}")
    print("=" * 65 + "\n")


def print_result(result):
    """Format and print an LLMResult object."""
    print("\n" + "=" * 60)
    print("LLM Execution Result")
    print("=" * 60)
    print(f"Provider:        {result.provider}")
    print(f"Model:           {result.model}")
    print(f"Task:            {result.task or 'N/A'}")
    print(f"Latency:         {result.latency_seconds:.4f}s")
    print(f"Input Tokens:    {result.input_tokens}")
    print(f"Output Tokens:   {result.output_tokens}")
    print(f"Thinking Tokens: {result.thinking_tokens if result.thinking_tokens is not None else 'N/A'}")
    print(f"Finish Reason:   {result.finish_reason or 'N/A'}")
    print("-" * 60)
    print("Output:")
    if isinstance(result.output, BaseModel):
        print(result.output.model_dump_json(indent=2))
    else:
        print(result.output)
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test and inspect Shorts Studio LLM routing and providers."
    )
    parser.add_argument(
        "--show-routing",
        action="store_true",
        help="Print the resolved task -> provider -> model routing table.",
    )
    parser.add_argument(
        "--task",
        choices=ALLOWED_TASKS,
        help="Call LLM via task routing (e.g. script, storyboard, metadata, factcheck).",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "gemini", "openai"],
        help="Call a provider directly (e.g. mock, gemini, openai).",
    )
    parser.add_argument(
        "--model",
        help="Optional model override when calling a provider directly.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a 15-second YouTube Short script about black holes.",
        help="Prompt to send to the LLM.",
    )
    parser.add_argument(
        "--system-prompt",
        help="Optional system instruction for the LLM.",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Test structured output using a sample Pydantic model.",
    )
    parser.add_argument(
        "--raw-usage",
        action="store_true",
        help="Print raw SDK response usage metadata and thinking level sent.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["low", "medium", "high"],
        help="Override thinking level for Gemini directly.",
    )

    args = parser.parse_args()

    if args.show_routing:
        print_routing_table()
        if not args.provider and not args.task:
            return

    if not args.provider and not args.task:
        if not args.show_routing:
            parser.print_help()
            print("\nExample commands:")
            print("  python scripts/try_llm.py --show-routing")
            print("  python scripts/try_llm.py --provider mock --prompt 'Tell me a joke'")
            print("  python scripts/try_llm.py --task script --prompt '10-sec hook on coffee'")
        return

    try:
        if args.task:
            llm = get_llm(task=args.task)
            task_name = args.task
        else:
            llm = create_llm_instance(provider=args.provider, model=args.model)
            task_name = None

        if args.thinking_level and hasattr(llm, "thinking_level"):
            llm.thinking_level = args.thinking_level

        captured = {}
        if args.raw_usage and hasattr(llm, "_call_with_retry"):
            orig_call = llm._call_with_retry
            def intercept(action_desc, func):
                resp = orig_call(action_desc, func)
                captured["raw_response"] = resp
                return resp
            llm._call_with_retry = intercept

        if args.structured:
            result = llm.generate_structured(
                prompt=args.prompt,
                response_model=SampleStructuredOutput,
                system_prompt=args.system_prompt,
                task=task_name,
            )
        else:
            result = llm.generate(
                prompt=args.prompt,
                system_prompt=args.system_prompt,
                task=task_name,
            )

        print_result(result)

        if args.raw_usage:
            print("=" * 60)
            print("Raw Usage Metadata & Thinking Config")
            print("=" * 60)
            if hasattr(llm, "thinking_level"):
                print(f"Thinking Level Sent: {llm.thinking_level}")
            if hasattr(llm, "reasoning_effort"):
                print(f"Reasoning Effort Sent: {llm.reasoning_effort or 'default'}")
            raw_resp = captured.get("raw_response")
            if raw_resp and hasattr(raw_resp, "usage_metadata") and raw_resp.usage_metadata:
                um = raw_resp.usage_metadata
                fields = getattr(type(um), "model_fields", {}).keys()
                for field in fields:
                    val = getattr(um, field, None)
                    print(f"  {field}: {val}")
            elif raw_resp and hasattr(raw_resp, "usage") and raw_resp.usage:
                u = raw_resp.usage
                print(f"  prompt_tokens: {u.prompt_tokens}")
                print(f"  completion_tokens: {u.completion_tokens}")
                print(f"  total_tokens: {u.total_tokens}")
                details = getattr(u, "completion_tokens_details", None)
                if details:
                    print(f"  completion_tokens_details.reasoning_tokens: {getattr(details, 'reasoning_tokens', None)}")
                    print(f"  completion_tokens_details.accepted_prediction_tokens: {getattr(details, 'accepted_prediction_tokens', None)}")
                    print(f"  completion_tokens_details.rejected_prediction_tokens: {getattr(details, 'rejected_prediction_tokens', None)}")
                prompt_details = getattr(u, "prompt_tokens_details", None)
                if prompt_details:
                    print(f"  prompt_tokens_details.cached_tokens: {getattr(prompt_details, 'cached_tokens', None)}")
            else:
                print("  usage_metadata / usage: None or not available")
            print("=" * 60 + "\n")

    except LLMConfigError as exc:
        print(f"\n[Configuration Error] {exc}\n", file=sys.stderr)
        sys.exit(1)
    except LLMTruncatedError as exc:
        print(f"\n[Truncation Error] {exc}\n", file=sys.stderr)
        sys.exit(1)
    except LLMError as exc:
        print(f"\n[LLM Error] {exc}\n", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
