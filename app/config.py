from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://studio:studio@localhost:5432/studio"
    secret_key: str = "dev-only-change-me"
    access_token_minutes: int = 60 * 24
    storage_dir: str = "./storage"

    # LLM global settings
    llm_provider: str = "mock"
    llm_timeout_seconds: float = 30.0
    llm_max_output_tokens: int = 4096
    llm_temperature: float = 0.7

    # Gemini settings
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    gemini_thinking_level: str = "low"  # low | medium | high

    # OpenAI settings
    openai_api_key: str | None = None
    openai_model: str | None = None
    openai_reasoning_effort: str | None = None  # low | medium | high (if model supports it)

    # Optional per-task overrides (task: script, storyboard, metadata, factcheck)
    llm_provider_script: str | None = None
    llm_model_script: str | None = None
    llm_provider_storyboard: str | None = None
    llm_model_storyboard: str | None = None
    llm_provider_metadata: str | None = None
    llm_model_metadata: str | None = None
    llm_provider_factcheck: str | None = None
    llm_model_factcheck: str | None = None

    # Image generation settings
    image_provider: str = "mock"  # mock | openai
    openai_image_api_key: str | None = None
    openai_image_model: str | None = None
    openai_image_size: str = "1024x1536"
    openai_image_quality: str = "low"
    openai_image_est_cost_usd: float | None = None
    image_max_per_video: int = 12
    image_timeout_seconds: float = 120.0

    # Voiceover generation settings
    voiceover_provider: str = "mock"  # mock | kokoro
    kokoro_model_path: str = "./storage/models/kokoro/kokoro-v1.0.onnx"
    kokoro_voices_path: str = "./storage/models/kokoro/voices-v1.0.bin"
    voiceover_default_voice: str = "af_heart"
    voiceover_default_speed: float = 1.0
    voiceover_timeout_seconds: float = 60.0


settings = Settings()
