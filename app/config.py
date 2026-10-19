from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://studio:studio@localhost:5432/studio"
    secret_key: str = "dev-only-change-me"
    access_token_minutes: int = 60 * 24
    storage_dir: str = "./storage"


settings = Settings()
