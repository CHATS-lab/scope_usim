from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://usim:usim@localhost:5432/usim_study"
    frontend_origin: str = "http://localhost:3000"

    openai_api_key: str = ""

    model_base: str = "gpt-5.4"
    model_rl_single: str = "gpt-5.4"
    model_cotraining: str = "gpt-5.4"

    openai_base_url_base: str = ""
    openai_base_url_rl_single: str = ""
    openai_base_url_cotraining: str = ""

    completion_code_secret: str = "change-me-in-production"

    max_turns: int = 30
    max_session_seconds: int = 2400


settings = Settings()
