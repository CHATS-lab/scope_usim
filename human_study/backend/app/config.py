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

    # Per-condition API key override. Leave blank to fall back to
    # openai_api_key. Needed when a condition's base_url points to a private
    # SGLang server that uses a different bearer token than real OpenAI.
    openai_api_key_base: str = ""
    openai_api_key_rl_single: str = ""
    openai_api_key_cotraining: str = ""

    completion_code_secret: str = "change-me-in-production"

    # Prolific completion codes (one per task_type). The participant must be
    # redirected to https://app.prolific.com/submissions/complete?cc=<code>
    # — these are the codes set up in the Prolific study form, NOT the internal
    # HMAC code we generate for our own session verification.
    prolific_completion_code_tau2: str = ""
    prolific_completion_code_p4g: str = ""

    max_turns: int = 30
    max_session_seconds: int = 2400


settings = Settings()
