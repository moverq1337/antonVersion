from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mistral_api_key: str = ''
    mistral_model: str = 'pixtral-large-latest'

    local_confidence_threshold: float = 0.72
    ocr_fast_mode: bool = True
    debug_overlay_enabled: bool = True
    auto_rotate_enabled: bool = True

    database_url: str = ''

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')


settings = Settings()
