from pathlib import Path
from pydantic_settings import (BaseSettings,
                               SettingsConfigDict
                               )
from pydantic import SecretStr

current_file = Path(__file__).absolute()
current_dir = current_file.parent
env_path = current_dir / '.env'


class Settings(BaseSettings):
    OPEN_AI_KEY: SecretStr
    GIGA_CHAT_API_KEY: SecretStr
    BASE_URL: str
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8'
    )


settings = Settings()
