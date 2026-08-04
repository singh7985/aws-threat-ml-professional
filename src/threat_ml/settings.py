from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import EmailStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    project_name: str = "autonomous-cloud-threat-ml"
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    aws_profile: str = "threat-ml-dev"
    aws_region: str = "us-east-1"
    aws_default_region: str = "us-east-1"

    alert_email: EmailStr = "replace-me@example.com"
    raw_data_path: Path = Field(default=Path("data/raw/security_events.jsonl"))
    model_path: Path = Field(default=Path("models/model.joblib"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
