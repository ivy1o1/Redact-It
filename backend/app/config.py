"""Application settings.

All configuration comes from the environment (optionally a local ``.env``).
Nothing about the local model lives in source code.
"""

from __future__ import annotations

import ipaddress
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent

Environment = Literal["development", "testing", "production"]

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def _split_csv(value: Any) -> Any:
    """Accept both ``a,b,c`` and a JSON array for list-valued settings."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass  # not valid JSON after all; fall through to the CSV split
        return [item.strip() for item in text.split(",") if item.strip()]
    return value


def _is_local_host(host: str) -> bool:
    """True when host resolves to loopback or another non-routable local address."""
    if host in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=("model_config",),
    )

    # ------------------------------------------------------------------ app
    app_name: str = "RedactIt API"
    app_version: str = "0.1.0"
    environment: Environment = "development"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_prefix: str = "/api/v1"
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    # -------------------------------------------------------------- storage
    storage_root: Path = BACKEND_DIR / "var"
    session_ttl_minutes: int = Field(default=30, ge=1, le=1440)
    download_ttl_minutes: int = Field(default=15, ge=1, le=1440)
    cleanup_interval_seconds: int = Field(default=60, ge=5, le=3600)

    max_upload_files: int = Field(default=5, ge=1, le=50)
    max_file_size_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)
    max_total_upload_bytes: int = Field(default=60 * 1024 * 1024, ge=1024)
    allowed_extensions: set[str] = Field(default_factory=lambda: {".pdf", ".docx"})

    # Character budget per document for extraction. Extraction past this point is
    # skipped and reported as a truncation warning - an unbounded read of a 400
    # page PDF would not fit in an 8 GB laptop alongside the model.
    max_extracted_chars: int = Field(default=200_000, ge=1_000, le=5_000_000)
    max_instruction_chars: int = Field(default=2000, ge=32, le=20_000)

    # ------------------------------------------------------------ local LLM
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_model: str = "gemma-3-1b-it"
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=120.0, gt=0, le=900)
    llm_connect_timeout_seconds: float = Field(default=5.0, gt=0, le=120)
    llm_max_tokens: int = Field(default=700, ge=64, le=4096)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_allow_cloud_fallback: bool = False
    llm_allow_non_loopback: bool = False
    llm_use_json_schema: bool = True
    llm_max_attempts: int = Field(default=2, ge=1, le=4)
    llm_max_prompt_chars: int = Field(default=6000, ge=0, le=200_000)
    # 'deterministic' skips the model entirely (keyword parsing only). There is no
    # cloud option: the only supported backends are 'model' (local) and
    # 'deterministic' (no model).
    interpretation_mode: Literal["model", "deterministic"] = "model"

    # ------------------------------------------------------------ redaction
    redaction_replacement_text: str = Field(default="[REDACTED]", max_length=64)
    pdf_redact_images: bool = True

    # -------------------------------------------------------------- logging
    log_level: str = "INFO"
    log_json: bool = True

    _normalize_csv = field_validator("cors_allow_origins", "allowed_extensions", mode="before")(
        _split_csv
    )

    @field_validator("allowed_extensions", mode="after")
    @classmethod
    def _lowercase_extensions(cls, value: set[str]) -> set[str]:
        normalised = {ext.strip().lower() for ext in value if ext.strip()}
        if not normalised:
            raise ValueError("ALLOWED_EXTENSIONS must not be empty")
        return {ext if ext.startswith(".") else f".{ext}" for ext in normalised}

    @field_validator("redaction_replacement_text", mode="after")
    @classmethod
    def _replacement_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("REDACTION_REPLACEMENT_TEXT must not be blank")
        return value

    @field_validator("api_prefix", mode="after")
    @classmethod
    def _clean_prefix(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if cleaned and not cleaned.startswith("/"):
            cleaned = f"/{cleaned}"
        return cleaned

    @model_validator(mode="after")
    def _enforce_local_only_inference(self) -> Settings:
        """Fail closed: cloud fallback is not a supported configuration."""
        if self.llm_allow_cloud_fallback:
            raise ValueError(
                "LLM_ALLOW_CLOUD_FALLBACK=true is not supported. RedactIt never sends "
                "document contents to a cloud provider; set it to false and point "
                "LLM_BASE_URL at a local llama.cpp server."
            )

        parsed = urlparse(self.llm_base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("LLM_BASE_URL must be an http:// or https:// URL")
        hostname = parsed.hostname or ""
        if not hostname:
            raise ValueError("LLM_BASE_URL must include a host")
        if not self.llm_allow_non_loopback and not _is_local_host(hostname):
            raise ValueError(
                f"LLM_BASE_URL host {hostname!r} is not a loopback address. "
                "RedactIt only talks to a model server on this machine. Start "
                "llama-server on 127.0.0.1, or set LLM_ALLOW_NON_LOOPBACK=true if you "
                "deliberately run the server on another local interface."
            )
        if self.max_total_upload_bytes < self.max_file_size_bytes:
            raise ValueError(
                "MAX_TOTAL_UPLOAD_BYTES must be at least MAX_FILE_SIZE_BYTES"
            )
        return self

    @property
    def resolved_storage_root(self) -> Path:
        path = self.storage_root
        if not path.is_absolute():
            path = BACKEND_DIR / path
        return path

    @property
    def chat_completions_url(self) -> str:
        return f"{self.llm_base_url.rstrip('/')}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.llm_base_url.rstrip('/')}/models"

    @property
    def health_url(self) -> str:
        base = self.llm_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return f"{base}/health"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
