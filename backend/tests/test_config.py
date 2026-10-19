"""Tests for configuration, especially the fail-closed local-inference guards.

The security-relevant property: RedactIt refuses to start with a configuration
that could route document text off this machine.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import BACKEND_DIR, Settings, get_settings


def make(**overrides) -> Settings:
    """Settings that never read a real .env or the ambient environment."""
    return Settings(_env_file=None, **overrides)


class TestLocalOnlyInference:
    def test_defaults_point_at_loopback(self) -> None:
        settings = make()
        assert settings.llm_base_url == "http://127.0.0.1:8080/v1"
        assert settings.llm_allow_cloud_fallback is False
        assert settings.llm_allow_non_loopback is False

    def test_cloud_fallback_is_refused_at_startup(self) -> None:
        with pytest.raises(ValidationError) as info:
            make(llm_allow_cloud_fallback=True)
        assert "LLM_ALLOW_CLOUD_FALLBACK=true is not supported" in str(info.value)
        assert "never sends document contents to a cloud provider" in str(info.value)

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.openai.com/v1",
            "https://generativelanguage.googleapis.com/v1beta",
            "http://10.0.0.5:8080/v1",
            "http://192.168.1.20:8080/v1",
            "http://model.internal.example.com/v1",
        ],
    )
    def test_a_non_loopback_model_host_is_refused(self, url: str) -> None:
        with pytest.raises(ValidationError) as info:
            make(llm_base_url=url)
        assert "not a loopback address" in str(info.value)

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8080/v1",
            "http://localhost:8080/v1",
            "http://[::1]:8080/v1",
            "https://127.0.0.1:8443/v1",
        ],
    )
    def test_loopback_hosts_are_accepted(self, url: str) -> None:
        assert make(llm_base_url=url).llm_base_url == url

    def test_a_non_loopback_host_needs_an_explicit_opt_in(self) -> None:
        settings = make(llm_base_url="http://192.168.1.20:8080/v1", llm_allow_non_loopback=True)
        assert settings.llm_allow_non_loopback is True

    @pytest.mark.parametrize("url", ["ftp://127.0.0.1/v1", "file:///tmp/x", "127.0.0.1:8080"])
    def test_the_scheme_must_be_http_or_https(self, url: str) -> None:
        with pytest.raises(ValidationError):
            make(llm_base_url=url)

    def test_a_url_with_no_host_is_refused(self) -> None:
        with pytest.raises(ValidationError) as info:
            make(llm_base_url="http:///v1")
        assert "LLM_BASE_URL" in str(info.value)


class TestDerivedUrls:
    def test_chat_and_model_urls_append_to_the_base(self) -> None:
        settings = make()
        assert settings.chat_completions_url == "http://127.0.0.1:8080/v1/chat/completions"
        assert settings.models_url == "http://127.0.0.1:8080/v1/models"

    def test_the_health_url_is_outside_the_v1_prefix(self) -> None:
        # llama-server serves /health at the server root, not under /v1.
        assert make().health_url == "http://127.0.0.1:8080/health"

    def test_a_trailing_slash_does_not_produce_a_double_slash(self) -> None:
        settings = make(llm_base_url="http://127.0.0.1:8080/v1/")
        assert settings.chat_completions_url == "http://127.0.0.1:8080/v1/chat/completions"
        assert settings.health_url == "http://127.0.0.1:8080/health"

    def test_a_base_without_v1_still_yields_a_health_url(self) -> None:
        assert make(llm_base_url="http://127.0.0.1:9090").health_url == "http://127.0.0.1:9090/health"


class TestSecrets:
    def test_the_api_key_is_not_rendered_by_default(self) -> None:
        settings = make(llm_api_key="sk-local-secret-value")
        assert "sk-local-secret-value" not in repr(settings)
        assert "sk-local-secret-value" not in str(settings.model_dump())
        assert settings.llm_api_key.get_secret_value() == "sk-local-secret-value"

    def test_no_api_key_is_the_default(self) -> None:
        assert make().llm_api_key is None


class TestExtensions:
    def test_extensions_are_lowercased_and_dotted(self) -> None:
        assert make(allowed_extensions=["PDF", "docx"]).allowed_extensions == {".pdf", ".docx"}

    def test_a_csv_string_is_split(self) -> None:
        assert make(allowed_extensions=".pdf, .docx").allowed_extensions == {".pdf", ".docx"}

    def test_a_json_array_string_is_accepted(self) -> None:
        assert make(allowed_extensions='[".pdf", ".docx"]').allowed_extensions == {".pdf", ".docx"}

    def test_a_malformed_json_array_falls_back_to_a_csv_split(self) -> None:
        assert make(cors_allow_origins="[http://localhost:3000").cors_allow_origins == [
            "[http://localhost:3000"
        ]

    def test_an_empty_extension_list_is_refused(self) -> None:
        with pytest.raises(ValidationError) as info:
            make(allowed_extensions=[])
        assert "must not be empty" in str(info.value)


class TestOtherNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("/api/v1", "/api/v1"), ("api/v1", "/api/v1"), ("/api/v1/", "/api/v1"), ("  /api/v1  ", "/api/v1")],
    )
    def test_api_prefix_is_normalised(self, raw: str, expected: str) -> None:
        assert make(api_prefix=raw).api_prefix == expected

    def test_cors_origins_accept_a_csv_string(self) -> None:
        settings = make(cors_allow_origins="http://localhost:3000, http://127.0.0.1:5173")
        assert settings.cors_allow_origins == ["http://localhost:3000", "http://127.0.0.1:5173"]

    def test_a_blank_replacement_text_is_refused(self) -> None:
        # An empty replacement would delete text without leaving any visible mark.
        with pytest.raises(ValidationError) as info:
            make(redaction_replacement_text="   ")
        assert "must not be blank" in str(info.value)

    def test_a_total_upload_cap_below_the_per_file_cap_is_refused(self) -> None:
        with pytest.raises(ValidationError) as info:
            make(max_file_size_bytes=2048, max_total_upload_bytes=1024)
        assert "MAX_TOTAL_UPLOAD_BYTES must be at least MAX_FILE_SIZE_BYTES" in str(info.value)

    def test_a_relative_storage_root_resolves_inside_the_backend(self) -> None:
        assert make(storage_root="var").resolved_storage_root == BACKEND_DIR / "var"

    def test_an_absolute_storage_root_is_used_as_is(self, tmp_path) -> None:
        assert make(storage_root=tmp_path).resolved_storage_root == tmp_path


class TestBounds:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("port", 0),
            ("port", 70000),
            ("session_ttl_minutes", 0),
            ("download_ttl_minutes", 0),
            ("cleanup_interval_seconds", 1),
            ("max_upload_files", 0),
            ("max_file_size_bytes", 10),
            ("llm_timeout_seconds", 0),
            ("llm_connect_timeout_seconds", 0),
            ("llm_max_tokens", 1),
            ("llm_temperature", 3.0),
            ("llm_max_attempts", 0),
            ("max_extracted_chars", 10),
            ("max_instruction_chars", 4),
        ],
    )
    def test_out_of_range_values_are_refused(self, field: str, value) -> None:
        with pytest.raises(ValidationError):
            make(**{field: value})

    def test_interpretation_mode_is_a_closed_choice(self) -> None:
        assert make(interpretation_mode="deterministic").interpretation_mode == "deterministic"
        with pytest.raises(ValidationError):
            make(interpretation_mode="cloud")

    def test_environment_is_a_closed_choice(self) -> None:
        with pytest.raises(ValidationError):
            make(environment="staging")

    def test_defaults_are_usable_on_a_small_machine(self) -> None:
        settings = make()
        assert settings.llm_max_prompt_chars <= 200_000
        assert settings.max_extracted_chars <= 5_000_000
        assert settings.llm_temperature == 0.0


class TestSettingsLoader:
    def test_get_settings_is_cached(self) -> None:
        assert get_settings() is get_settings()

    def test_unknown_environment_variables_are_ignored(self) -> None:
        assert make(SOMETHING_ELSE="x").app_name == "RedactIt API"
