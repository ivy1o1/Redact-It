"""End-to-end workflow against the real local model.

This is the only module in the suite that talks to a running server, and it
talks to ``127.0.0.1`` only. It exists because a mocked adapter proves the
error mapping but not that Gemma's actual output survives validation.

Run it with the model loaded:
    .venv\\Scripts\\python.exe -m pytest -m integration
It is deselected from the default run (``-m "not integration"`` is not even
needed - ``addopts`` in pytest.ini keeps the suite offline) and skips itself if
no local server answers.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pymupdf
import pytest

from app.config import Settings
from app.llm.adapter import LlamaCppClient
from app.main import create_app
from fastapi.testclient import TestClient

from .conftest import INSTRUCTION, build_pdf, make_settings, upload

pytestmark = pytest.mark.integration

_SCRATCH = Path(tempfile.mkdtemp(prefix="redactit-live-"))


def live_settings() -> Settings:
    return make_settings(
        _SCRATCH,
        interpretation_mode="model",
        llm_timeout_seconds=180.0,
        llm_connect_timeout_seconds=3.0,
        llm_max_prompt_chars=1500,
    )


@pytest.fixture(scope="module")
def model_health():
    """Probe the server once. A port that answers is not proof the model loaded."""
    settings = live_settings()

    async def probe() -> object:
        # One event loop for both the request and the teardown: an httpx client
        # created in one asyncio.run() and closed in another raises
        # "Event loop is closed" on Windows.
        client = LlamaCppClient(settings)
        try:
            return await client.health()
        finally:
            await client.aclose()

    health = asyncio.run(probe())
    if not health.reachable:
        pytest.skip(f"no local model server at {settings.llm_base_url}: {health.detail}")
    if health.status != "ok":
        pytest.skip(f"local model server is {health.status}: {health.detail}")
    if settings.llm_model.lower() not in [m.lower() for m in health.models]:
        pytest.skip(
            f"server is up but does not report {settings.llm_model!r}; it lists {health.models}"
        )
    return health


@pytest.fixture
def live_client(model_health):
    app = create_app(live_settings())
    with TestClient(app) as test_client:
        yield test_client
        app.state.store.purge_all()


@pytest.fixture
def synthetic_pdf() -> bytes:
    return build_pdf()


def analyze(client: TestClient, instruction: str, name: str = "report.pdf", data: bytes = None):  # type: ignore[assignment]
    return client.post(
        "/api/v1/analyze",
        files=[upload(name, data or build_pdf())],
        data={"instruction": instruction},
    )


class TestServerReportsReady:
    def test_the_model_endpoint_confirms_the_expected_alias(self, live_client: TestClient) -> None:
        body = live_client.get("/api/v1/health/model").json()
        assert body["status"] == "ready"
        assert body["model"]["expected_model_loaded"] is True
        assert body["model"]["base_url"].startswith("http://127.0.0.1")
        assert body["interpretation_mode"] == "model"

    def test_the_probe_sends_no_document_content(self, live_client: TestClient) -> None:
        response = live_client.get("/api/v1/health/model")
        assert response.status_code == 200
        assert INSTRUCTION not in response.text


class TestAnalysisUsesTheModel:
    def test_a_natural_instruction_produces_a_validated_plan(self, live_client: TestClient) -> None:
        response = analyze(live_client, INSTRUCTION)
        assert response.status_code == 200, response.text
        body = response.json()

        interpretation = body["interpretation"]
        assert interpretation["model_used"] is True
        assert interpretation["interpretation_mode"] == "model"
        assert interpretation["source"] == "model+deterministic"

        patterns = {t["pattern"] for t in interpretation["targets"] if t["kind"] == "pattern"}
        assert {"person_name", "phone", "address"} <= patterns

        assert body["summary"]["detections"] > 0
        matched = {d["matched_text"] for d in body["detections"]}
        assert "Alice Johnson" in matched

    def test_the_model_cannot_widen_redaction_into_kept_data(self, live_client: TestClient) -> None:
        body = analyze(live_client, INSTRUCTION).json()
        # Whatever the model proposed, the keep-clause protection is enforced by
        # the deterministic layer, so no grade or course value may be offered.
        assert not any("Algebra" in d["matched_text"] for d in body["detections"])
        assert not any(d["matched_text"].strip() == "A" for d in body["detections"])

    def test_every_target_is_registry_backed_or_literal(self, live_client: TestClient) -> None:
        body = analyze(live_client, INSTRUCTION).json()
        for target in body["interpretation"]["targets"]:
            if target["kind"] == "pattern":
                assert target["pattern"] in {
                    "person_name", "phone", "address", "email", "ssn", "date",
                    "credit_card", "zip_code", "ip_address", "url",
                }
            else:
                assert target["kind"] == "exact"

    def test_a_vague_instruction_is_answered_not_guessed(self, live_client: TestClient) -> None:
        response = analyze(live_client, "redact stuff")
        if response.status_code == 200:
            body = response.json()
            assert body["summary"]["detections"] >= 0
            assert body["interpretation"]["targets"]
        else:
            assert response.status_code == 422
            error = response.json()["error"]
            assert error["code"] == "INSTRUCTION_AMBIGUOUS"
            assert "supported_detectors" in error["details"]


class TestFullWorkflowWithTheModel:
    def test_analyse_review_redact_download(self, live_client: TestClient) -> None:
        body = analyze(live_client, INSTRUCTION).json()
        session_id = body["session_id"]
        name_to_ids = {d["detection_id"]: d["matched_text"] for d in body["detections"]}

        approved = [
            detection_id
            for detection_id, text in name_to_ids.items()
            if "Alice Johnson" in text or "555" in text or "Maple" in text
        ]
        assert approved, "the synthetic report should offer name/phone/address matches"

        review = live_client.post(
            f"/api/v1/sessions/{session_id}/review",
            json={"approved_detection_ids": approved, "rejected_detection_ids": []},
        )
        assert review.status_code == 200, review.text

        redact = live_client.post(f"/api/v1/sessions/{session_id}/redact")
        assert redact.status_code == 200, redact.text
        result = redact.json()
        assert result["outputs"]

        token = result["outputs"][0]["download_token"]
        download = live_client.get(f"/api/v1/downloads/{token}")
        assert download.status_code == 200
        assert download.headers.get("content-type", "").startswith("application/")

        with pymupdf.open(stream=download.content, filetype="pdf") as document:
            text = "".join(page.get_text() for page in document)
        assert "Alice Johnson" not in text
        assert "42 Maple Street" not in text
        # Kept data must still read clearly - redaction is not deletion of the file.
        assert "Algebra" in text

    def test_the_model_path_still_reports_its_own_limitations(self, live_client: TestClient) -> None:
        body = analyze(live_client, INSTRUCTION).json()
        # Honest hedging is required whether or not the model was consulted.
        assert body["limitations"]
        assert any("guarantee" in line.lower() for line in body["limitations"])
        assert isinstance(body["interpretation"]["rejected_model_proposals"], list)
