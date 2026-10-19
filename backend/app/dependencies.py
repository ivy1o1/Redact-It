"""FastAPI dependencies.

Everything resolves from ``app.state`` so the lifespan wiring is the only place
that constructs anything, and tests can override a single dependency.
"""

from __future__ import annotations

from fastapi import Request

from .config import Settings
from .llm.adapter import LlamaCppClient
from .services.analysis import AnalysisService
from .storage import SessionStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> SessionStore:
    return request.app.state.store


def get_service(request: Request) -> AnalysisService:
    return request.app.state.service


def get_llm_client(request: Request) -> LlamaCppClient | None:
    return getattr(request.app.state, "llm_client", None)


__all__ = ["get_llm_client", "get_service", "get_settings", "get_store"]
