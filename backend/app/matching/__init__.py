"""Deterministic matching engine and detector registry."""

from __future__ import annotations

from .engine import (
    Detection,
    DetectionLocation,
    Exclusion,
    MatchPlan,
    MatchResult,
    Target,
    detect,
    detection_id,
)
from .patterns import PATTERN_REGISTRY, PatternSpec, RegexRule, supported_pattern_names

__all__ = [
    "PATTERN_REGISTRY",
    "Detection",
    "DetectionLocation",
    "Exclusion",
    "MatchPlan",
    "MatchResult",
    "PatternSpec",
    "RegexRule",
    "Target",
    "detect",
    "detection_id",
    "supported_pattern_names",
]
