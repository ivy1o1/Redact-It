"""Local model integration: llama.cpp adapter, instruction parsing, interpretation."""

from __future__ import annotations

from ._loopback import assert_local_target, host_is_local
from .adapter import LlamaCppClient, ModelHealth, extract_json_object
from .instruction_parser import ParsedInstruction, exclusion_variants, parse_instruction
from .interpreter import Interpretation, build_plan_schema, interpret_instruction

__all__ = [
    "Interpretation",
    "LlamaCppClient",
    "ModelHealth",
    "ParsedInstruction",
    "assert_local_target",
    "build_plan_schema",
    "exclusion_variants",
    "extract_json_object",
    "host_is_local",
    "interpret_instruction",
    "parse_instruction",
]
