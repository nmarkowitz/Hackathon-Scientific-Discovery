from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

MODEL_ID = "global.anthropic.claude-sonnet-4-6"


@dataclass
class FigurePlan:
    figure_id: str
    title: str
    caption: str
    figure_type: str
    message: str
    elements: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    style: dict[str, Any] = field(default_factory=dict)


@dataclass
class FigureArtifact:
    figure_id: str
    title: str
    caption: str
    path: str
    markdown: str
    critique: str = ""


def ask_llm(system: str, user: str) -> str:
    try:
        from hackathon_science.utils import call_llm
    except Exception:
        return ""

    messages = [{"role": "user", "content": [{"text": user}]}]
    try:
        response = call_llm(
            messages=messages,
            model_id=MODEL_ID,
            system=[{"text": system}],
        )
        content = response.get("output", {}).get("message", {}).get("content", [])
        return content[0].get("text", "") if content else ""
    except Exception:
        return ""


def extract_json(text: str, fallback: Any) -> Any:
    if not text:
        return fallback

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()

    try:
        return json.loads(stripped)
    except Exception:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start:end + 1])
            except Exception:
                pass

        start = stripped.find("[")
        end = stripped.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start:end + 1])
            except Exception:
                pass

    return fallback
