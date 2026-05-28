import json
import re
from .base import llm_call


def review_paper(reviewer_id: str, draft: dict[str, str]) -> list[dict]:
    paper_text = "\n\n".join(
        f"## {section.upper()}\n{text}"
        for section, text in draft.items()
        if text
    )
    system = f"You are {reviewer_id}, an expert scientific peer reviewer."
    user = f"""Review this scientific paper and return a JSON list of issues.

Each issue must have exactly these keys:
- "section": which section contains the issue (e.g. "results", "intro")
- "issue": a specific description of the problem
- "suggestion": a concrete suggestion to fix it

Paper:
{paper_text}

Return ONLY a JSON array. Example format:
[{{"section": "results", "issue": "Missing sample size", "suggestion": "State n= for each group."}}]"""

    response = llm_call(system, user)
    try:
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return []


def merge_reviews(reviews1: list[dict], reviews2: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    merged = []
    for item in reviews1 + reviews2:
        key = (item.get("section", ""), item.get("issue", "")[:50])
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def apply_revisions(sections: dict[str, str], feedback: list[dict]) -> dict[str, str]:
    revised = {}
    for section, text in sections.items():
        if not text:
            revised[section] = text
            continue
        section_notes = [f for f in feedback if f.get("section") == section]
        if not section_notes:
            revised[section] = text
            continue
        notes_text = "\n".join(
            f"- Issue: {f['issue']}\n  Fix: {f['suggestion']}"
            for f in section_notes
        )
        system = "You are a scientific paper editor. Revise the section to address reviewer feedback."
        user = f"""Revise this {section} section based on reviewer feedback.

Original text:
{text}

Reviewer feedback:
{notes_text}

Return the revised section text only, no commentary."""
        revised[section] = llm_call(system, user) or text
    return revised
