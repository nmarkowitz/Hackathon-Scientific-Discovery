# Multi-Agent Paper Writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a writing-layer package under `agents/paper-pushers/writing/` with nine specialized agents (6 section writers, fig-generator, 2 peer reviewers) orchestrated via a topological scheduler with parallel execution.

**Architecture:** A `ResearchBrief` dataclass serves as the input contract for the future data layer. Nine agents run in dependency order via `ThreadPoolExecutor` — independent agents (method, references) fire immediately in parallel, downstream agents unlock as deps complete. Two independent peer reviewers critique the assembled draft, their feedback is merged and applied per-section.

**Tech Stack:** Python 3.11, `hackathon_science.utils.call_llm` (AWS Bedrock via `global.anthropic.claude-sonnet-4-6`), `hackathon_science.tools.run_code` + `image_to_base64`, `concurrent.futures.ThreadPoolExecutor`, `pytest`.

---

## File Map

| File | Responsibility |
|---|---|
| `agents/paper-pushers/writing/__init__.py` | Package exports |
| `agents/paper-pushers/writing/brief.py` | `ResearchBrief` dataclass — input contract |
| `agents/paper-pushers/writing/base.py` | `llm_call(system, user) -> str` — thin wrapper over `call_llm` |
| `agents/paper-pushers/writing/sections.py` | 6 section writers: method, references, results, intro, discussion, abstract |
| `agents/paper-pushers/writing/figures.py` | `generate_figures(brief, deps) -> str` — specs, runs, embeds figures |
| `agents/paper-pushers/writing/editors.py` | `review_paper`, `merge_reviews`, `apply_revisions` |
| `agents/paper-pushers/writing/orchestrator.py` | Topo scheduler, assembly, editor pass → `Paper` |
| `tests/conftest.py` | Adds `agents/paper-pushers` to sys.path |
| `tests/test_writing.py` | Integration tests (real Bedrock calls, `@pytest.mark.integration`) |

---

## Task 1: Test infrastructure + ResearchBrief

**Files:**
- Create: `tests/conftest.py`
- Create: `agents/paper-pushers/writing/__init__.py`
- Create: `agents/paper-pushers/writing/brief.py`
- Test: `tests/test_writing.py` (partial)

- [ ] **Step 1: Create the test file with the ResearchBrief test**

Create `tests/test_writing.py`:

```python
import pytest
from writing.brief import ResearchBrief


def test_research_brief_defaults():
    brief = ResearchBrief(problem_domain="effect of caffeine on reaction time")
    assert brief.problem_domain == "effect of caffeine on reaction time"
    assert brief.findings == ""
    assert brief.data == {}
    assert brief.figure_specs == []
    assert brief.citations == []
    assert brief.notes == ""


def test_research_brief_full():
    brief = ResearchBrief(
        problem_domain="caffeine study",
        findings="Caffeine reduces reaction time by 15%.",
        data={"reaction_times_ms": [320, 290, 305, 270, 315]},
        figure_specs=[{"type": "bar", "title": "Reaction time by dose"}],
        citations=[{"title": "Smith 2020", "url": "https://example.com"}],
        notes="Preliminary findings only.",
    )
    assert brief.data["reaction_times_ms"] == [320, 290, 305, 270, 315]
    assert len(brief.citations) == 1
```

- [ ] **Step 2: Run to verify it fails (writing module not found)**

```bash
uv run pytest tests/test_writing.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'writing'`

- [ ] **Step 3: Create conftest.py to add agents/paper-pushers to sys.path**

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agents" / "paper-pushers"))
```

- [ ] **Step 4: Run again — still fails (brief.py not found)**

```bash
uv run pytest tests/test_writing.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'writing'` (package dir doesn't exist yet)

- [ ] **Step 5: Create the writing package and brief.py**

Create `agents/paper-pushers/writing/__init__.py` (empty file):

```python
```

Create `agents/paper-pushers/writing/brief.py`:

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchBrief:
    problem_domain: str
    findings: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    figure_specs: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    notes: str = ""
```

- [ ] **Step 6: Run to verify tests pass**

```bash
uv run pytest tests/test_writing.py::test_research_brief_defaults tests/test_writing.py::test_research_brief_full -v
```

Expected:
```
PASSED tests/test_writing.py::test_research_brief_defaults
PASSED tests/test_writing.py::test_research_brief_full
```

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py tests/test_writing.py agents/paper-pushers/writing/__init__.py agents/paper-pushers/writing/brief.py
git commit -m "feat: add ResearchBrief dataclass and test infrastructure"
```

---

## Task 2: LLM base helper

**Files:**
- Create: `agents/paper-pushers/writing/base.py`
- Modify: `tests/test_writing.py`

- [ ] **Step 1: Add failing test for base helper**

Append to `tests/test_writing.py`:

```python
from unittest.mock import patch, MagicMock
from writing.base import llm_call, MODEL_ID


def test_llm_call_extracts_text():
    mock_response = {
        "output": {
            "message": {
                "content": [{"text": "This is the section text."}]
            }
        }
    }
    with patch("writing.base.call_llm", return_value=mock_response) as mock:
        result = llm_call("You are a writer.", "Write a methods section.")
    assert result == "This is the section text."
    mock.assert_called_once()


def test_llm_call_returns_empty_on_error():
    with patch("writing.base.call_llm", side_effect=Exception("Bedrock down")):
        result = llm_call("system", "user")
    assert result == ""


def test_model_id_is_correct_inference_profile():
    assert MODEL_ID == "global.anthropic.claude-sonnet-4-6"
    assert not MODEL_ID.endswith("v1:0")
```

- [ ] **Step 2: Run to verify tests fail**

```bash
uv run pytest tests/test_writing.py::test_llm_call_extracts_text tests/test_writing.py::test_llm_call_returns_empty_on_error tests/test_writing.py::test_model_id_is_correct_inference_profile -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'writing.base'`

- [ ] **Step 3: Implement base.py**

Create `agents/paper-pushers/writing/base.py`:

```python
from hackathon_science.utils import call_llm as _call_llm

MODEL_ID = "global.anthropic.claude-sonnet-4-6"


def llm_call(system: str, user: str) -> str:
    messages = [{"role": "user", "content": [{"text": user}]}]
    try:
        response = _call_llm(
            messages=messages,
            model_id=MODEL_ID,
            system=[{"text": system}],
        )
        content = response.get("output", {}).get("message", {}).get("content", [])
        return content[0].get("text", "") if content else ""
    except Exception:
        return ""
```

- [ ] **Step 4: Run to verify tests pass**

```bash
uv run pytest tests/test_writing.py::test_llm_call_extracts_text tests/test_writing.py::test_llm_call_returns_empty_on_error tests/test_writing.py::test_model_id_is_correct_inference_profile -v
```

Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/paper-pushers/writing/base.py tests/test_writing.py
git commit -m "feat: add LLM base helper with correct Bedrock inference profile ID"
```

---

## Task 3: Section writers

**Files:**
- Create: `agents/paper-pushers/writing/sections.py`
- Modify: `tests/test_writing.py`

- [ ] **Step 1: Add failing integration tests for section writers**

Append to `tests/test_writing.py`:

```python
import pytest
from writing.sections import (
    write_method, write_references, write_results,
    write_intro, write_discussion, write_abstract,
)


SAMPLE_BRIEF = ResearchBrief(
    problem_domain="effect of caffeine on human reaction time",
    findings="Participants given 200mg caffeine showed 15% faster reaction times vs placebo (n=40, p<0.01).",
    data={"reaction_times_ms": {"caffeine": [290, 305, 270, 315], "placebo": [340, 355, 320, 360]}},
    citations=[
        {"title": "Caffeine and Cognition (Smith 2020)", "journal": "J. Neuroscience", "year": 2020},
        {"title": "Stimulant Effects Review (Jones 2019)", "journal": "Pharmacology", "year": 2019},
    ],
    notes="Double-blind RCT. All participants healthy adults aged 18-35.",
)


@pytest.mark.integration
def test_write_method_returns_nonempty():
    result = write_method(SAMPLE_BRIEF, {})
    assert isinstance(result, str)
    assert len(result) > 100


@pytest.mark.integration
def test_write_references_returns_nonempty():
    result = write_references(SAMPLE_BRIEF, {})
    assert isinstance(result, str)
    assert len(result) > 50


@pytest.mark.integration
def test_write_results_uses_method_dep():
    method = write_method(SAMPLE_BRIEF, {})
    result = write_results(SAMPLE_BRIEF, {"method": method})
    assert isinstance(result, str)
    assert len(result) > 100


@pytest.mark.integration
def test_write_intro_returns_nonempty():
    method = "Participants were randomized to 200mg caffeine or placebo in a double-blind design."
    results = "Caffeine reduced reaction time by 15% (p<0.01)."
    result = write_intro(SAMPLE_BRIEF, {"method": method, "results": results})
    assert isinstance(result, str)
    assert len(result) > 100


@pytest.mark.integration
def test_write_discussion_returns_nonempty():
    results = "Caffeine reduced reaction time by 15% (p<0.01)."
    result = write_discussion(SAMPLE_BRIEF, {"results": results, "figures": ""})
    assert isinstance(result, str)
    assert len(result) > 100


@pytest.mark.integration
def test_write_abstract_returns_nonempty():
    intro = "Caffeine is a widely consumed stimulant."
    discussion = "These findings support caffeine's role in enhancing cognitive performance."
    results = "Caffeine reduced reaction time by 15% (p<0.01)."
    result = write_abstract(SAMPLE_BRIEF, {"intro": intro, "discussion": discussion, "results": results})
    assert isinstance(result, str)
    assert len(result) > 50
```

- [ ] **Step 2: Run to verify tests fail**

```bash
uv run pytest tests/test_writing.py -m integration -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'writing.sections'`

- [ ] **Step 3: Implement sections.py**

Create `agents/paper-pushers/writing/sections.py`:

```python
from .brief import ResearchBrief
from .base import llm_call


def write_method(brief: ResearchBrief, deps: dict[str, str]) -> str:
    system = "You are a scientific methods writer. Write rigorous, reproducible methods sections."
    user = f"""Write the Methods section for a scientific paper.

Problem domain: {brief.problem_domain}
Findings summary: {brief.findings}
Study notes: {brief.notes}

Write 2-4 paragraphs in formal academic style covering study design, participants/materials, procedure, and analysis."""
    return llm_call(system, user)


def write_references(brief: ResearchBrief, deps: dict[str, str]) -> str:
    citations_text = "\n".join(
        f"- {c.get('title', '')} ({c.get('year', '')}). {c.get('journal', '')}."
        for c in brief.citations
    ) if brief.citations else "No citations provided."

    system = "You are a scientific reference formatter. Format references in APA style."
    user = f"""Format the following sources as a References section in APA style.

Sources:
{citations_text}

Problem domain (for context): {brief.problem_domain}

Return a numbered reference list."""
    return llm_call(system, user)


def write_results(brief: ResearchBrief, deps: dict[str, str]) -> str:
    method = deps.get("method", "")
    system = "You are a scientific results writer. Present findings clearly and objectively."
    user = f"""Write the Results section for a scientific paper.

Problem domain: {brief.problem_domain}
Findings: {brief.findings}
Data: {brief.data}
Methods used: {method[:500] if method else 'Not provided'}

Write 2-4 paragraphs presenting the findings objectively. Reference specific numbers and statistics from the data."""
    return llm_call(system, user)


def write_intro(brief: ResearchBrief, deps: dict[str, str]) -> str:
    method = deps.get("method", "")
    results = deps.get("results", "")
    system = "You are a scientific introduction writer. Establish context, gap, and contribution."
    user = f"""Write the Introduction section for a scientific paper.

Problem domain: {brief.problem_domain}
Methods overview: {method[:300] if method else 'Not provided'}
Key findings: {results[:300] if results else brief.findings}

Write 3-5 paragraphs: (1) background and motivation, (2) gap in existing knowledge, (3) this paper's approach and contribution."""
    return llm_call(system, user)


def write_discussion(brief: ResearchBrief, deps: dict[str, str]) -> str:
    results = deps.get("results", "")
    figures = deps.get("figures", "")
    system = "You are a scientific discussion writer. Interpret findings and situate them in the literature."
    user = f"""Write the Discussion section for a scientific paper.

Problem domain: {brief.problem_domain}
Results: {results[:600] if results else brief.findings}
Figures context: {'Figures were generated.' if figures else 'No figures.'}
Notes: {brief.notes}

Write 3-5 paragraphs: (1) interpretation of key findings, (2) limitations, (3) implications and future directions."""
    return llm_call(system, user)


def write_abstract(brief: ResearchBrief, deps: dict[str, str]) -> str:
    intro = deps.get("intro", "")
    discussion = deps.get("discussion", "")
    results = deps.get("results", "")
    system = "You are a scientific abstract writer. Write concise, structured abstracts."
    user = f"""Write the Abstract for a scientific paper (150-250 words).

Problem domain: {brief.problem_domain}
Introduction excerpt: {intro[:300] if intro else ''}
Key results: {results[:300] if results else brief.findings}
Discussion excerpt: {discussion[:300] if discussion else ''}

Structure: background (1-2 sentences), objective (1 sentence), methods (1-2 sentences), results (2-3 sentences), conclusion (1-2 sentences)."""
    return llm_call(system, user)
```

- [ ] **Step 4: Run integration tests**

```bash
uv run pytest tests/test_writing.py -m integration -v
```

Expected: all 6 section tests PASSED (requires valid AWS Bedrock credentials)

- [ ] **Step 5: Commit**

```bash
git add agents/paper-pushers/writing/sections.py tests/test_writing.py
git commit -m "feat: add six section writer agents"
```

---

## Task 4: Figure generator

**Files:**
- Create: `agents/paper-pushers/writing/figures.py`
- Modify: `tests/test_writing.py`

- [ ] **Step 1: Add failing integration test for figures**

Append to `tests/test_writing.py`:

```python
from writing.figures import generate_figures


@pytest.mark.integration
def test_generate_figures_returns_string():
    results = "Caffeine group: mean 293ms (SD=19). Placebo group: mean 344ms (SD=22). t(38)=8.3, p<0.001."
    result = generate_figures(SAMPLE_BRIEF, {"results": results})
    assert isinstance(result, str)


@pytest.mark.integration
def test_generate_figures_embeds_image_or_empty():
    results = "Caffeine reduced reaction time by 15% vs placebo."
    result = generate_figures(SAMPLE_BRIEF, {"results": results})
    # Either embedded base64 image or empty string (if run_code/matplotlib fails)
    assert result == "" or result.startswith("![")
```

- [ ] **Step 2: Run to verify tests fail**

```bash
uv run pytest tests/test_writing.py::test_generate_figures_returns_string tests/test_writing.py::test_generate_figures_embeds_image_or_empty -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'writing.figures'`

- [ ] **Step 3: Implement figures.py**

Create `agents/paper-pushers/writing/figures.py`:

```python
import tempfile
from pathlib import Path

from hackathon_science.tools import run_code, image_to_base64
from .brief import ResearchBrief
from .base import llm_call


def generate_figures(brief: ResearchBrief, deps: dict[str, str]) -> str:
    results_text = deps.get("results", "")
    tmpdir = tempfile.mkdtemp()

    system = "You are a scientific figure generator. Write Python matplotlib code to visualize results."
    user = f"""Write Python code to generate 1-3 figures for a scientific paper.

Results section:
{results_text}

Available data:
{brief.data}

Figure hints:
{brief.figure_specs}

Requirements:
- Save each figure as an absolute path: '{tmpdir}/figure_1.png', '{tmpdir}/figure_2.png', etc.
- Use: plt.savefig('ABSOLUTE_PATH', dpi=150, bbox_inches='tight')
- Call plt.close() after each figure
- Import matplotlib.pyplot as plt, numpy as np, and anything else needed
- Generate at most 3 figures

Return ONLY Python code, no explanation or markdown."""
    code = llm_call(system, user)
    if not code:
        return ""

    run_code(code)

    figures_md = []
    for i in range(1, 10):
        png_path = Path(tmpdir) / f"figure_{i}.png"
        if not png_path.exists():
            break
        try:
            md = image_to_base64(str(png_path), f"Figure {i}")
            figures_md.append(md)
        except Exception:
            break

    return "\n\n".join(figures_md)
```

- [ ] **Step 4: Run integration tests**

```bash
uv run pytest tests/test_writing.py::test_generate_figures_returns_string tests/test_writing.py::test_generate_figures_embeds_image_or_empty -v
```

Expected: both tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/paper-pushers/writing/figures.py tests/test_writing.py
git commit -m "feat: add figure generator agent"
```

---

## Task 5: Editors (peer review + revision)

**Files:**
- Create: `agents/paper-pushers/writing/editors.py`
- Modify: `tests/test_writing.py`

- [ ] **Step 1: Add failing tests for editors**

Append to `tests/test_writing.py`:

```python
from writing.editors import review_paper, merge_reviews, apply_revisions


def test_merge_reviews_deduplicates():
    r1 = [
        {"section": "results", "issue": "No error bars shown", "suggestion": "Add SD or SE to all plots."},
        {"section": "intro", "issue": "Missing motivation", "suggestion": "Add a sentence on why this matters."},
    ]
    r2 = [
        {"section": "results", "issue": "No error bars shown", "suggestion": "Add error bars."},
        {"section": "discussion", "issue": "No limitations", "suggestion": "Add a limitations paragraph."},
    ]
    merged = merge_reviews(r1, r2)
    sections = [item["section"] for item in merged]
    assert "results" in sections
    assert "intro" in sections
    assert "discussion" in sections
    # Exact duplicate (same section + same first 50 chars of issue) is removed
    results_items = [item for item in merged if item["section"] == "results"]
    assert len(results_items) == 1


@pytest.mark.integration
def test_review_paper_returns_list():
    draft = {
        "intro": "Caffeine is a stimulant consumed worldwide.",
        "method": "Participants were randomized to caffeine or placebo.",
        "results": "Caffeine reduced reaction time by 15% (p<0.01).",
        "discussion": "Results support prior literature.",
        "abstract": "We studied caffeine effects on reaction time.",
        "references": "1. Smith 2020. J. Neuroscience.",
    }
    reviews = review_paper("Reviewer 1", draft)
    assert isinstance(reviews, list)


@pytest.mark.integration
def test_apply_revisions_returns_nonempty_sections():
    sections = {
        "results": "Caffeine reduced reaction time by 15%.",
        "intro": "Caffeine is a stimulant.",
    }
    feedback = [
        {"section": "results", "issue": "No statistics", "suggestion": "Add p-value and sample size."},
    ]
    revised = apply_revisions(sections, feedback)
    assert "results" in revised
    assert "intro" in revised
    assert isinstance(revised["results"], str)
    assert len(revised["results"]) > 0
```

- [ ] **Step 2: Run to verify tests fail**

```bash
uv run pytest tests/test_writing.py::test_merge_reviews_deduplicates -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'writing.editors'`

- [ ] **Step 3: Implement editors.py**

Create `agents/paper-pushers/writing/editors.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_writing.py::test_merge_reviews_deduplicates -v
uv run pytest tests/test_writing.py::test_review_paper_returns_list tests/test_writing.py::test_apply_revisions_returns_nonempty_sections -m integration -v
```

Expected: all 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/paper-pushers/writing/editors.py tests/test_writing.py
git commit -m "feat: add peer review editors with merge and revision pass"
```

---

## Task 6: Orchestrator

**Files:**
- Create: `agents/paper-pushers/writing/orchestrator.py`
- Modify: `agents/paper-pushers/writing/__init__.py`
- Modify: `tests/test_writing.py`

- [ ] **Step 1: Add failing end-to-end integration test**

Append to `tests/test_writing.py`:

```python
from writing.orchestrator import orchestrate
from hackathon_science.models import Paper


@pytest.mark.integration
def test_orchestrate_returns_valid_paper():
    result = orchestrate(SAMPLE_BRIEF)
    assert isinstance(result, Paper)
    assert len(result.title) > 0
    assert len(result.introduction) > 0
    assert len(result.methods) > 0
    assert len(result.results) > 0


@pytest.mark.integration
def test_orchestrate_with_minimal_brief():
    brief = ResearchBrief(problem_domain="effect of sleep deprivation on memory recall")
    result = orchestrate(brief)
    assert isinstance(result, Paper)
    assert len(result.title) > 0
```

- [ ] **Step 2: Run to verify tests fail**

```bash
uv run pytest tests/test_writing.py::test_orchestrate_returns_valid_paper -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'writing.orchestrator'`

- [ ] **Step 3: Implement orchestrator.py**

Create `agents/paper-pushers/writing/orchestrator.py`:

```python
import threading
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED

from hackathon_science.models import Paper
from .brief import ResearchBrief
from .base import llm_call
from .sections import (
    write_method, write_references, write_results,
    write_intro, write_discussion, write_abstract,
)
from .figures import generate_figures
from .editors import review_paper, merge_reviews, apply_revisions


DEPENDENCIES: dict[str, list[str]] = {
    "method": [],
    "references": [],
    "results": ["method"],
    "figures": ["results"],
    "discussion": ["results", "figures"],
    "intro": ["method", "results"],
    "abstract": ["intro", "discussion", "results"],
}

_AGENTS = {
    "method": write_method,
    "references": write_references,
    "results": write_results,
    "figures": generate_figures,
    "discussion": write_discussion,
    "intro": write_intro,
    "abstract": write_abstract,
}


def orchestrate(brief: ResearchBrief) -> Paper:
    results: dict[str, str] = {}
    lock = threading.Lock()
    completed: set[str] = set()
    submitted: set[str] = set()

    def run_agent(name: str) -> None:
        with lock:
            deps = {k: results[k] for k in DEPENDENCIES[name]}
        try:
            text = _AGENTS[name](brief, deps)
        except Exception as e:
            print(f"[orchestrator] agent '{name}' failed: {e}")
            text = ""
        with lock:
            results[name] = text

    with ThreadPoolExecutor(max_workers=4) as executor:
        pending: dict[Future, str] = {}

        def submit_ready() -> None:
            for name in DEPENDENCIES:
                if name not in submitted and all(d in completed for d in DEPENDENCIES[name]):
                    f = executor.submit(run_agent, name)
                    pending[f] = name
                    submitted.add(name)

        submit_ready()
        while pending:
            done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
            for f in done:
                name = pending.pop(f)
                completed.add(name)
            submit_ready()

    figures_text = results.get("figures", "")
    sections = {
        "intro": results.get("intro", ""),
        "method": results.get("method", ""),
        "results": results.get("results", "") + ("\n\n" + figures_text if figures_text else ""),
        "discussion": results.get("discussion", ""),
        "abstract": results.get("abstract", ""),
        "references": results.get("references", ""),
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(review_paper, "Reviewer 1", sections)
        f2 = executor.submit(review_paper, "Reviewer 2", sections)
        reviews1 = f1.result()
        reviews2 = f2.result()

    feedback = merge_reviews(reviews1, reviews2)
    revised = apply_revisions(sections, feedback)

    title = _generate_title(brief, revised.get("abstract", ""))

    return Paper(
        title=title,
        introduction=revised.get("intro", ""),
        methods=revised.get("method", ""),
        results=revised.get("results", ""),
        references=revised.get("references", ""),
        appendix="",
        tags=[brief.problem_domain],
    )


def _generate_title(brief: ResearchBrief, abstract: str) -> str:
    system = "You are a scientific paper title writer."
    user = f"""Write a concise, specific title for this paper (10 words or fewer).

Problem domain: {brief.problem_domain}
Abstract: {abstract[:400] if abstract else ''}

Return only the title, no quotes, no explanation."""
    return llm_call(system, user) or brief.problem_domain
```

- [ ] **Step 4: Update writing/__init__.py to export orchestrate**

Replace contents of `agents/paper-pushers/writing/__init__.py`:

```python
from .brief import ResearchBrief
from .orchestrator import orchestrate

__all__ = ["ResearchBrief", "orchestrate"]
```

- [ ] **Step 5: Run end-to-end integration tests**

```bash
uv run pytest tests/test_writing.py::test_orchestrate_returns_valid_paper tests/test_writing.py::test_orchestrate_with_minimal_brief -m integration -v
```

Expected: both tests PASSED (this will take ~60-120s due to multiple LLM calls)

- [ ] **Step 6: Run full test suite to catch regressions**

```bash
uv run pytest tests/test_writing.py -v
```

Expected: all non-integration tests PASSED immediately; integration tests PASSED with valid credentials

- [ ] **Step 7: Commit**

```bash
git add agents/paper-pushers/writing/orchestrator.py agents/paper-pushers/writing/__init__.py tests/test_writing.py
git commit -m "feat: add topo-scheduled orchestrator with peer review pass"
```

---

## Running integration tests

Integration tests require valid AWS Bedrock credentials. Run them explicitly:

```bash
# Run only unit tests (fast, no credentials needed)
uv run pytest tests/test_writing.py -v -m "not integration"

# Run integration tests (requires AWS credentials)
uv run pytest tests/test_writing.py -v -m integration

# Run everything
uv run pytest tests/test_writing.py -v
```

To register the `integration` marker and suppress warnings, add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: marks tests that call real external APIs (requires AWS credentials)",
]
```
