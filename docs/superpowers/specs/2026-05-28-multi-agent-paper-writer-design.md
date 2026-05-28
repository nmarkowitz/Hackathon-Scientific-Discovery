# Multi-Agent Paper Writer — Design Spec

**Date:** 2026-05-28  
**Team:** paper-pushers  
**Status:** Approved

---

## Overview

A writing-layer package (`agents/paper-pushers/writing/`) composed of nine specialized agents: six section writers (abstract, intro, method, results, discussion, references), one figure generator, and two independent peer reviewers. The layer is built as standalone importable modules with a clean `ResearchBrief` input interface. It is not wired into `run()` yet — the future data-gathering and synthesis agents will populate a `ResearchBrief` and pass it in.

---

## Module Layout

```
agents/paper-pushers/writing/
  __init__.py
  brief.py          # ResearchBrief dataclass — input contract
  base.py           # Shared call_llm wrapper + system prompt scaffolding
  sections.py       # 6 section-writer agents
  figures.py        # fig-generator agent
  editors.py        # 2 reviewer agents + feedback merge + revision pass
  orchestrator.py   # Topo scheduler, dependency graph, assembly -> Paper

tests/
  test_writing.py   # Integration test harness with a sample ResearchBrief
```

---

## Input Contract — ResearchBrief

`ResearchBrief` is the seam between this writing layer and the future data/synthesis agents.

```python
@dataclass
class ResearchBrief:
    problem_domain: str          # the research area prompt
    findings: str = ""           # synthesized narrative of what was discovered
    data: dict = field(...)      # named arrays/tables for fig-generator to plot
    figure_specs: list = field(...)  # optional hints: what figures to make
    citations: list = field(...)     # source records for references agent
    notes: str = ""              # any extra context
```

The future data and synthesis agents own populating this. The test harness hand-writes a sample brief for now.

---

## Agent Interface

Each agent is a function:

```python
def write_<section>(brief: ResearchBrief, deps: dict[str, str]) -> str
```

- `deps` holds upstream section text keyed by section name (e.g. `{"method": "...", "results": "..."}`)
- Returns the section as a markdown string
- A small `base.py` helper wraps `call_llm`, injects a per-agent system prompt, and handles the response extraction

**Model used throughout:** `global.anthropic.claude-sonnet-4-6` (Bedrock inference profile — no `-v1:0` suffix).

---

## Dependency Graph

```
method        ← brief only
references    ← brief only
results       ← brief, method
figures       ← brief, results
discussion    ← brief, results, figures
intro         ← brief, method, results
abstract      ← brief, intro, discussion, results

editor1       ← full assembled draft  (independent peer reviewer)
editor2       ← full assembled draft  (independent peer reviewer)
```

---

## Orchestration

`orchestrator.py` runs the following pipeline:

1. **Topo scheduler** — resolves the DAG; dispatches agents via `ThreadPoolExecutor` as soon as their dependencies complete. `method` and `references` fire immediately in parallel; remaining agents unlock wave by wave.
2. **Assembly** — once all 7 section/figure agents finish, assembles a draft `Paper`.
3. **Review** — runs `editor1` and `editor2` in parallel against the full draft. Each returns a structured critique: list of `{section, issue, suggestion}` dicts.
4. **Merge** — combines both review lists and de-duplicates.
5. **Revision pass** — one LLM call per section to apply the merged review notes.
6. **Final assembly** — returns the revised `Paper` object.

---

## Fig-Generator

`figures.py` owns the full figure lifecycle:

1. Reads the results section text to decide what figures the paper needs.
2. Generates matplotlib plotting code from that decision.
3. Executes it via `run_code`.
4. Saves PNGs, converts to base64 markdown via `image_to_base64`.
5. Returns an embedded-image string appended to the results section.

**Fallback:** if `run_code` fails or produces no PNG, fig-generator returns an empty string. Results and discussion continue without embedded figures.

---

## Error Handling

- Each agent call is wrapped: on LLM error, `run_code` timeout, or empty response, it returns an empty string for that section rather than raising.
- The orchestrator logs failures but completes with whatever sections succeeded — partial paper beats a crash.
- The revision pass skips sections where the section text is empty.

---

## Test Harness

`tests/test_writing.py` — integration tests (real Bedrock calls, not mocked):

| Test | Assertion |
|---|---|
| Each section agent individually | Returns non-empty string given sample brief + deps |
| Full orchestrator end-to-end | Returns `Paper` with non-empty `title`, `introduction`, `methods`, `results` |
| Editor feedback merge | Merged list contains items from both reviewers |

Sample brief: `problem_domain = "effect of caffeine on reaction time"`, handwritten findings + one data array.

---

## Out of Scope (this spec)

- Wiring into `run()` — deferred to data-layer integration
- Data gathering, web search, or synthesis agents — future layer
- UI or publishing — platform concern
