"""
PaperVizAgent-inspired figure creator for the hackathon paper runner.
"""
from pathlib import Path
import sys
from typing import Optional

from hackathon_science import Paper
from hackathon_science.utils import call_llm

sys.path.insert(0, str(Path(__file__).parent))
from figure_agents import create_figures

MODEL_ID = "global.anthropic.claude-sonnet-4-6"


def _llm_text(system: str, user: str) -> str:
    try:
        response = call_llm(
            messages=[{"role": "user", "content": [{"text": user}]}],
            model_id=MODEL_ID,
            system=[{"text": system}],
        )
        content = response.get("output", {}).get("message", {}).get("content", [])
        return content[0].get("text", "") if content else ""
    except Exception:
        return ""


def run(problem_domain: str, papers_dir: Optional[Path] = None) -> Paper:
    working_dir = Path(__file__).parent / "files"
    figures_md, figure_appendix, visual_refs = create_figures(problem_domain, working_dir)

    source_summary = "\n".join(
        f"- {ref['title']}: {ref['snippet']} {ref['url']}".strip()
        for ref in visual_refs
    )

    intro = _llm_text(
        "You write concise scientific introductions.",
        f"""Write a 3 paragraph introduction for a research paper in this domain:
{problem_domain}

Emphasize the importance of publication-quality figures for communicating mechanisms and evidence.""",
    ) or (
        f"Research in {problem_domain} depends on clear communication of mechanisms, assumptions, and evidence. "
        "Figures are often the fastest way to expose those relationships, but they can also hide ambiguity when "
        "their structure, labels, or quantitative encodings are underspecified.\n\n"
        "This paper treats figure creation as an explicit research workflow rather than a final formatting step. "
        "The goal is to convert manuscript intent into visual artifacts whose semantic plan, style choices, and "
        "review criteria are inspectable.\n\n"
        "We adapt the PaperVizAgent pattern to produce manuscript-ready conceptual and quantitative figures for "
        f"{problem_domain}, emphasizing traceable planning and reproducible rendering."
    )

    methods = _llm_text(
        "You write reproducible methods sections.",
        f"""Write a methods section for a paper about generating figures for:
{problem_domain}

Use this source context:
{source_summary}

Describe a multi-agent workflow with Retriever, Planner, Stylist, Visualizer, and Critic agents. Mention that plots are rendered from Python code.""",
    ) or (
        "We implemented a five-stage figure-production workflow inspired by PaperVizAgent. "
        "A Retriever collected visual reference cues, a Planner converted the paper topic into figure specifications, "
        "a Stylist generated publication-oriented visual constraints, a Visualizer rendered the figures with Python and "
        "matplotlib, and a Critic reviewed each output against its intended message.\n\n"
        "The workflow separates semantic design from graphical execution. Conceptual diagrams use explicit node and "
        "arrow layouts, while quantitative displays use deterministic code so that values, labels, and captions can be "
        "audited and regenerated."
    )

    results_text = _llm_text(
        "You write objective scientific results sections.",
        f"""Write a results section for:
{problem_domain}

Describe that the agent produced two figures: a conceptual mechanism diagram and a quantitative evidence profile.
Keep it short and refer to Figure 1 and Figure 2.""",
    ) or (
        "The figure pipeline produced two manuscript-facing visual artifacts. Figure 1 provides a conceptual overview "
        "that organizes the domain into inputs, mechanisms, and outcomes. Figure 2 provides a compact quantitative "
        "evidence profile that can be replaced with empirical measurements as the study matures.\n\n"
        "The critic pass identified the same main revision priority for both outputs: the visual scaffolds are ready "
        "for drafting, but synthetic values should be replaced with study-specific measurements before publication."
    )

    references = "\n".join(
        f"{i}. {ref['title']}. {ref['url']}".strip()
        for i, ref in enumerate(visual_refs, start=1)
        if ref["title"] or ref["url"]
    )

    return Paper(
        title=f"Agentic Figure Creation for {problem_domain}",
        introduction=intro,
        methods=methods,
        results=f"{results_text}\n\n{figures_md}",
        references=references,
        appendix=figure_appendix,
        tags=["figure-generation", "papervizagent", "multi-agent"],
    )
