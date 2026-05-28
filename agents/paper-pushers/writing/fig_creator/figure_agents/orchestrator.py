from __future__ import annotations

from pathlib import Path

from .critic import critique_figure
from .planner import plan_figures
from .retriever import retrieve_visual_references
from .stylist import apply_style
from .visualizer import render_figures


def create_figures(problem_domain: str, working_dir: Path) -> tuple[str, str, list[dict[str, str]]]:
    references = retrieve_visual_references(problem_domain)
    plans = apply_style(problem_domain, plan_figures(problem_domain, references))
    artifacts = render_figures(plans, working_dir)

    figure_sections = []
    critiques = []
    for plan, artifact in zip(plans, artifacts):
        artifact.critique = critique_figure(problem_domain, plan, artifact)
        figure_sections.append(
            f"**{artifact.title}.** {artifact.caption}\n\n{artifact.markdown}"
        )
        critiques.append(f"### {artifact.title}\n{artifact.critique}")

    methods_note = (
        "Figures were generated with a PaperVizAgent-style multi-agent workflow: "
        "reference retrieval, semantic planning, style synthesis, deterministic matplotlib "
        "rendering, and critic review."
    )
    appendix = "## Figure Critic Notes\n\n" + "\n\n".join(critiques)
    reference_rows = [
        {"title": ref.get("title", ""), "url": ref.get("url", ""), "snippet": ref.get("snippet", "")}
        for ref in references
    ]

    return "\n\n".join(figure_sections), f"{methods_note}\n\n{appendix}", reference_rows
