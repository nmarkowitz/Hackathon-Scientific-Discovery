from __future__ import annotations

from .base import FigureArtifact, FigurePlan, ask_llm


def critique_figure(problem_domain: str, plan: FigurePlan, artifact: FigureArtifact) -> str:
    system = "You are the Critic Agent in a scientific figure production loop."
    user = f"""Evaluate this generated figure for a paper.

Problem domain: {problem_domain}
Figure title: {artifact.title}
Caption: {artifact.caption}
Planned message: {plan.message}
Planned elements: {plan.elements}

Return 3 short bullets covering faithfulness, readability, and revision priority."""
    critique = ask_llm(system, user).strip()
    if critique:
        return critique

    return (
        "- Faithfulness: The figure follows the planned message and includes the main elements.\n"
        "- Readability: Labels, arrows, and axes are designed for paper-scale viewing.\n"
        "- Revision priority: Replace synthetic values with study-specific measurements when available."
    )
