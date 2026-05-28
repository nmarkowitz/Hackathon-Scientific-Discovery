from __future__ import annotations

from .base import FigurePlan, ask_llm, extract_json


DEFAULT_STYLE = {
    "palette": ["#2B6CB0", "#38A169", "#D69E2E", "#805AD5"],
    "font": "DejaVu Sans",
    "tone": "publication-ready, restrained, high-contrast",
    "rules": [
        "Prefer direct labels over legends when space allows.",
        "Use minimal grid lines and readable axis labels.",
        "Keep conceptual diagrams left-to-right with clear causal arrows.",
    ],
}


def apply_style(problem_domain: str, plans: list[FigurePlan]) -> list[FigurePlan]:
    plan_text = "\n".join(f"{plan.figure_id}: {plan.title} - {plan.message}" for plan in plans)
    system = "You are the Stylist Agent for scientific figures."
    user = f"""Create concise visual style guidance for these figures about {problem_domain}.

Plans:
{plan_text}

Return JSON only with keys palette, font, tone, rules."""
    style = extract_json(ask_llm(system, user), DEFAULT_STYLE)
    if not isinstance(style, dict):
        style = DEFAULT_STYLE

    for plan in plans:
        plan.style = {**DEFAULT_STYLE, **style}

    return plans
