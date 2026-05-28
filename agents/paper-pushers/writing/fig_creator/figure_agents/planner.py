from __future__ import annotations

from .base import FigurePlan, ask_llm, extract_json


def plan_figures(problem_domain: str, references: list[dict[str, str]]) -> list[FigurePlan]:
    reference_text = "\n".join(
        f"- {ref.get('title', '')}: {ref.get('snippet', '')}" for ref in references
    )
    fallback = [
        {
            "figure_id": "fig1",
            "title": "Mechanistic overview",
            "caption": f"Conceptual model linking core variables in {problem_domain}.",
            "figure_type": "diagram",
            "message": "Show how inputs flow through mechanisms into measurable outcomes.",
            "elements": ["Research context", "Intervention or exposure", "Mechanism", "Measured outcome"],
        },
        {
            "figure_id": "fig2",
            "title": "Evidence profile",
            "caption": f"Synthetic evidence profile for candidate hypotheses in {problem_domain}.",
            "figure_type": "plot",
            "message": "Compare plausible hypotheses with transparent effect and confidence scores.",
            "elements": ["Hypotheses", "Effect score", "Confidence interval", "Interpretive annotation"],
        },
    ]

    system = "You are the Planner Agent in a PaperVizAgent-style scientific figure pipeline."
    user = f"""Create exactly two figure plans for a paper about:
{problem_domain}

Reference cues:
{reference_text}

Return JSON only as an array of objects with keys:
figure_id, title, caption, figure_type, message, elements.
Use one conceptual diagram and one quantitative plot."""
    data = extract_json(ask_llm(system, user), fallback)

    plans: list[FigurePlan] = []
    for index, item in enumerate(data[:2], start=1):
        plans.append(
            FigurePlan(
                figure_id=str(item.get("figure_id") or f"fig{index}"),
                title=str(item.get("title") or fallback[index - 1]["title"]),
                caption=str(item.get("caption") or fallback[index - 1]["caption"]),
                figure_type=str(item.get("figure_type") or fallback[index - 1]["figure_type"]),
                message=str(item.get("message") or fallback[index - 1]["message"]),
                elements=list(item.get("elements") or fallback[index - 1]["elements"]),
            )
        )

    return plans
