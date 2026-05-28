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
