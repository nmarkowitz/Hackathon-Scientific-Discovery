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
