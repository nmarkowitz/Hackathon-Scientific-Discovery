"""
Paper-pushers run agent.

Workflow:
  1. pull_papers()  — LLM-generated queries → PubMed + web search → saved paper files
  2. _build_brief() — synthesize findings from fetched papers → ResearchBrief
  3. orchestrate()  — parallel section writing + fig_creator + peer review → Paper
  4. save_paper_as_pdf() — embed figures, save to papers/<title>/paper.pdf
"""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from hackathon_science import Paper
from hackathon_science.utils import call_llm

from pull_papers import pull_papers
from writing import ResearchBrief, orchestrate
from writing.pdf import save_paper_as_pdf

_AGENT_DIR = Path(__file__).parent
_PAPERS_DIR = _AGENT_DIR / "papers"
MODEL_ID = "global.anthropic.claude-sonnet-4-6"


def _llm(system: str, user: str) -> str:
    try:
        r = call_llm(
            messages=[{"role": "user", "content": [{"text": user}]}],
            model_id=MODEL_ID,
            system=[{"text": system}],
        )
        content = r.get("output", {}).get("message", {}).get("content", [])
        return content[0].get("text", "") if content else ""
    except Exception:
        return ""


def _build_brief(problem_domain: str, papers_data: list) -> ResearchBrief:
    """Synthesize a ResearchBrief from pull_papers() results."""
    citations = []
    snippets = []

    for result, _query, _source in papers_data[:20]:
        title = result.get("title", "")
        if title:
            citations.append({
                "title": title,
                "journal": result.get("journal", ""),
                "year": (result.get("publication_date") or "")[:4],
                "doi": result.get("doi", ""),
                "url": result.get("url", ""),
            })
        snippet = result.get("snippet") or result.get("content", "")
        if snippet:
            snippets.append(f"- {title}: {snippet[:300]}")

    snippets_text = "\n".join(snippets[:15])
    findings = _llm(
        "You are a scientific research synthesizer.",
        f"""Synthesize the key findings from these research snippets into 3-4 sentences
relevant to the problem domain.

Problem domain: {problem_domain}

Research snippets:
{snippets_text}

Return only the synthesized findings paragraph.""",
    ) or f"Literature review identified {len(citations)} relevant sources covering {problem_domain}."

    return ResearchBrief(
        problem_domain=problem_domain,
        findings=findings,
        citations=citations,
        notes=f"Based on {len(citations)} papers retrieved via automated literature search.",
    )


def run(problem_domain: str, papers_dir: Optional[Path] = None) -> Paper:
    # Step 1: fetch research papers
    research_dir = _AGENT_DIR / "research" / "".join(
        c if c.isalnum() or c in "-_" else "_" for c in problem_domain
    )[:60]
    print(f"[run] fetching papers for: {problem_domain!r}")
    papers_data = pull_papers(
        problem_domain,
        output_dir=str(research_dir),
        n_queries=5,
        use_exa=False,
        use_pubmed=True,
    )
    print(f"[run] fetched {len(papers_data)} papers")

    # Step 2: build ResearchBrief
    brief = _build_brief(problem_domain, papers_data)

    # Step 3: create paper output dir (use topic slug; renamed after title is known)
    topic_slug = "".join(c if c.isalnum() or c in " -_" else "_" for c in problem_domain)[:60].strip()
    paper_dir = _PAPERS_DIR / topic_slug
    paper_dir.mkdir(parents=True, exist_ok=True)
    brief.output_dir = paper_dir

    # Step 4: run writing pipeline
    print("[run] running writing pipeline...")
    paper = orchestrate(brief)

    # Rename dir to final title
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in paper.title)[:60].strip()
    final_dir = _PAPERS_DIR / safe_title
    if paper_dir.exists() and paper_dir != final_dir:
        paper_dir.rename(final_dir)
    paper_dir = final_dir

    # Step 5: save PDF with embedded figures
    figures_dir = paper_dir / "figures"
    figure_paths = sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []
    pdf_path = paper_dir / "paper.pdf"
    save_paper_as_pdf(paper, pdf_path, figure_paths=figure_paths)
    print(f"[run] PDF saved → {pdf_path} ({len(figure_paths)} figure(s))")

    return paper
