"""
Quick end-to-end test: search PubMed → build ResearchBrief → run writing pipeline.
Topic: "how to ride a bicycle using quantum mechanics and a squirrel"

Run with:
    uv run python agents/paper-pushers/test_paper.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pymed import PubMed
from writing import ResearchBrief, orchestrate
from writing.pdf import save_paper_as_pdf

PAPERS_DIR = Path(__file__).parent / "papers"

TOPIC = "how to ride a bicycle using quantum mechanics and a squirrel"

QUERIES = [
    ("quantum tunneling locomotion", 4),
    ("bicycle dynamics stability balance", 4),
    ("squirrel biomechanics agility", 3),
    ("quantum biology animal navigation", 3),
]


def fetch_papers(queries: list[tuple[str, int]]) -> list[dict]:
    import time
    pubmed = PubMed(tool="HackathonPaperWriter", email="noahb.markowitz@gmail.com")
    papers = []
    for i, (query, max_results) in enumerate(queries):
        if i > 0:
            time.sleep(1)  # PubMed rate limit: max 3 requests/sec without API key
        print(f"  searching: {query!r} ...")
        try:
            results = list(pubmed.query(query, max_results=max_results))
            for r in results:
                if r.title:
                    papers.append({
                        "title": r.title or "",
                        "abstract": r.abstract or "",
                        "journal": getattr(r, "journal", "") or "",
                        "year": str(r.publication_date.year) if r.publication_date else "",
                        "doi": r.doi or "",
                    })
        except Exception as e:
            print(f"    warning: {e}")
    return papers


def build_brief(papers: list[dict]) -> ResearchBrief:
    findings = (
        "Quantum coherence effects observed in biological systems suggest that "
        "macroscopic locomotion may exploit quantum mechanical phenomena. "
        "Squirrels demonstrate exceptional dynamic balance on complex surfaces, "
        "with studies showing rapid gyroscopic correction analogous to bicycle "
        "self-stabilization. Bicycle stability analysis reveals angular momentum "
        "conservation principles that parallel quantum spin dynamics at macro scale."
    )

    citations = [
        {"title": p["title"], "journal": p["journal"], "year": p["year"]}
        for p in papers
        if p["title"]
    ]

    notes = (
        "This paper explores the theoretical intersection of quantum mechanics, "
        "classical bicycle dynamics, and squirrel locomotion biomechanics. "
        "The squirrel serves as a biological analogue for the quantum-classical "
        "boundary in balance control systems."
    )

    return ResearchBrief(
        problem_domain=TOPIC,
        findings=findings,
        citations=citations,
        notes=notes,
        # output_dir set by main() after title is known
    )


def print_paper(paper) -> None:
    sep = "\n" + "=" * 70 + "\n"
    print(sep)
    print(f"TITLE: {paper.title}")
    print(sep)
    print("ABSTRACT / INTRODUCTION")
    print("-" * 40)
    print(paper.introduction[:800] + "..." if len(paper.introduction) > 800 else paper.introduction)
    print(sep)
    print("METHODS")
    print("-" * 40)
    print(paper.methods[:600] + "..." if len(paper.methods) > 600 else paper.methods)
    print(sep)
    print("RESULTS")
    print("-" * 40)
    # Skip embedded base64 images in terminal output
    results_preview = paper.results
    if "data:image" in results_preview:
        idx = results_preview.find("data:image")
        results_preview = results_preview[:idx] + "[... embedded figures ...]"
    print(results_preview[:800] + "..." if len(results_preview) > 800 else results_preview)
    print(sep)
    print("REFERENCES")
    print("-" * 40)
    print(paper.references[:400] + "..." if len(paper.references) > 400 else paper.references)
    if paper.appendix:
        print(sep)
        print("APPENDIX (figure critic notes)")
        print("-" * 40)
        print(paper.appendix[:400] + "..." if len(paper.appendix) > 400 else paper.appendix)
    print(sep)
    print(f"Tags: {paper.tags}")


def main():
    print(f"\nTopic: {TOPIC!r}\n")
    print("Step 1: Fetching papers from PubMed...")
    papers = fetch_papers(QUERIES)
    print(f"  found {len(papers)} papers\n")
    for p in papers[:5]:
        print(f"  - {p['title'][:80]}")
    print()

    print("Step 2: Building ResearchBrief...")
    brief = build_brief(papers)
    print(f"  {len(brief.citations)} citations loaded\n")

    # Create a temporary paper dir using the topic slug; will be renamed once title is known
    topic_slug = "".join(c if c.isalnum() or c in " -_" else "_" for c in TOPIC)[:60].strip()
    paper_dir = PAPERS_DIR / topic_slug
    paper_dir.mkdir(parents=True, exist_ok=True)
    brief.output_dir = paper_dir

    print("Step 3: Running writing pipeline (this takes ~2-3 minutes)...")
    print("  [method + references running in parallel]")
    print("  [results, figures, intro, discussion, abstract follow in waves]")
    print("  [two peer reviewers run in parallel on the assembled draft]")
    print("  [revision pass applies reviewer feedback per section]")
    print()

    paper = orchestrate(brief)

    # Rename dir to final title
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in paper.title)[:60].strip()
    final_dir = PAPERS_DIR / safe_title
    if paper_dir != final_dir:
        paper_dir.rename(final_dir)
    paper_dir = final_dir

    # Collect generated PNGs from figures subdir
    figures_dir = paper_dir / "figures"
    figure_paths = sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []

    print("Step 4: Saving PDF...")
    pdf_path = paper_dir / "paper.pdf"
    save_paper_as_pdf(paper, pdf_path, figure_paths=figure_paths)
    print(f"  saved → {pdf_path}")
    if figure_paths:
        print(f"  embedded {len(figure_paths)} figure(s): {[p.name for p in figure_paths]}")
    print()

    print("Step 5: Done!")
    print_paper(paper)


if __name__ == "__main__":
    main()
