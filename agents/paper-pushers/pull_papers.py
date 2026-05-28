"""Pull relevant research papers and web content based on problem description."""

from pathlib import Path
from hackathon_science import call_llm, search_web
import json
import os

try:
    from hackathon_science import search_exa
except ImportError:
    search_exa = None  # type: ignore

try:
    from hackathon_science import search_pubmed
    _has_search_pubmed = True
except ImportError:
    _has_search_pubmed = False

def _pubmed_fallback(query: str, max_results: int = 5) -> list[dict]:
    """Fallback PubMed search using pymed when search_pubmed isn't available."""
    try:
        from pymed import PubMed
        import time
        pubmed = PubMed(tool="HackathonPaperWriter", email="hackathon@example.com")
        time.sleep(0.4)
        results = []
        for r in pubmed.query(query, max_results=max_results):
            if not r.title:
                continue
            results.append({
                "title": r.title,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{r.pubmed_id}/" if r.pubmed_id else "",
                "snippet": r.abstract[:500] if r.abstract else "",
                "content": r.abstract or "",
                "pubmed_id": r.pubmed_id or "",
                "authors": [str(a) for a in (r.authors or [])[:5]],
                "journal": getattr(r, "journal", "") or "",
                "publication_date": str(r.publication_date) if r.publication_date else "",
                "doi": r.doi or "",
                "source": "pubmed",
            })
        return results
    except Exception:
        return []

if not _has_search_pubmed:
    search_pubmed = _pubmed_fallback  # type: ignore


def generate_search_queries(problem_description: str, n: int = 10, previous_queries: list[str] = None) -> list[str]:
    """Generate diverse search queries from problem description using LLM.

    Args:
        problem_description: Description of the research problem/paper topic
        n: Number of queries to generate (default: 10)
        previous_queries: List of queries from previous runs to avoid duplication

    Returns:
        List of search query strings
    """
    previous_context = ""
    if previous_queries:
        previous_context = f"""

IMPORTANT: The following queries have already been used in previous runs. Generate NEW queries that explore DIFFERENT aspects:

Previous queries:
{chr(10).join(f"- {q}" for q in previous_queries)}

Your new queries must be meaningfully different and explore areas not covered by the previous queries."""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": f"""Generate {n} diverse, specific search queries to find relevant research papers, blog posts, and articles for the following problem:

{problem_description}

Requirements:
- Make queries specific and targeted (e.g., include technical terms, methodologies, key concepts)
- Vary the perspective (theoretical foundations, practical applications, recent advances, related work)
- Include queries for both academic papers and practical implementations/blog posts
- Focus on actionable, searchable terms{previous_context}

Return ONLY a JSON array of {n} query strings, nothing else."""
                }
            ]
        }
    ]

    response = call_llm(
        messages=messages,
        model_id="global.anthropic.claude-sonnet-4-6"
    )

    # Extract text from response
    content = response["output"]["message"]["content"]
    text = ""
    for block in content:
        if "text" in block:
            text += block["text"]

    # Parse JSON array from response
    try:
        queries = json.loads(text.strip())
        return queries[:n]
    except json.JSONDecodeError:
        # Fallback: extract lines that look like queries
        lines = [line.strip(' ",-[]') for line in text.split('\n') if line.strip()]
        return [line for line in lines if len(line) > 10][:n]


def generate_pubmed_queries(problem_description: str, n: int = 5, previous_queries: list[str] = None) -> list[str]:
    """Generate simple, PubMed-optimized queries for biomedical literature search.

    Args:
        problem_description: Description of the research problem/paper topic
        n: Number of queries to generate (default: 5)
        previous_queries: List of queries from previous runs to avoid duplication

    Returns:
        List of simple search query strings optimized for PubMed
    """
    previous_context = ""
    if previous_queries:
        previous_context = f"""

IMPORTANT: The following PubMed queries have already been used. Generate NEW queries that are different:

Previous PubMed queries:
{chr(10).join(f"- {q}" for q in previous_queries)}

Your new queries must explore different aspects."""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": f"""Generate {n} simple PubMed search queries for the following problem:

{problem_description}

CRITICAL PubMed Query Requirements:
- Use 2-5 words MAXIMUM per query (PubMed works best with simple queries)
- Focus ONLY on biomedical/life science aspects (genes, proteins, diseases, drugs, therapies, biology, medicine)
- Use specific scientific terms, not abstract concepts
- Examples of GOOD queries: "CRISPR cancer therapy", "machine learning drug discovery", "deep learning protein folding"
- Examples of BAD queries: "AI-guided experimental design optimization", "transformer models cross-domain knowledge", "automated hypothesis generation systems"
- If the topic has no biomedical connection, return an empty array []
- Return queries that will actually find papers in PubMed's biomedical database{previous_context}

Return ONLY a JSON array of {n} simple query strings (or empty array if not biomedical), nothing else."""
                }
            ]
        }
    ]

    response = call_llm(
        messages=messages,
        model_id="global.anthropic.claude-sonnet-4-6"
    )

    # Extract text from response
    content = response["output"]["message"]["content"]
    text = ""
    for block in content:
        if "text" in block:
            text += block["text"]

    # Parse JSON array from response
    try:
        queries = json.loads(text.strip())
        # Filter to ensure they're actually simple (2-5 words)
        simple_queries = [q for q in queries if len(q.split()) <= 5]
        return simple_queries[:n]
    except json.JSONDecodeError:
        # Fallback: extract lines that look like queries
        lines = [line.strip(' ",-[]') for line in text.split('\n') if line.strip()]
        simple_lines = [line for line in lines if 5 < len(line) < 50 and len(line.split()) <= 5]
        return simple_lines[:n]


def load_previous_queries(output_dir: str) -> tuple[list[str], list[str]]:
    """Load all queries from previous runs to ensure diversity.

    Args:
        output_dir: Directory where previous summaries are stored

    Returns:
        Tuple of (web_queries, pubmed_queries) from previous runs
    """
    summary_file = Path(output_dir) / "summary.txt"
    if not summary_file.exists():
        return [], []

    web_queries = []
    pubmed_queries = []

    with open(summary_file, 'r') as f:
        content = f.read()

        # Find web queries section
        if "All Web Queries (Current + Previous Runs):" in content:
            queries_section = content.split("All Web Queries (Current + Previous Runs):")[1]
            if "All PubMed Queries" in queries_section:
                queries_section = queries_section.split("All PubMed Queries")[0]
            elif "=" in queries_section:
                queries_section = queries_section.split("=")[0]

            for line in queries_section.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    query = line.split('.', 1)[-1].strip()
                    if query and len(query) > 10:
                        web_queries.append(query)

        # Find PubMed queries section
        if "All PubMed Queries (Current + Previous Runs):" in content:
            queries_section = content.split("All PubMed Queries (Current + Previous Runs):")[1]
            if "=" in queries_section:
                queries_section = queries_section.split("=")[0]

            for line in queries_section.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    query = line.split('.', 1)[-1].strip()
                    if query and len(query) > 5:
                        pubmed_queries.append(query)

    return web_queries, pubmed_queries


def pull_papers(problem_description: str, output_dir: str = "data/external_research", n_queries: int = 10, use_exa: bool = True, use_pubmed: bool = True):
    """Pull relevant research content based on problem description.

    Args:
        problem_description: Description of research problem/paper topic
        output_dir: Directory to save results (default: data/external_research)
        n_queries: Number of search queries to generate (default: 10)
        use_exa: Use Exa for better search results if API key available (default: True)
        use_pubmed: Use PubMed for academic papers (default: True)
    """
    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Create papers subdirectory
    papers_path = out_path / "papers"
    papers_path.mkdir(exist_ok=True)

    # Load previous queries to ensure diversity
    previous_web_queries, previous_pubmed_queries = load_previous_queries(output_dir)
    if previous_web_queries or previous_pubmed_queries:
        print(f"Found {len(previous_web_queries)} web queries and {len(previous_pubmed_queries)} PubMed queries from previous runs")

    # Check if Exa is available
    has_exa = use_exa and search_exa is not None and os.environ.get('EXA_API_KEY')
    search_engine = "Exa" if has_exa else "DuckDuckGo"
    print(f"Using {search_engine} for web searches")

    print(f"\nGenerating {n_queries} NEW web search queries...")
    web_queries = generate_search_queries(problem_description, n=n_queries, previous_queries=previous_web_queries)

    print(f"\nGenerated web queries:")
    for i, query in enumerate(web_queries, 1):
        print(f"  {i}. {query}")

    # Generate simpler PubMed-specific queries if enabled
    pubmed_queries = []
    if use_pubmed:
        print(f"\nGenerating {n_queries // 2} simple PubMed queries...")
        pubmed_queries = generate_pubmed_queries(problem_description, n=max(3, n_queries // 2), previous_queries=previous_pubmed_queries)

        if pubmed_queries:
            print(f"\nGenerated PubMed queries:")
            for i, query in enumerate(pubmed_queries, 1):
                print(f"  {i}. {query}")
        else:
            print("  (No biomedical queries generated - topic may not be suitable for PubMed)")

    # Load existing papers to continue numbering and avoid reprocessing
    existing_papers = {}  # url -> filename
    if papers_path.exists():
        for paper_file in papers_path.glob("paper_*.txt"):
            # Read the URL from the file
            with open(paper_file, 'r') as f:
                for line in f:
                    if line.startswith("URL: "):
                        url = line[5:].strip()
                        existing_papers[url] = paper_file.name
                        break

    # Track unique papers by URL for deduplication
    seen_urls = {}  # url -> (result, query, source)
    paper_count = len(existing_papers)
    print(f"Found {paper_count} existing papers in database")

    # Track sources
    sources_count = {'pubmed': 0, 'exa': 0, 'duckduckgo': 0}

    # First, search PubMed with simple queries
    if use_pubmed and pubmed_queries:
        print(f"\n{'='*80}")
        print(f"PUBMED SEARCH - {len(pubmed_queries)} simple queries")
        print(f"{'='*80}")

        for i, query in enumerate(pubmed_queries, 1):
            print(f"\nPubMed ({i}/{len(pubmed_queries)}): {query}")
            try:
                pubmed_results = search_pubmed(query, max_results=5)
                print(f"  Found {len(pubmed_results)} papers")

                for result in pubmed_results:
                    result['source'] = 'pubmed'
                    url = result['url']

                    if not url or url in existing_papers or url in seen_urls:
                        continue

                    seen_urls[url] = (result, query, 'pubmed')
                    sources_count['pubmed'] += 1
                    paper_count += 1

                    # Save paper file
                    title = result['title'][:100]
                    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
                    safe_title = safe_title.strip().replace(' ', '_')
                    paper_file = papers_path / f"paper_{paper_count:03d}_{safe_title}.txt"

                    with open(paper_file, 'w') as f:
                        f.write(f"Title: {result['title']}\n")
                        f.write(f"URL: {result['url']}\n")
                        f.write(f"Query: {query}\n")
                        f.write(f"Source: pubmed\n")
                        if result.get('pubmed_id'):
                            f.write(f"PMID: {result['pubmed_id']}\n")
                        if result.get('authors'):
                            f.write(f"Authors: {', '.join(result['authors'][:5])}\n")
                        if result.get('journal'):
                            f.write(f"Journal: {result['journal']}\n")
                        if result.get('publication_date'):
                            f.write(f"Published: {result['publication_date']}\n")
                        if result.get('doi'):
                            f.write(f"DOI: {result['doi']}\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(f"Snippet:\n{result['snippet']}\n\n")
                        if result.get('content'):
                            f.write("=" * 80 + "\n")
                            f.write("Full Content:\n")
                            f.write("=" * 80 + "\n\n")
                            f.write(f"{result['content']}\n")
            except Exception as e:
                print(f"  PubMed search failed: {e}")

        print(f"\n✅ PubMed complete: Added {sources_count['pubmed']} papers")

    # Then, search web with detailed queries
    print(f"\n{'='*80}")
    print(f"WEB SEARCH - {len(web_queries)} detailed queries")
    print(f"{'='*80}")

    for i, query in enumerate(web_queries, 1):
        print(f"\nWeb ({i}/{len(web_queries)}): {query}")

        all_results = []

        # Search web for additional content
        try:
            if has_exa:
                web_results = search_exa(query, max_results=10, include_text=True)
                for result in web_results:
                    result['source'] = 'exa'
            else:
                web_results = search_web(query, max_results=10, use_exa=False)
                for result in web_results:
                    result['source'] = 'duckduckgo'
            all_results.extend(web_results)
            print(f"  Found {len(web_results)} results")
        except Exception as e:
            print(f"  Web search failed: {e}")

        print(f"  Total results: {len(all_results)}")

        # Process web results
        new_papers_this_query = 0
        for result in all_results:
            url = result['url']

            # Skip empty URLs
            if not url:
                continue

            # Check if already exists in database
            if url in existing_papers:
                print(f"  Already in database: {result['title'][:60]}...")
                continue

            # Deduplicate within current run
            if url not in seen_urls:
                source = result.get('source', 'unknown')
                seen_urls[url] = (result, query, source)
                sources_count[source] = sources_count.get(source, 0) + 1
                paper_count += 1
                new_papers_this_query += 1

                # Create safe filename from title
                title = result['title'][:100]  # Limit length
                safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
                safe_title = safe_title.strip().replace(' ', '_')

                # Save individual paper file
                paper_file = papers_path / f"paper_{paper_count:03d}_{safe_title}.txt"

                with open(paper_file, 'w') as f:
                    f.write(f"Title: {result['title']}\n")
                    f.write(f"URL: {result['url']}\n")
                    f.write(f"Query: {query}\n")
                    f.write(f"Source: {source}\n")

                    # Exa-specific metadata
                    if source == 'exa':
                        if result.get('published_date'):
                            f.write(f"Published: {result['published_date']}\n")
                        if result.get('author'):
                            f.write(f"Author: {result['author']}\n")

                    f.write("=" * 80 + "\n\n")

                    # Add snippet
                    f.write(f"Snippet:\n{result['snippet']}\n\n")

                    # Add full content if available
                    if result.get('content'):
                        f.write("=" * 80 + "\n")
                        f.write("Full Content:\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(f"{result['content']}\n")
            else:
                print(f"  Duplicate in current run: {result['title'][:60]}...")

        print(f"  Added {new_papers_this_query} new papers")

    # Combine current and previous queries for tracking
    all_web_queries = previous_web_queries + web_queries
    all_pubmed_queries = previous_pubmed_queries + pubmed_queries

    # Save summary with all results
    summary_file = out_path / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Research Pull Summary\n")
        f.write(f"Problem: {problem_description}\n")
        f.write(f"Sources: PubMed" + (", Exa" if has_exa else ", DuckDuckGo") + "\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Current run web queries: {len(web_queries)}\n")
        f.write(f"Current run PubMed queries: {len(pubmed_queries)}\n")
        f.write(f"Previous run web queries: {len(previous_web_queries)}\n")
        f.write(f"Previous run PubMed queries: {len(previous_pubmed_queries)}\n")
        f.write(f"Total web queries across all runs: {len(all_web_queries)}\n")
        f.write(f"Total PubMed queries across all runs: {len(all_pubmed_queries)}\n")
        f.write(f"Unique papers in database: {len(seen_urls)}\n")
        f.write(f"Papers with full content: {sum(1 for r, q, s in seen_urls.values() if r.get('content'))}\n\n")
        f.write(f"Sources breakdown:\n")
        f.write(f"  - PubMed: {sources_count.get('pubmed', 0)} papers\n")
        f.write(f"  - Exa: {sources_count.get('exa', 0)} results\n")
        f.write(f"  - DuckDuckGo: {sources_count.get('duckduckgo', 0)} results\n\n")

        f.write("Current Run Web Queries:\n")
        for i, query in enumerate(web_queries, 1):
            f.write(f"  {i}. {query}\n")

        if pubmed_queries:
            f.write(f"\nCurrent Run PubMed Queries:\n")
            for i, query in enumerate(pubmed_queries, 1):
                f.write(f"  {i}. {query}\n")

        if previous_web_queries:
            f.write(f"\nPrevious Run Web Queries ({len(previous_web_queries)}):\n")
            for i, query in enumerate(previous_web_queries, 1):
                f.write(f"  {i}. {query}\n")

        if previous_pubmed_queries:
            f.write(f"\nPrevious Run PubMed Queries ({len(previous_pubmed_queries)}):\n")
            for i, query in enumerate(previous_pubmed_queries, 1):
                f.write(f"  {i}. {query}\n")

        f.write("\n" + "=" * 80 + "\n\n")
        f.write("All Web Queries (Current + Previous Runs):\n")
        for i, query in enumerate(all_web_queries, 1):
            f.write(f"  {i}. {query}\n")

        if all_pubmed_queries:
            f.write("\n" + "=" * 80 + "\n\n")
            f.write("All PubMed Queries (Current + Previous Runs):\n")
            for i, query in enumerate(all_pubmed_queries, 1):
                f.write(f"  {i}. {query}\n")

        f.write("\n" + "=" * 80 + "\n\n")
        f.write("Papers Index:\n")
        for i, (url, (result, query, source)) in enumerate(sorted(seen_urls.items()), 1):
            f.write(f"{i:3d}. {result['title']}\n")
            f.write(f"     URL: {url}\n")
            f.write(f"     Source: {source}\n")
            f.write(f"     Query: {query}\n")
            if source == 'pubmed' and result.get('pubmed_id'):
                f.write(f"     PMID: {result['pubmed_id']}\n")
            f.write("\n")

    total_papers = len(existing_papers) + len(seen_urls)

    print(f"\n✅ Run complete!")
    print(f"  - New papers added: {len(seen_urls)}")
    print(f"    • PubMed: {sources_count.get('pubmed', 0)}")
    print(f"    • Exa: {sources_count.get('exa', 0)}")
    print(f"    • DuckDuckGo: {sources_count.get('duckduckgo', 0)}")
    print(f"  - Total papers in database: {total_papers}")
    print(f"  - Location: {output_dir}/papers/")
    print(f"  - Summary with full query history: summary.txt")
    if seen_urls:
        print(f"  - Full content available for {sum(1 for r, q, s in seen_urls.values() if r.get('content'))} new papers")

    return list(seen_urls.values())


if __name__ == "__main__":
    # Example usage
    problem = """
    Investigating the application of large language models to scientific discovery,
    specifically focusing on automated hypothesis generation and experimental design
    in AI thinking.
    """

    pull_papers(problem, n_queries=10)
