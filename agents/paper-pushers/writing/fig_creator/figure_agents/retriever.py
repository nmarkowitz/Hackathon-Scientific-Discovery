from __future__ import annotations

def retrieve_visual_references(problem_domain: str) -> list[dict[str, str]]:
    try:
        from hackathon_science.tools import search_web
    except Exception:
        search_web = None

    queries = [
        f"{problem_domain} scientific figure schematic",
        f"{problem_domain} review paper figure",
        f"{problem_domain} statistical plot visualization",
    ]
    references: list[dict[str, str]] = []

    for query in queries:
        try:
            results = search_web(query, max_results=2) if search_web else []
        except Exception:
            results = []

        for result in results:
            references.append(
                {
                    "query": query,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", ""),
                }
            )

    if references:
        return references[:6]

    return [
        {
            "query": "fallback",
            "title": "Conceptual mechanism schematic",
            "url": "",
            "snippet": "Use a left-to-right workflow with inputs, mechanisms, measurements, and expected outcomes.",
        },
        {
            "query": "fallback",
            "title": "Comparative evidence plot",
            "url": "",
            "snippet": "Use a compact bar or line chart with uncertainty bands and direct annotations.",
        },
    ]
