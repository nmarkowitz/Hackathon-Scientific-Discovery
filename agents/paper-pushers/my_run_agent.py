"""
Run agent template for paper-pushers.
Write your paper-generation logic here.
"""
from pathlib import Path
from typing import Optional
from hackathon_science import Paper
from hackathon_science.tools import run_code, search_web, get_paper, image_to_base64
from hackathon_science.utils import call_llm


def run(
    problem_domain: str,
    papers_dir: Optional[Path] = None
) -> Paper:
    """
    Generate a research paper.

    Args:
        problem_domain: Research area prompt (same for all teams)
        papers_dir: Optional path to papers directory (use with get_paper)

    Returns:
        Paper with structured fields (title, introduction, methods, results, references, appendix)
        Note: appendix is auto-populated with code from script.py in working_dir if not set

    Available tools:
        - run_code(command, timeout=300): Execute bash in container
        - search_web(query, max_results=10): Search web for background
        - get_paper(paper_id, papers_dir): Read full paper by ID from ecosystem
        - image_to_base64(image_path, alt_text): Convert image to base64 markdown
        - call_llm(messages, model_id, tools=None): Call Bedrock or OpenAI model

    Example:
        # Run experiment
        code_output = run_code("python -c 'import numpy as np; print(np.mean([1,2,3]))'")

        # Search for context
        search_results = search_web("multi-agent cooperation game theory")

        # Read and build on related paper
        if papers_dir:
            related_paper = get_paper("abc12345", papers_dir)
            if related_paper:
                print(f"Building on: {related_paper['title']}")

        # Embed images in paper
        # Create a plot and save it
        run_code('''
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.savefig("plot.png")
''')
        # Convert to base64 markdown for embedding
        plot_md = image_to_base64("plot.png", "Experimental Results")

        # Call LLM via Bedrock
        response = call_llm(
            messages=[{"role": "user", "content": [{"text": "Analyze this data..."}]}],
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        # Or use OpenAI (requires OPENAI_API_KEY env var)
        response = call_llm(
            messages=[{"role": "user", "content": [{"text": "Analyze this data..."}]}],
            model_id="gpt-4o"
        )
        # Extract text from response: response["output"]["message"]["content"][0]["text"]
        output = response.get("output", {}).get("message", {}).get("content", [])
        text = output[0].get("text", "") if output else ""
    """

    # TODO: Implement your agent logic here
    # 1. Optionally load existing papers with get_paper() to review or build upon
    # 2. Design experiments and run code
    # 3. Analyze results
    # 4. Write paper sections

    return Paper(
        title="",
        introduction="",
        methods="",
        results="",
        references="",  # Optional: citations and references
        appendix="",    # Optional: auto-populated from script.py if empty
        tags=[]
    )
