import tempfile
from pathlib import Path

from hackathon_science.tools import run_code, image_to_base64
from .brief import ResearchBrief
from .base import llm_call


def generate_figures(brief: ResearchBrief, deps: dict[str, str]) -> str:
    results_text = deps.get("results", "")
    tmpdir = tempfile.mkdtemp()

    system = "You are a scientific figure generator. Write Python matplotlib code to visualize results."
    user = f"""Write Python code to generate 1-3 figures for a scientific paper.

Results section:
{results_text}

Available data:
{brief.data}

Figure hints:
{brief.figure_specs}

Requirements:
- Save each figure as an absolute path: '{tmpdir}/figure_1.png', '{tmpdir}/figure_2.png', etc.
- Use: plt.savefig('ABSOLUTE_PATH', dpi=150, bbox_inches='tight')
- Call plt.close() after each figure
- Import matplotlib.pyplot as plt, numpy as np, and anything else needed
- Generate at most 3 figures

Return ONLY Python code, no explanation or markdown."""
    code = llm_call(system, user)
    if not code:
        return ""

    run_code(code)

    figures_md = []
    for i in range(1, 10):
        png_path = Path(tmpdir) / f"figure_{i}.png"
        if not png_path.exists():
            break
        try:
            md = image_to_base64(str(png_path), f"Figure {i}")
            figures_md.append(md)
        except Exception:
            break

    return "\n\n".join(figures_md)
