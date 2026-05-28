import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "fig_creator"))
from figure_agents import create_figures  # noqa: E402

from .brief import ResearchBrief


def generate_figures(brief: ResearchBrief, deps: dict[str, str]) -> tuple[str, str]:
    working_dir = Path(__file__).parent / "fig_creator" / "files"
    working_dir.mkdir(parents=True, exist_ok=True)
    try:
        figures_md, figure_appendix, _ = create_figures(brief.problem_domain, working_dir)
    except Exception as e:
        print(f"[figures] fig_creator failed: {e}")
        return "", ""
    return figures_md, figure_appendix
