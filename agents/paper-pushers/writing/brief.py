from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ResearchBrief:
    problem_domain: str
    findings: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    figure_specs: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    notes: str = ""
    output_dir: Optional[Path] = None  # paper-specific dir; figures saved to output_dir/figures/
