from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchBrief:
    problem_domain: str
    findings: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    figure_specs: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    notes: str = ""
