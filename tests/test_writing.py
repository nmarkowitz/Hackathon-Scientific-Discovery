import pytest
from writing.brief import ResearchBrief


def test_research_brief_defaults():
    brief = ResearchBrief(problem_domain="effect of caffeine on reaction time")
    assert brief.problem_domain == "effect of caffeine on reaction time"
    assert brief.findings == ""
    assert brief.data == {}
    assert brief.figure_specs == []
    assert brief.citations == []
    assert brief.notes == ""


def test_research_brief_full():
    brief = ResearchBrief(
        problem_domain="caffeine study",
        findings="Caffeine reduces reaction time by 15%.",
        data={"reaction_times_ms": [320, 290, 305, 270, 315]},
        figure_specs=[{"type": "bar", "title": "Reaction time by dose"}],
        citations=[{"title": "Smith 2020", "url": "https://example.com"}],
        notes="Preliminary findings only.",
    )
    assert brief.problem_domain == "caffeine study"
    assert brief.findings == "Caffeine reduces reaction time by 15%."
    assert brief.data["reaction_times_ms"] == [320, 290, 305, 270, 315]
    assert brief.figure_specs == [{"type": "bar", "title": "Reaction time by dose"}]
    assert len(brief.citations) == 1
    assert brief.citations[0]["title"] == "Smith 2020"
    assert brief.notes == "Preliminary findings only."
