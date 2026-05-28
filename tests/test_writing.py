import pytest
from writing.brief import ResearchBrief
from unittest.mock import patch
from writing.base import llm_call, MODEL_ID


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


def test_llm_call_extracts_text():
    mock_response = {
        "output": {
            "message": {
                "content": [{"text": "This is the section text."}]
            }
        }
    }
    with patch("writing.base.call_llm", return_value=mock_response) as mock:
        result = llm_call("You are a writer.", "Write a methods section.")
    assert result == "This is the section text."
    mock.assert_called_once()


def test_llm_call_returns_empty_on_error():
    with patch("writing.base.call_llm", side_effect=Exception("Bedrock down")):
        result = llm_call("system", "user")
    assert result == ""


def test_model_id_is_correct_inference_profile():
    assert MODEL_ID == "global.anthropic.claude-sonnet-4-6"
    assert not MODEL_ID.endswith("v1:0")


from writing.sections import (
    write_method, write_references, write_results,
    write_intro, write_discussion, write_abstract,
)


SAMPLE_BRIEF = ResearchBrief(
    problem_domain="effect of caffeine on human reaction time",
    findings="Participants given 200mg caffeine showed 15% faster reaction times vs placebo (n=40, p<0.01).",
    data={"reaction_times_ms": {"caffeine": [290, 305, 270, 315], "placebo": [340, 355, 320, 360]}},
    citations=[
        {"title": "Caffeine and Cognition (Smith 2020)", "journal": "J. Neuroscience", "year": 2020},
        {"title": "Stimulant Effects Review (Jones 2019)", "journal": "Pharmacology", "year": 2019},
    ],
    notes="Double-blind RCT. All participants healthy adults aged 18-35.",
)


@pytest.mark.integration
def test_write_method_returns_nonempty():
    result = write_method(SAMPLE_BRIEF, {})
    assert isinstance(result, str)
    assert len(result) > 100


@pytest.mark.integration
def test_write_references_returns_nonempty():
    result = write_references(SAMPLE_BRIEF, {})
    assert isinstance(result, str)
    assert len(result) > 50


@pytest.mark.integration
def test_write_results_uses_method_dep():
    method = write_method(SAMPLE_BRIEF, {})
    result = write_results(SAMPLE_BRIEF, {"method": method})
    assert isinstance(result, str)
    assert len(result) > 100


@pytest.mark.integration
def test_write_intro_returns_nonempty():
    method = "Participants were randomized to 200mg caffeine or placebo in a double-blind design."
    results = "Caffeine reduced reaction time by 15% (p<0.01)."
    result = write_intro(SAMPLE_BRIEF, {"method": method, "results": results})
    assert isinstance(result, str)
    assert len(result) > 100


@pytest.mark.integration
def test_write_discussion_returns_nonempty():
    results = "Caffeine reduced reaction time by 15% (p<0.01)."
    result = write_discussion(SAMPLE_BRIEF, {"results": results, "figures": ""})
    assert isinstance(result, str)
    assert len(result) > 100


@pytest.mark.integration
def test_write_abstract_returns_nonempty():
    intro = "Caffeine is a widely consumed stimulant."
    discussion = "These findings support caffeine's role in enhancing cognitive performance."
    results = "Caffeine reduced reaction time by 15% (p<0.01)."
    result = write_abstract(SAMPLE_BRIEF, {"intro": intro, "discussion": discussion, "results": results})
    assert isinstance(result, str)
    assert len(result) > 50


from writing.figures import generate_figures


@pytest.mark.integration
def test_generate_figures_returns_tuple():
    results = "Caffeine group: mean 293ms (SD=19). Placebo group: mean 344ms (SD=22). t(38)=8.3, p<0.001."
    figures_md, appendix = generate_figures(SAMPLE_BRIEF, {"results": results})
    assert isinstance(figures_md, str)
    assert isinstance(appendix, str)


@pytest.mark.integration
def test_generate_figures_embeds_image_or_empty():
    results = "Caffeine reduced reaction time by 15% vs placebo."
    figures_md, appendix = generate_figures(SAMPLE_BRIEF, {"results": results})
    assert figures_md == "" or figures_md.startswith("![")


from writing.editors import review_paper, merge_reviews, apply_revisions


def test_merge_reviews_deduplicates():
    r1 = [
        {"section": "results", "issue": "No error bars shown", "suggestion": "Add SD or SE to all plots."},
        {"section": "intro", "issue": "Missing motivation", "suggestion": "Add a sentence on why this matters."},
    ]
    r2 = [
        {"section": "results", "issue": "No error bars shown", "suggestion": "Add error bars."},
        {"section": "discussion", "issue": "No limitations", "suggestion": "Add a limitations paragraph."},
    ]
    merged = merge_reviews(r1, r2)
    sections = [item["section"] for item in merged]
    assert "results" in sections
    assert "intro" in sections
    assert "discussion" in sections
    results_items = [item for item in merged if item["section"] == "results"]
    assert len(results_items) == 1


@pytest.mark.integration
def test_review_paper_returns_list():
    draft = {
        "intro": "Caffeine is a stimulant consumed worldwide.",
        "method": "Participants were randomized to caffeine or placebo.",
        "results": "Caffeine reduced reaction time by 15% (p<0.01).",
        "discussion": "Results support prior literature.",
        "abstract": "We studied caffeine effects on reaction time.",
        "references": "1. Smith 2020. J. Neuroscience.",
    }
    reviews = review_paper("Reviewer 1", draft)
    assert isinstance(reviews, list)


@pytest.mark.integration
def test_apply_revisions_returns_nonempty_sections():
    sections = {
        "results": "Caffeine reduced reaction time by 15%.",
        "intro": "Caffeine is a stimulant.",
    }
    feedback = [
        {"section": "results", "issue": "No statistics", "suggestion": "Add p-value and sample size."},
    ]
    revised = apply_revisions(sections, feedback)
    assert "results" in revised
    assert "intro" in revised
    assert isinstance(revised["results"], str)
    assert len(revised["results"]) > 0


from writing.orchestrator import orchestrate
from hackathon_science.models import Paper


@pytest.mark.integration
def test_orchestrate_returns_valid_paper():
    result = orchestrate(SAMPLE_BRIEF)
    assert isinstance(result, Paper)
    assert len(result.title) > 0
    assert len(result.introduction) > 0
    assert len(result.methods) > 0
    assert len(result.results) > 0


@pytest.mark.integration
def test_orchestrate_with_minimal_brief():
    brief = ResearchBrief(problem_domain="effect of sleep deprivation on memory recall")
    result = orchestrate(brief)
    assert isinstance(result, Paper)
    assert len(result.title) > 0
