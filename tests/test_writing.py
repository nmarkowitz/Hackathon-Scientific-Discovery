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
