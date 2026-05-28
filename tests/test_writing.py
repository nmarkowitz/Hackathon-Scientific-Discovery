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
