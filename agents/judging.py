import json
import re
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_SCORES = {"tech": 5, "novelty": 5, "clarity": 5, "significance": 5}
DEFAULT_CRITIQUE = {"tech": "", "novelty": "", "clarity": "", "significance": ""}

MODE_CONFIG = {
    "paper": {
        "tone": "rigorous, adversarial, evidence-based",
        "review_goal": "Evaluate a research paper as a skeptical reviewer.",
        "score_bias": "Be conservative. Lower scores when validation is thin or claims exceed evidence.",
        "focus": [
            "technical rigor, experimental design, baselines, ablations, statistical testing, reproducibility",
            "novelty relative to prior work and whether it is an adaptation versus a new method",
            "clarity, precision, organization, missing figures/tables, and terminology consistency",
            "significance, practical or scientific impact, and whether claims are supported",
        ],
    },
    "code": {
        "tone": "strict implementation audit",
        "review_goal": "Evaluate a code artifact as a skeptical code reviewer.",
        "score_bias": "Be conservative. If there is no runnable code, no environment, or no tests, score very low.",
        "focus": [
            "code technical quality, architecture, style, safety, maintainability, and completeness",
            "reproducibility, environment setup, dependency management, seeds, scripts, and determinism",
            "correctness, edge cases, runtime behavior, and test coverage",
            "alignment between code and the paper or stated method",
        ],
    },
    "extension": {
        "tone": "comparison-focused extension reviewer",
        "review_goal": "Evaluate how strong this work is as an extension of prior work.",
        "score_bias": "Judge whether the extension adds real method, evidence, or scope beyond the reference.",
        "focus": [
            "how much new methodology is added beyond the reference paper",
            "whether the extension resolves prior limitations or just reuses the same template",
            "new domain or new setting value versus true technical advance",
            "whether evidence is stronger, broader, or better controlled than before",
        ],
    },
    "strict-checklist": {
        "tone": "hard-nosed checklist auditor",
        "review_goal": "Apply a conservative checklist and flag every weakness.",
        "score_bias": "Prefer lower scores unless the paper is clearly strong on the checklist item.",
        "focus": [
            "missing baselines, missing ablations, missing stats, missing replication details",
            "overclaiming, vague impact claims, and unsupported novelty claims",
            "clarity gaps, inconsistent terminology, and weak presentation",
            "limitations, scope bounds, and whether the verdict matches the evidence",
        ],
    },
}

SYSTEM_PROMPT_TEMPLATE = """You are a rigorous, adversarial peer reviewer.
Your job is to find weaknesses, not praise strengths.

Review mode: {review_mode}
Tone: {tone}
Goal: {review_goal}
Score bias: {score_bias}

Use the following focus points:
{focus_points}

Score each dimension 1-10.
Calibration:
- 1-4 = serious flaws
- 5-6 = mediocre
- 7-8 = solid
- 9-10 = exceptional

Anti-inflation rules:
- No ablations -> tech <= 6
- No prior work comparison -> novelty <= 5
- Vague impact claims -> significance <= 5
- When uncertain, score lower not higher

Return ONLY valid JSON, no markdown fences, matching this schema exactly:
{
  "scores": {"tech": int, "novelty": int, "clarity": int, "significance": int},
  "critique": {"tech": str, "novelty": str, "clarity": str, "significance": str},
  "suggestions": [str, str, str],
  "flags": [str]
}
"""

SECTION_PROMPT_TEMPLATE = """You are writing a detailed review section.
Be specific, skeptical, and evidence-based.
Use formal reviewer language.
Do not invent facts.

Section: {section_name}
Score: {score}/10
Flags: {flags}
Critique notes: {critique}

Write 2-4 sentences that sound like a serious research review.
"""

FINAL_VERDICT_PROMPT = """You are writing an overall verdict for a research paper review.
Synthesize the main weaknesses and strengths without exaggeration.
Use a formal reviewer tone.
Keep it to 3-5 sentences.

Scores: {scores}
Flags: {flags}
Critique summary: {critique_summary}
"""

SECTION_NAMES = ["technical_quality", "novelty", "clarity", "significance"]
SECTION_LABELS = {
    "technical_quality": "Technical Quality",
    "novelty": "Novelty",
    "clarity": "Clarity",
    "significance": "Significance",
}

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _default_result() -> Dict[str, Any]:
    return {
        "scores": DEFAULT_SCORES.copy(),
        "critique": DEFAULT_CRITIQUE.copy(),
        "suggestions": [],
        "flags": [],
    }


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _default_result()
    return parsed if isinstance(parsed, dict) else _default_result()


def _coerce_int_score(value: Any, default: int = 5) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0, min(10, int(round(value))))
    return default


def _coerce_review_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = _default_result()

    scores = payload.get("scores", {})
    if isinstance(scores, dict):
        for k in DEFAULT_SCORES:
            result["scores"][k] = _coerce_int_score(scores.get(k, DEFAULT_SCORES[k]), DEFAULT_SCORES[k])

    critique = payload.get("critique", {})
    if isinstance(critique, dict):
        for k in DEFAULT_CRITIQUE:
            v = critique.get(k, "")
            result["critique"][k] = v if isinstance(v, str) else ""

    suggestions = payload.get("suggestions", [])
    if isinstance(suggestions, list):
        result["suggestions"] = [s for s in suggestions if isinstance(s, str)][:3]

    flags = payload.get("flags", [])
    if isinstance(flags, list):
        result["flags"] = [f for f in flags if isinstance(f, str)][:10]

    return result


def _overall_score(scores: Dict[str, int]) -> float:
    return round(
        0.4 * scores.get("tech", 5)
        + 0.3 * scores.get("novelty", 5)
        + 0.15 * scores.get("clarity", 5)
        + 0.15 * scores.get("significance", 5),
        2,
    )


def _weakest_dim(scores: Dict[str, int]) -> str:
    allowed = {k: scores.get(k, 5) for k in DEFAULT_SCORES}
    return min(allowed, key=allowed.get)


def _make_system_prompt(review_mode: str) -> str:
    cfg = MODE_CONFIG.get(review_mode, MODE_CONFIG["paper"])
    focus_points = "\n".join(f"- {x}" for x in cfg["focus"])
    return SYSTEM_PROMPT_TEMPLATE.format(
        review_mode=review_mode,
        tone=cfg["tone"],
        review_goal=cfg["review_goal"],
        score_bias=cfg["score_bias"],
        focus_points=focus_points,
    )


def _call_llm_text(prompt: str, model_id: str) -> str:
    response = call_llm(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        model_id=model_id,
    )
    output = response.get("output", {}).get("message", {}).get("content", [])
    return output.get("text", "") if output else ""


def extract_sections(text: str, fallback_title: str = "") -> Dict[str, str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()

    headings = []
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip().lower()
            headings.append((i, level, title))

    sections = {"title": fallback_title, "introduction": "", "methods": "", "results": "", "references": ""}

    if not headings:
        sections["introduction"] = text.strip()
        return sections

    preamble = "\n".join(lines[:headings]).strip()
    if preamble and not sections["title"]:
        sections["title"] = preamble.splitlines().strip()

    wanted = {
        "introduction": ["introduction", "intro"],
        "methods": ["methods", "method"],
        "results": ["results"],
        "references": ["references", "reference", "bibliography"],
    }

    for idx, (start_i, _, htitle) in enumerate(headings):
        end_i = headings[idx + 1] if idx + 1 < len(headings) else len(lines)
        content = "\n".join(lines[start_i + 1:end_i]).strip()
        for key, aliases in wanted.items():
            if any(htitle == a or htitle.startswith(a + " ") for a in aliases):
                if not sections[key]:
                    sections[key] = content
                break

    if not sections["introduction"]:
        sections["introduction"] = text.strip()

    return sections


def _paper_text(paper_dict: Dict[str, Any], problem_domain: str) -> str:
    return f"""PROBLEM DOMAIN: {problem_domain}

TITLE: {paper_dict.get('title', '')}

INTRODUCTION:
{paper_dict.get('introduction', '')}

METHODS:
{paper_dict.get('methods', '')}

RESULTS:
{paper_dict.get('results', '')}

REFERENCES:
{paper_dict.get('references', '')}
"""


def self_review(paper_dict: Dict[str, Any], problem_domain: str, review_mode: str = "paper") -> Dict[str, Any]:
    model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    paper_text = _paper_text(paper_dict, problem_domain)
    system_prompt = _make_system_prompt(review_mode)
    raw = _call_llm_text(system_prompt + "\n\n---\n\n" + paper_text, model_id)
    parsed = _safe_json_loads(raw)
    result = _coerce_review_payload(parsed)
    result["overall"] = _overall_score(result["scores"])
    result["weakest_dim"] = _weakest_dim(result["scores"])
    return result


def _section_prompt(section_name: str, score: int, critique: str, flags: List[str]) -> str:
    return SECTION_PROMPT_TEMPLATE.format(
        section_name=SECTION_LABELS.get(section_name, section_name),
        score=score,
        critique=critique,
        flags=", ".join(flags) if flags else "none",
    )


def _final_verdict_prompt(scores: Dict[str, int], flags: List[str], critique_summary: str) -> str:
    return FINAL_VERDICT_PROMPT.format(
        scores=json.dumps(scores, ensure_ascii=False),
        flags=", ".join(flags) if flags else "none",
        critique_summary=critique_summary,
    )


def render_review(review: Dict[str, Any], model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0") -> Dict[str, Any]:
    flags = review.get("flags", []) if isinstance(review.get("flags", []), list) else []
    critique = review.get("critique", {}) if isinstance(review.get("critique", {}), dict) else {}
    scores = review.get("scores", {}) if isinstance(review.get("scores", {}), dict) else DEFAULT_SCORES.copy()

    sections = {}
    for key in SECTION_NAMES:
        score_key = key if key != "technical_quality" else "tech"
        prompt = _section_prompt(key, scores.get(score_key, 5), critique.get(score_key, ""), flags)
        text = _call_llm_text(prompt, model_id).strip()
        sections[key] = {"score": scores.get(score_key, 5), "text": text}

    critique_summary = " ".join([
        critique.get("tech", ""),
        critique.get("novelty", ""),
        critique.get("clarity", ""),
        critique.get("significance", ""),
    ]).strip()

    verdict = _call_llm_text(_final_verdict_prompt(scores, flags, critique_summary), model_id).strip()

    return {
        "sections": sections,
        "overall": review.get("overall", _overall_score(scores)),
        "weakest_dim": review.get("weakest_dim", _weakest_dim(scores)),
        "flags": flags,
        "overall_verdict": verdict,
    }


def review_paper(paper_dict: Dict[str, Any], problem_domain: str, review_mode: str = "paper") -> Dict[str, Any]:
    scored = self_review(paper_dict, problem_domain, review_mode=review_mode)
    rendered = render_review(scored)
    rendered["scores"] = scored["scores"]
    rendered["critique"] = scored["critique"]
    rendered["suggestions"] = scored["suggestions"]
    return rendered


def review_folder_to_json_files(
    input_folder: str,
    output_folder: str,
    problem_domain: str,
    review_mode: str = "paper",
    glob_pattern: str = "*.md",
) -> List[Dict[str, Any]]:
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for file_path in sorted(in_dir.glob(glob_pattern)):
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        paper_dict = extract_sections(raw, fallback_title=file_path.stem)
        review = review_paper(paper_dict, problem_domain, review_mode=review_mode)
        review["file"] = file_path.name

        out_path = out_dir / f"{file_path.stem}.json"
        out_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")

        results.append(review)

    return results


def main():
    input_folder = "papers"
    output_folder = "reviews"
    problem_domain = "scientific hypothesis generation"

    review_folder_to_json_files(
        input_folder=input_folder,
        output_folder=output_folder,
        problem_domain=problem_domain,
        review_mode="paper",
        glob_pattern="*.md",
    )


if __name__ == "__main__":
    main()