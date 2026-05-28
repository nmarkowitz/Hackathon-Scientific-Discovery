from hackathon_science.utils import call_llm

MODEL_ID = "global.anthropic.claude-sonnet-4-6"


def llm_call(system: str, user: str) -> str:
    messages = [{"role": "user", "content": [{"text": user}]}]
    try:
        response = call_llm(
            messages=messages,
            model_id=MODEL_ID,
            system=[{"text": system}],
        )
        content = response.get("output", {}).get("message", {}).get("content", [])
        return content[0].get("text", "") if content else ""
    except Exception:
        return ""
