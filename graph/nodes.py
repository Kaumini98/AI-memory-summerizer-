import json

from graph.persistence import save_memories
from graph.llm import DEEPSEEK_MODEL, get_deepseek_client
from graph.state import SummaryState

MEMORY_CATEGORIES = {"people", "events", "preferences", "important_facts"}

def summarize_paragraph(state: SummaryState) -> dict[str, str]:
    if not state.get("paragraph", "").strip():
        return {"summary": ""}

    length_guidance = {
        "Short": "Write a concise summary in 1 or 2 sentences.",
        "Medium": "Write a clear summary in 3 or 4 sentences.",
        "Detailed": "Write a thorough summary in 5 to 7 sentences, retaining the key details.",
    }
    length = state.get("summary_length", "Medium")
    instruction = length_guidance.get(length, length_guidance["Medium"])
    prompt = (
        f"{instruction} Use clear, simple language and preserve the main ideas.\n\n"
        f"Paragraph:\n{state['paragraph']}"
    )

    client = get_deepseek_client()
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    return {"summary": response.choices[0].message.content or ""}


def extract_memories(state: SummaryState) -> dict[str, list[dict[str, str]]]:
    """Extract useful facts from the original paragraph using the same DeepSeek API."""
    paragraph = state.get("paragraph", "")
    if not paragraph.strip():
        return {"memories": []}

    try:
        client = get_deepseek_client()
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract only useful information explicitly stated in the text. "
                        "Return valid JSON only as an object with a memories array. "
                        "Each array item must have string keys content and category. "
                        "Choose exactly one category for each item using these rules: "
                        "people is for personal identity and details about a person, "
                        "especially the user's name (for example, 'My name is John' "
                        "must be category people); events is for things that happened "
                        "or are planned; preferences is for stated likes, dislikes, "
                        "or choices; important_facts is for other useful facts such as "
                        "a person's job, organization, or stable circumstances. "
                        "Do not put names or personal identity details under "
                        "important_facts. Allowed category values are exactly people, "
                        "events, preferences, and important_facts. Do not invent facts. "
                        "Return an empty memories array if nothing is useful."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Extract memories from this paragraph:\n\n{paragraph}",
                },
            ],
            stream=False,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        parsed = json.loads(content)

        # Accept either the instructed array or a JSON object wrapping that array.
        if isinstance(parsed, dict):
            parsed = parsed.get("memories", [])
        if not isinstance(parsed, list):
            return {"memories": []}

        memories = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            text = item.get("content")
            category = item.get("category")
            if (
                isinstance(text, str)
                and text.strip()
                and isinstance(category, str)
                and category in MEMORY_CATEGORIES
            ):
                memories.append({"content": text.strip(), "category": category})
        return {"memories": memories}
    except Exception:
        # Extraction is optional: malformed output or an API failure should not
        # prevent the existing summary node from returning its result.
        return {"memories": []}


def persist_memories(state: SummaryState) -> dict[str, str | int]:
    """Persist extracted memories without allowing storage errors to fail the graph."""
    memories = state.get("memories", [])
    if not memories:
        return {
            "memory_storage_status": "skipped",
            "memory_storage_warning": "",
            "memory_storage_count": 0,
        }

    try:
        saved_count = save_memories(memories)
        return {
            "memory_storage_status": "saved",
            "memory_storage_warning": "",
            "memory_storage_count": saved_count,
        }
    except Exception as exc:
        # Include only the exception type; database exceptions can contain the URL.
        return {
            "memory_storage_status": "failed",
            "memory_storage_count": 0,
            "memory_storage_warning": (
                "Could not save the extracted memories to PostgreSQL "
                f"({type(exc).__name__}). Check that PostgreSQL is running and "
                "DATABASE_URL and USER_ID are configured. Your summary and "
                "extracted memories are still available. Retrying is safe."
            ),
        }
