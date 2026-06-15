import psycopg2
from urllib.parse import urlparse
from typing import TypedDict, List, Dict
from typing import NotRequired

DATABASE_URL = "postgresql://postgres:postgres@postgres:5432/postgres"

parsed = urlparse(DATABASE_URL)

DEFAULT_SYSTEM_PROMPT = (
    "Generate answers with only text, commas and puntuation for natural conversation flow."
    "The answers are for voice generation so do NOT make headline or output symbols like *."
    "You are a voice assistant, do not output * (asterisks), tags or lists."
    "You hate asterisks and would never use them or generate them."
)

DEFAULT_SYSTEM_PROMPT_original = (
    "You are a tutor. Your name is BuddhAi. Base your personality on Buddha"
    "Use gentle Socratic questioning to guide the student, but keep answers clear and factual."
    "With every question, give specific answers that are satisfying, then proceed with socratic questioning"
    "Do not repeat the user's question. Do not roleplay. Do not output XML tags."
)

conn = psycopg2.connect(
    dbname=parsed.path[1:],
    user=parsed.username,
    password=parsed.password,
    host=parsed.hostname,
    port=parsed.port
)
conn.autocommit = True


class MemoryState(TypedDict):
    user_id: str
    input: str | list[dict]
    time_stamp: str
    system_prompt: str
    params: dict
    history: List[Dict[str, str]]
    payload: NotRequired[dict]
    response: NotRequired[str]


def postgres_memory_node(state: MemoryState):
    print("\n=== MEMORY NODE INPUT ===")
    print(state)
    user_id = state["user_id"]
    user_input = state["input"]
    time_stamp = state["time_stamp"]

    # 1. Load history
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT role, content
            FROM conversation_memory
            WHERE user_id = %s
            ORDER BY turn_id ASC, time_stamp ASC
            """,
            (user_id,)
        )
        rows = cur.fetchall()

    history = [{"role": role, "content": content} for role, content in rows]

    print("\n=== MEMORY NODE HISTORY LOADED FROM DB ===")
    print(history)

    # 2. Trim old turns
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM conversation_memory
            WHERE user_id = %s
            AND turn_id NOT IN (
                SELECT turn_id FROM (
                    SELECT DISTINCT turn_id
                    FROM conversation_memory
                    WHERE user_id = %s
                    ORDER BY turn_id DESC
                    LIMIT 10
                ) AS keepers
            )
            """,
            (user_id, user_id)
        )
        conn.commit()

    print("\n=== MEMORY NODE OUTPUT ===")
    print({
        "user_id": user_id,
        "input": user_input,
        "history": history
    })

    return {
        "user_id": user_id,
        "input": user_input,
        "time_stamp": time_stamp,
        "history": history,
    }


def normalize_to_string(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join([m.get("content", "") for m in x])
    return str(x)


def prompt_builder_node(state: MemoryState):
    history = state["history"]
    user_input = state["input"]

    messages = []
    system_prompt = state.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    # System prompt
    messages.append({
        "role": "system",
        "content": [
            {"type": "input_text", "text": system_prompt}
        ]
    })

    # History
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": [
                {"type": "input_text", "text": msg["content"]}
            ]
        })

    # Latest user message
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": normalize_to_string(user_input)}
        ]
    })

    return {"payload": {"input": messages}}

