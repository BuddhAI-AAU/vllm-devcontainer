import psycopg2
from urllib.parse import urlparse
from typing import TypedDict, List, Dict, NotRequired, Any

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
    input: str
    time_stamp: str
    system_prompt: str
    params: dict
    history: List[Dict[str, str]]
    response: NotRequired[str]


# ---------- UTILITIES ----------

def _normalize_to_string(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join([m.get("content", "") for m in x if isinstance(m, dict)])
    if isinstance(x, dict):
        return x.get("content", str(x))
    return str(x)


# ---------- LOAD HISTORY (STREAMING-FRIENDLY) ----------

def load_history(user_id: str, limit: int = 10) -> List[Dict[str, str]]:
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
    return history[-limit:]


# ---------- TRIM HISTORY (KEEP LAST N TURNS) ----------

def trim_history(user_id: str, keep_turns: int = 10) -> None:
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
                    LIMIT %s
                ) AS keepers
            )
            """,
            (user_id, user_id, keep_turns)
        )
        conn.commit()


# ---------- PROMPT BUILDER FOR STREAMING ----------

def build_messages_with_history(
    user_id: str,
    user_input: str,
    system_prompt: str | None = None,
    history_limit: int = 10
) -> List[Dict[str, Any]]:

    history = load_history(user_id, limit=history_limit)
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    messages: List[Dict[str, Any]] = []

    # system prompt
    messages.append({
        "role": "system",
        "content": [
            {"type": "input_text", "text": system_prompt}
        ]
    })

    # history
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": [
                {"type": "input_text", "text": msg["content"]}
            ]
        })

    # latest user message
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": _normalize_to_string(user_input)}
        ]
    })

    return messages


# ---------- WRITE TURN (USER + ASSISTANT) ----------

def write_turn(state: MemoryState) -> MemoryState:
    user_id = state["user_id"]
    user_input_raw = state["input"]
    assistant_output = state.get("response", "")
    time_stamp = state["time_stamp"]

    user_input = _normalize_to_string(user_input_raw)

    if not assistant_output or not assistant_output.strip():
        return state

    with conn.cursor() as cur:
        # compute next turn_id
        cur.execute(
            """
            SELECT MAX(turn_id)
            FROM conversation_memory
            WHERE user_id = %s
            """,
            (user_id,)
        )
        last_turn = cur.fetchone()[0] or 0
        turn_id = last_turn + 1

        # user message
        cur.execute(
            """
            INSERT INTO conversation_memory (user_id, turn_id, role, content, time_stamp)
            VALUES (%s, %s, 'user', %s, %s)
            """,
            (user_id, turn_id, user_input, time_stamp)
        )

        # assistant message
        cur.execute(
            """
            INSERT INTO conversation_memory (user_id, turn_id, role, content, time_stamp)
            VALUES (%s, %s, 'assistant', %s, %s)
            """,
            (user_id, turn_id, assistant_output, time_stamp)
        )

        conn.commit()

    # optional: trim after each write
    trim_history(user_id, keep_turns=10)

    return state
