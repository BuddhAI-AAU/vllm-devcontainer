import psycopg2
from urllib.parse import urlparse
from typing import TypedDict, Optional

DATABASE_URL = "postgresql://postgres:postgres@postgres:5432/postgres"



parsed = urlparse(DATABASE_URL)

conn = psycopg2.connect(
    dbname=parsed.path[1:],   # remove leading slash
    user=parsed.username,
    password=parsed.password,
    host=parsed.hostname,
    port=parsed.port
)
conn.autocommit = True

class MemoryState(TypedDict):
     user_id: str 
     input: str 
     history: Optional[str] 
     payload: Optional[dict] 
     response: Optional[str]

def postgres_memory_node(state: MemoryState) -> MemoryState:
    user_id = state["user_id"]
    new_input = state["input"]

    with conn.cursor() as cur:
        # Load previous history
        cur.execute(
            "SELECT history FROM conversation_memory WHERE user_id = %s",
            (user_id,)
        )
        row = cur.fetchone()
        prev_history = row[0] if row else ""

        # Append new input
        updated_history = prev_history + "\n" + new_input

        # Save updated history
        cur.execute("""
            INSERT INTO conversation_memory (user_id, history)
            VALUES (%s, %s)
            ON CONFLICT (user_id)
            DO UPDATE SET history = EXCLUDED.history
        """, (user_id, updated_history))

    return {
        "user_id": user_id,
        "input": new_input,
        "history": updated_history
    }


def prompt_builder_node(state: MemoryState):
    history = state["history"] or ""
    user_input = state["input"]

    # System message containing memory
    system_message = {
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": f"Conversation so far:\n{history}"
            }
        ]
    }

    # User message (latest input)
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": user_input
            }
        ]
    }

    # This matches EXACTLY what your client sends
    payload = {
        "input": [
            system_message,
            user_message
        ]
    }

    return {"payload": payload}
