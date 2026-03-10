import psycopg2
from urllib.parse import urlparse
from typing import TypedDict, List, Dict
from typing import NotRequired
DATABASE_URL = "postgresql://postgres:postgres@postgres:5432/postgres"

parsed = urlparse(DATABASE_URL)
SYSTEM_PROMPT =    ("You are a tutor. Your name is BuddhAi. Base your personality on Buddha" 
                    "Use gentle Socratic questioning to guide the student, but keep answers clear and factual." 
                    "With every question, give specific answers that are satisfying, then proceed with socratic questioning"
                    "Do not repeat the user's question. Do not roleplay. Do not output XML tags.")


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
    history: List[Dict[str, str]]
    payload: NotRequired[dict] 
    response: NotRequired[str]

def postgres_memory_node(state: MemoryState):

    print("\n=== MEMORY NODE INPUT ===")
    print(state)
    user_id = state["user_id"]
    user_input = state["input"]
    time_stamp = state["time_stamp"]


    # 1. Load history as structured messages
    with conn.cursor() as cur:
        cur.execute("""
            SELECT role, content FROM conversation_memory WHERE user_id = %s ORDER BY time_stamp ASC

                    """, (user_id,))
        rows = cur.fetchall()

    history = [{"role": role, "content": content} for role, content in rows]

    print("\n=== MEMORY NODE HISTORY LOADED FROM DB ===")
    print(history)

    # 2. Append the new user message
    history.append({"role": "user", "content": user_input})
    
    # 3. Save the new message/insertion into SQL
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO conversation_memory (user_id, role, content, time_stamp)
            VALUES (%s, %s, %s, %s)
        """, (user_id, "user", user_input, time_stamp))
        conn.commit()

    print("\n=== MEMORY NODE OUTPUT ===")
    print({
    "user_id": user_id,
    "input": user_input,
    "history": history
    })

    # 4. Return structured history
    return {
        "user_id": user_id,
        "input": user_input,
        "history": history
    }


def prompt_builder_node(state: MemoryState):
    history = state["history"]
    user_input = state["input"]

    messages = []

    # 1. System prompt
    messages.append({
        "role": "system",
        "content": [
            {"type": "input_text", "text": SYSTEM_PROMPT}
        ]
    })

    # 2. Add each message from history
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": [
                {"type": "input_text", "text": msg["content"]}
            ]
        })

    # 3. Add latest user message (again, as the final message)
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": user_input}
        ]
    })

    return {"payload": {"input": messages}}

