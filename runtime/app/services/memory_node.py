import psycopg2
from urllib.parse import urlparse
from typing import TypedDict, List, Dict
from typing import NotRequired
from services.longterm_mem import MemoryClient

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
    #memory_client: MemoryClient #EverMemOS
    history: List[Dict[str, str]]
    payload: NotRequired[dict] 
    response: NotRequired[str]
    retrieved_longterm: NotRequired[List[dict]]

def postgres_memory_node(state: MemoryState):

    print("\n=== MEMORY NODE INPUT ===")
    print(state)
    user_id = state["user_id"]
    user_input = state["input"]
    time_stamp = state["time_stamp"]


    # 1. Load history as structured messages        added turn_id in ORDER BY
    with conn.cursor() as cur:
        cur.execute("""
            SELECT role, content FROM conversation_memory WHERE user_id = %s ORDER BY turn_id ASC, time_stamp ASC 

                    """, (user_id,))
        rows = cur.fetchall()

    history = [{"role": role, "content": content} for role, content in rows]

    print("\n=== MEMORY NODE HISTORY LOADED FROM DB ===")
    print(history)


#this deletes user conversations older than x (int in limit), does it pairwise based on turn_id

    with conn.cursor() as cur:
        cur.execute("""
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
        """, (user_id, user_id))
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
        "time_stamp": time_stamp,
        "history": history,
        #"memory_client": state["memory_client"] #EverMemOS
    }


def prompt_builder_node(state: MemoryState):
    history = state["history"]
    user_input = state["input"]
    #longterm = state.get("retrieved_longterm", []) #EverMemOS

    messages = []

    # 1. System prompt
    messages.append({
        "role": "system",
        "content": [
            {"type": "input_text", "text": SYSTEM_PROMPT}
        ]
    })

     # Inject long-term memory
   # if longterm:
  #      mem_text = "\n".join([m["text"] for m in longterm])
   #     messages.append({
   #         "role": "system",
    #        "content": [
     #           {"type": "input_text", "text": f"Relevant memories:\n{mem_text}"}
    #        ]
    #    })

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

#Long term EverMemOS memory
def retrieve_longterm_node(state: MemoryState):
    client = state["memory_client"]

    results = client.retrieve_memory(
        user_id=state["user_id"],
        query=state["input"],
        top_k=5,
    )

    # Normalize EverMemOS structure → list of {"text": "..."}
    normalized = []

    # EverMemOS returns: { "data": { "groups": [ { "memories": [...] } ] } }
    data = results.get("data", {}) if isinstance(results, dict) else {}
    groups = data.get("groups", [])

    for group in groups:
        for mem in group.get("memories", []):
            # EverMemOS uses "content" for message text
            text = (
                mem.get("content")
                or mem.get("text")
                or str(mem)
            )
            normalized.append({"text": text})

    state["retrieved_longterm"] = normalized
    return state

def write_longterm_node(state: MemoryState):
    client = state["memory_client"]
    response = state.get("response")

    # Store assistant responses
    if response and len(response) > 20:
        client.write_memory(
            user_id=state["user_id"],
            text=response,
            metadata={"source": "assistant"}
        )

    return state
