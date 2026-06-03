import requests
import json
from services.memory_node import MemoryState, conn
from services.tts_client import synthesize_tts


BASE_URL = "http://localhost:9000/responses"

#group test lynch

def llm_node(state: MemoryState):
    payload = state["payload"]
    user_id = state["user_id"]

    raw_input = payload["input"]

    # Normalize input
    if isinstance(raw_input, str):
        messages = [{"role": "user", "content": raw_input}]
    else:
        messages = raw_input

    params = state.get("params", {})
    params["stream"] = False

    gateway_payload = {
        "input": messages,
        "params": params
    }

    print("\n=== LLM NODE PAYLOAD SENT TO GATEWAY ===")
    print(json.dumps(gateway_payload, indent=2))

    r = requests.post(BASE_URL, json=gateway_payload, timeout=120)
    r.raise_for_status()

    data = r.json()
    full_output = data.get("response", "")

    print("\n=== LLM NODE STORED ASSISTANT MESSAGE ===")
    print(full_output)

    # Trim old messages
    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM conversation_memory
            WHERE user_id = %s
            AND id NOT IN (
                SELECT id
                FROM conversation_memory
                WHERE user_id = %s
                ORDER BY id DESC
                LIMIT 10
            )
        """, (user_id, user_id))
        conn.commit()

    return {"response": full_output}


# writes turn number after storing messages
from typing import Any

def _normalize_to_string(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        # list of {"role": "...", "content": "..."}
        return "\n".join([m.get("content", "") for m in x if isinstance(m, dict)])
    if isinstance(x, dict):
        # single {"role": "...", "content": "..."} case
        return x.get("content", str(x))
    return str(x)


def write_turn_node(state: MemoryState):
    user_id = state["user_id"]
    user_input_raw = state["input"]
    assistant_output = state["response"]
    time_stamp = state["time_stamp"]

    # normalize input to plain text for storage
    user_input = _normalize_to_string(user_input_raw)

    # Skip storing empty assistant messages
    if not assistant_output or not assistant_output.strip():
        return state

    with conn.cursor() as cur:
        # get last turn_id
        cur.execute("""
            SELECT MAX(turn_id) 
            FROM conversation_memory 
            WHERE user_id = %s
        """, (user_id,))
        last_turn = cur.fetchone()[0] or 0
        turn_id = last_turn + 1

        # store user message
        cur.execute("""
            INSERT INTO conversation_memory (user_id, turn_id, role, content, time_stamp)
            VALUES (%s, %s, 'user', %s, %s)
        """, (user_id, turn_id, user_input, time_stamp))

        # store assistant message
        cur.execute("""
            INSERT INTO conversation_memory (user_id, turn_id, role, content, time_stamp)
            VALUES (%s, %s, 'assistant', %s, %s)
        """, (user_id, turn_id, assistant_output, time_stamp))

        conn.commit()

    return state

#deprecated
"""
def tts_node(state):
    text = state["response"]
    audio_bytes = synthesize_tts(text)
    state["tts_audio"] = audio_bytes
    return state
"""
