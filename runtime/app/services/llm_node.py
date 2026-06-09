import requests
import json
import time
from services.memory_node import MemoryState, conn
from services.perf_logger import log_perf   # <-- CSV logger

BASE_URL = "http://localhost:9000/responses"

def now():
    return time.perf_counter()

def llm_node(state: MemoryState):
    t0 = now()

    payload = state["payload"]
    user_id = state["user_id"]

    raw_input = payload["input"]

    # Normalize input
    if isinstance(raw_input, str):
        messages = [{"role": "user", "content": raw_input}]
    else:
        messages = raw_input

    params = state.get("params", {})
    params["stream"] = True   # <-- IMPORTANT

    gateway_payload = {
        "input": messages,
        "params": params
    }

    print("\n=== LLM NODE PAYLOAD SENT TO GATEWAY ===")
    print(json.dumps(gateway_payload, indent=2))

    # ---- STREAMING REQUEST ----
    full_output = ""

    with requests.post(BASE_URL, json=gateway_payload, stream=True) as r:
        r.raise_for_status()

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue

            # vLLM gateway sends raw text chunks
            full_output += line

    print("\n=== LLM NODE STORED ASSISTANT MESSAGE ===")
    print(full_output)

    # ---- Trim old messages ----
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

    dt = now() - t0
    print(f"[TIMING] Node llm: {dt:.3f} sec")

    # ---- CSV LOGGING ----
    log_perf(
        user_id=user_id,
        stage="llm_node",
        duration=dt,
        extra=f"model={params.get('model')}"
    )

    return {"response": full_output}


# writes turn number after storing messages
from typing import Any

def _normalize_to_string(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join([m.get("content", "") for m in x if isinstance(m, dict)])
    if isinstance(x, dict):
        return x.get("content", str(x))
    return str(x)


def write_turn_node(state: MemoryState):
    t0 = now()

    user_id = state["user_id"]
    user_input_raw = state["input"]
    assistant_output = state["response"]
    time_stamp = state["time_stamp"]

    user_input = _normalize_to_string(user_input_raw)

    if not assistant_output or not assistant_output.strip():
        return state

    with conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(turn_id) 
            FROM conversation_memory 
            WHERE user_id = %s
        """, (user_id,))
        last_turn = cur.fetchone()[0] or 0
        turn_id = last_turn + 1

        cur.execute("""
            INSERT INTO conversation_memory (user_id, turn_id, role, content, time_stamp)
            VALUES (%s, %s, 'user', %s, %s)
        """, (user_id, turn_id, user_input, time_stamp))

        cur.execute("""
            INSERT INTO conversation_memory (user_id, turn_id, role, content, time_stamp)
            VALUES (%s, %s, 'assistant', %s, %s)
        """, (user_id, turn_id, assistant_output, time_stamp))

        conn.commit()

    dt = now() - t0
    print(f"[TIMING] Node write_turn: {dt:.3f} sec")

    log_perf(
        user_id=user_id,
        stage="write_turn",
        duration=dt
    )

    return state
