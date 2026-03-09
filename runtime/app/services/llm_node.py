import requests
import json
from services.memory_node import MemoryState, conn

BASE_URL = "http://localhost:9000/responses"


def llm_node(state: MemoryState):
    payload = state["payload"]
    user_id = state["user_id"]

    print("\n=== LLM NODE PAYLOAD SENT TO GATEWAY ===")
    print(json.dumps(payload, indent=2))

    full_output = ""

    # Stream from gateway
    with requests.post(BASE_URL, json=payload, stream=True) as r:
        event = None

        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue

            print("RAW STREAM:", raw)

            if raw.startswith("event:"):
                event = raw.split("event:")[1].strip()

            elif raw.startswith("data:"):
                data = raw.split("data:")[1].strip()

                if event == "response.output_text.delta":
                    delta = json.loads(data)["delta"]
                    full_output += delta

                elif event == "response.completed":
                    break

    # Debug: show final assembled output
    print("\n=== LLM NODE RAW OUTPUT ===")
    print(full_output)

    # Store assistant message ONLY if not empty
    if full_output.strip():
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_memory (user_id, role, content)
                VALUES (%s, %s, %s)
            """, (user_id, "assistant", full_output))
            conn.commit()

        print("\n=== LLM NODE STORED ASSISTANT MESSAGE ===")
        print(full_output)
    else:
        print("\n=== LLM NODE: EMPTY ASSISTANT MESSAGE SKIPPED ===")

    return {"response": full_output}
