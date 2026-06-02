import requests
import json
from services.memory_node import MemoryState, conn
from services.tts_client import synthesize_tts


BASE_URL = "http://localhost:9000/responses"

#group test lynch


def llm_node(state: MemoryState):
    payload = state["payload"]
    user_id = state["user_id"]


    messages = state["payload"]["input"]
    params = state.get("params", {})

    # Build the payload that goes to the gateway
    gateway_payload = {
        "input": messages,
        "params": params
    }

    print("\n=== LLM NODE PAYLOAD SENT TO GATEWAY ===")
    print(json.dumps(gateway_payload, indent=2))

    full_output = ""

    # Stream from gateway
    with requests.post(BASE_URL, json=gateway_payload, stream=True) as r:           #to revert replace json with "payload" like line 11
        event = None

        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue

            #print("RAW STREAM:", raw)

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
    #print("\n=== LLM NODE RAW OUTPUT ===")
    #print(full_output)

    # Store assistant message ONLY if not empty
#    if full_output.strip():
 #       with conn.cursor() as cur:
  #          cur.execute("""
   #             INSERT INTO conversation_memory (user_id, role, content, time_stamp)
    #            VALUES (%s, %s, %s, NOW())
     #       """, (user_id, "assistant", full_output))
      #      conn.commit()

    print("\n=== LLM NODE STORED ASSISTANT MESSAGE ===")
    print(full_output)

        # Remove old messages (keep last 10)
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


    #else:
     #   print("\n=== LLM NODE: EMPTY ASSISTANT MESSAGE SKIPPED ===")



    return {"response": full_output}

# writes turn number after storing messages
def write_turn_node(state: MemoryState):
    user_id = state["user_id"]
    user_input = state["input"]
    assistant_output = state["response"]
    time_stamp = state["time_stamp"]

    # Skip storing empty assistant messages
    if not assistant_output.strip():
        return state

    with conn.cursor() as cur:
        # get last turn_id
        cur.execute("""
            SELECT MAX(turn_id) 
            FROM conversation_memory 
            WHERE user_id = %s
        """, (user_id,))
        last_turn = cur.fetchone()[0] or 0 #retrieves the result of SQL
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
