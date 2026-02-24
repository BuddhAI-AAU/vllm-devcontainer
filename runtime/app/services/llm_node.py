import requests
import json
from services.memory_node import MemoryState

BASE_URL = "http://localhost:9000/responses"


def llm_node(state: MemoryState):
    payload = state["payload"]

    full_output = ""

    with requests.post(BASE_URL, json=payload, stream=True) as r:
        event = None

        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue

            if raw.startswith("event:"):
                event = raw.split("event:")[1].strip()

            elif raw.startswith("data:"):
                data = raw.split("data:")[1].strip()

                if event == "response.output_text.delta":
                    delta = json.loads(data)["delta"]
                    full_output += delta

                elif event == "response.completed":
                    break

    return {"response": full_output}
