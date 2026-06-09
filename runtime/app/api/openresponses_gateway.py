from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import json
import httpx
import time
from services.perf_logger import log_perf   # CSV logger

router = APIRouter()

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "mistralai/Ministral-3-14B-Reasoning-2512"

def now():
    return time.perf_counter()

def convert_openresponses_to_chat(messages):
    chat = []
    for msg in messages:
        text_parts = []
        for c in msg["content"]:
            if c["type"] in ("input_text", "output_text"):
                text_parts.append(c["text"])
        chat.append({
            "role": msg["role"],
            "content": "\n".join(text_parts)
        })
    return chat


@router.post("/responses")
async def get_open_responses(request: Request):

    body = await request.json()

    chat_messages = convert_openresponses_to_chat(body["input"])
    params = body.get("params", {})

    # ---- Build STREAMING vLLM payload ----
    vllm_payload = {
        "model": params.get("model", MODEL),
        "messages": chat_messages,
        "max_tokens": params.get("max_tokens", 500),
        "temperature": params.get("temperature", 0.7),
        "stream": True
    }

    async def token_stream():
        t0 = now()

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", VLLM_URL, json=vllm_payload) as resp:

                async for line in resp.aiter_lines():
                    if not line:
                        continue

                    if not line.startswith("data: "):
                        continue

                    data = line[len("data: "):].strip()

                    if data == "[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            # Yield raw token text
                            yield delta
                    except Exception:
                        continue

        # ---- Log streaming duration ----
        t_vllm = now() - t0
        print(f"[TIMING] vLLM_stream: {t_vllm:.3f} sec")
        log_perf("gateway", "vllm_stream", t_vllm)

    # Return streaming text/plain response
    return StreamingResponse(token_stream(), media_type="text/plain")
