from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import httpx
import json

router = APIRouter()

VLLM_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "mistralai/Ministral-3-14B-Reasoning-2512"


@router.post("/v1/chat/completions", response_class=StreamingResponse)
async def openai_chat(request: Request):
    body = await request.json()

    vllm_payload = {
        "model": body.get("model", DEFAULT_MODEL),
        "messages": body["messages"],
        "max_tokens": body.get("max_tokens", 500),
        "temperature": body.get("temperature", 0.7),
        "stream": True
    }

    async with httpx.AsyncClient(timeout=None) as client:
        vllm_response = await client.post(VLLM_URL, json=vllm_payload)

    async def event_generator():
        async for line in vllm_response.aiter_lines():
            if not line or not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except Exception:
                # Skip non‑JSON lines (vLLM sometimes sends headers)
                continue

            # Detect end of stream
            finish = chunk.get("choices", [{}])[0].get("finish_reason")
            if finish == "stop":
                yield "data: [DONE]\n\n"
                return

            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                yield (
                    "data: "
                    + json.dumps({"choices": [{"delta": {"content": delta}}]})
                    + "\n\n"
                )

        # Safety: ensure stream ends cleanly
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
