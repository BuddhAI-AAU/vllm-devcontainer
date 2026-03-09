from fastapi import APIRouter, Request
import json
import httpx
from fastapi.responses import StreamingResponse

router = APIRouter()

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "mistralai/Ministral-3-3B-Reasoning-2512"


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


@router.post("/responses", response_class=StreamingResponse)
async def get_open_responses(request: Request):
    body = await request.json()

    chat_messages = convert_openresponses_to_chat(body["input"])

    vllm_payload = {
        "model": MODEL,
        "messages": chat_messages,
        "max_tokens": 500,
        "temperature": 0.7,
        "stream": True
    }

    async with httpx.AsyncClient() as client:
        vllm_response = await client.post(VLLM_URL, json=vllm_payload, timeout=None)

    async def event_generator():
        async for line in vllm_response.aiter_lines():
            if not line.startswith("data: "):
                continue

            data = line[6:]

            if data == "[DONE]":
                yield "event: response.completed\ndata: {}\n\n"
                break

            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content", "")

            if delta:
                yield (
                    "event: response.output_text.delta\n"
                    f"data: {json.dumps({'delta': delta})}\n\n"
                )

    return StreamingResponse(event_generator(), media_type="text/event-stream")
