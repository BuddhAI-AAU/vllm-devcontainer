from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import json

app = FastAPI(title="LangGraph → LLM Direct Gateway")

VLLM_URL = "http://wonderful_yonath:9000/v1/chat/completions"
DEFAULT_MODEL = "mistralai/Ministral-3-14B-Reasoning-2512"


@app.post("/chat/completions", response_class=StreamingResponse)
async def chat_endpoint(request: Request):
    body = await request.json()

    # Detect payload type
    if "message" in body:
        # LangGraph payload
        user_message = body["message"]
        system_prompt = body.get("system_prompt", "")
        params = body.get("params", {})

        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.append({"role": "user", "content": user_message})

    elif "input" in body:
        # EverMem payload
        chat_messages = []
        for msg in body["input"]:
            text_parts = []
            for c in msg["content"]:
                if c["type"] in ("input_text", "output_text"):
                    text_parts.append(c["text"])
            chat_messages.append({
                "role": msg["role"],
                "content": "\n".join(text_parts)
            })

        params = body.get("params", {})

    else:
        raise ValueError("Unsupported payload format: expected 'message' or 'input'")

    # Build vLLM payload
    vllm_payload = {
        "model": params.get("model", DEFAULT_MODEL),
        "messages": chat_messages,
        "max_tokens": params.get("max_tokens", 500),
        "temperature": params.get("temperature", 0.7),
        "stream": True
    }

    async with httpx.AsyncClient(timeout=None) as client:
        vllm_response = await client.post(VLLM_URL, json=vllm_payload)

    async def event_generator():
        async for line in vllm_response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue

            data = line[len("data: "):]

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
