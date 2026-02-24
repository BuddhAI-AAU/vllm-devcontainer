from fastapi import APIRouter, Request
import json
import httpx
from fastapi.responses import StreamingResponse

router = APIRouter()

VLLM_URL = "http://localhost:8000/v1/completions"
MODEL = "nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4"


def build_prompt_from_openresponses(input_list):
    prompt = ""
    for item in input_list:
        role = item["role"]
        for content in item["content"]:
            if content["type"] == "input_text":
                prompt += f"{role}: {content['text']}\n"
            elif content["type"] == "output_text":
                prompt += f"{role}: {content['text']}\n"
    return prompt


@router.post("/responses", response_class=StreamingResponse)
async def get_open_responses(request: Request):
    body = await request.json()

    # Convert full OpenResponses conversation → prompt
    prompt = build_prompt_from_openresponses(body.get("input", []))

    vllm_payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.7,
        "stream": True
    }

    async with httpx.AsyncClient() as client:
        vllm_response = await client.post(VLLM_URL, json=vllm_payload, timeout=None)

    assistant_buffer = ""

    async def event_generator():
        nonlocal assistant_buffer

        async for line in vllm_response.aiter_lines():
            if line.startswith("data: "):
                data = line[len("data: "):]

                if data == "[DONE]":
                    yield "event: response.completed\ndata: {}\n\n"
                    break

                chunk = json.loads(data)
                delta_text = chunk["choices"][0]["text"]

                assistant_buffer += delta_text

                event_data = json.dumps({"delta": delta_text})
                yield f"event: response.output_text.delta\ndata: {event_data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
