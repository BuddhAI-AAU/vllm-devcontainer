#What this file does is accept OpenResponses, extracts user text, call vLLM and streams back open responses events
from fastapi import APIRouter, Request
import json
import httpx
from fastapi.responses import StreamingResponse
from services.conversation_store import JsonConversationSore 

router = APIRouter()

VLLM_URL = "http://localhost:8000/v1/completions"
MODEL = "nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4"

store = JsonConversationSore()

@router.post("/responses", response_class=StreamingResponse)    #opening endpoint for OpenResponses
async def get_open_responses(request: Request):                 #Define async function to handle incoming requests
    body = await request.json()                                 #Parse incoming JSON request body

    conversation_id = body.get("conversation_id")               #load or create conversation
    conversation_id = store.check_conv(conversation_id) 

    user_input = ""                                             #Initialize empty string for user input
    
    # Extract user text from OpenResponses format 
    # - Makes a single string prompt to send to vLLM containing all user inputs, stores in user_input
    for item in body.get("input", []):
        if item.get("role") == "user":
            for content in item.get("content", []):
                if content.get("type") == "input_text":
                    user_input += content.get("text", "") + "\n"


    store.add_message(conversation_id, "user", user_input)      #save user_input

    #The payload to send to vLLM
    vllm_payload = {
        "model": MODEL,
        "prompt": user_input,
        "max_tokens": 500,
        "temperature": 0.7,
        "stream": True
    }

    async with httpx.AsyncClient() as client:
        vllm_response = await client.post(VLLM_URL, json=vllm_payload, timeout=None)

    assistant_buffer = ""

    async def event_generator():
        nonlocal assistant_buffer

        # Send conversation_id to client
        yield f"event: conversation_id\ndata: {conversation_id}\n\n"

        async for line in vllm_response.aiter_lines():
            if line.startswith("data: "):
                data = line[len("data: "):]

                if data == "[DONE]":
                    # Save assistant message
                    store.add_message(conversation_id, "assistant", assistant_buffer)

                    yield "event: response.completed\ndata: {}\n\n"
                    break

                else:
                    chunk = json.loads(data)
                    delta_text = chunk["choices"][0]["text"]

                    # Accumulate streamed text
                    assistant_buffer += delta_text

                    event_data = json.dumps({"delta": delta_text})
                    yield f"event: response.output_text.delta\ndata: {event_data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

""""
        async def event_generator():
            async for line in vllm_response.aiter_lines():
                if line.startswith("data: "):
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        yield f"event: response.completed\ndata: {{}}\n\n"
                        break
                    else:
                        chunk = json.loads(data)
                        delta_text = chunk["choices"][0]["text"]
                        event_data = json.dumps({"delta": delta_text})
                        yield f"event: response.output_text.delta\ndata: {event_data}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
"""