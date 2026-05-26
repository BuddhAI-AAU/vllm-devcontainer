from fastapi import APIRouter, Request
import json
import httpx
from fastapi.responses import StreamingResponse
import base64
router = APIRouter()

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "mistralai/Ministral-3-14B-Reasoning-2512"


TTS_URL = "http://localhost:7000/v1/audio/speech"
TTS_MODEL = "mistralai/Voxtral-4B-TTS-2603"
TTS_VOICE = "casual_male"

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

    # Convert full OpenResponses conversation -> prompt
    #messages = build_prompt_from_openresponses(body.get("input", []))
    chat_messages = convert_openresponses_to_chat(body["input"])

    #read parameters from client
    params = body.get("params", {})

    print("\n=== CLIENT PARAMS RECEIVED ===")
    print(json.dumps(params, indent=2))

    #check if TTS is enabled, so we can attempt to use it later.
    tts_enabled = params.get("tts", True)

    #Build the payload from client data for processing through the pipeline, ending up at vllm.
    vllm_payload = {
        "model": params.get("model", MODEL),
        "messages": chat_messages,
        "max_tokens":params.get("max_tokens", 500),
        "temperature":params.get("temperature", 0.7),
        "stream":params.get("stream", True)
    }
    print("\n=== VLLM PAYLOAD SENT TO MODEL - FROM GATEWAY ===")
    print(json.dumps(vllm_payload, indent=2))

    async with httpx.AsyncClient() as client:
        vllm_response = await client.post(VLLM_URL, json=vllm_payload, timeout=None)

    assistant_buffer = ""

    async def event_generator():
        nonlocal assistant_buffer

        async for line in vllm_response.aiter_lines():
            if line.startswith("data: "):
                data = line[len("data: "):]

                if data == "[DONE]":
                    # Now that streaming is finished, generate TTS once
                    if tts_enabled and assistant_buffer.strip():
                        try:
                            tts_payload = {
                                "input": assistant_buffer,   # ✔ final output only
                                "model": TTS_MODEL,
                                "voice": TTS_VOICE,
                                "response_format": "wav",
                        }

                            async with httpx.AsyncClient() as client:
                                tts_resp = await client.post(TTS_URL, json=tts_payload, timeout=20.0)
                                tts_resp.raise_for_status()
                                audio_bytes = tts_resp.content

                            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                            audio_event = json.dumps({"audio_base64": audio_b64})

                            yield f"event: response.audio.speech\ndata: {audio_event}\n\n"

                        except Exception as e:
                            print(f"[TTS ERROR] {e}")

                    yield "event: response.completed\ndata: {}\n\n"
                    break

                chunk = json.loads(data)
                delta_text = chunk["choices"][0]["delta"].get("content", "")
                assistant_buffer += delta_text

                event_data = json.dumps({"delta": delta_text})
                yield f"event: response.output_text.delta\ndata: {event_data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
""""
    async def event_generator():
        nonlocal assistant_buffer

        async for line in vllm_response.aiter_lines():
            if line.startswith("data: "):
                data = line[len("data: "):]

                if data == "[DONE]":
                    yield "event: response.completed\ndata: {}\n\n"
                    break

                chunk = json.loads(data)
                delta_text = chunk["choices"][0]["delta"].get("content", "")

                assistant_buffer += delta_text

                event_data = json.dumps({"delta": delta_text})
                yield f"event: response.output_text.delta\ndata: {event_data}\n\n"


            #this is later xD here handle the TTS logic, like decoding base64, defining payload and posting to client
            if tts_enabled and assistant_buffer.strip():
                try:
                    tts_payload = {
                        "input": assistant_buffer,
                        "model": TTS_MODEL,
                        "voice": TTS_VOICE,
                        "response_format": "wav",
                    }

                    async with httpx.AsyncClient() as client:
                        tts_resp = await client.post(TTS_URL, json=tts_payload, timeout=20.0)
                        tts_resp.raise_for_status()
                        audio_bytes = tts_resp.content

                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                    audio_event = json.dumps({"audio_base64": audio_b64})

                    # separate event for audio so client can choose what to do
                    yield f"event: response.audio.speech\ndata: {audio_event}\n\n"

                except Exception as e:
                    # TTS failed – do nothing, text already delivered
                    print(f"[TTS ERROR] {e}")
"""


    
