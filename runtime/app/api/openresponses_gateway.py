from fastapi import APIRouter, Request
import json
import httpx
import base64
import time
from services.perf_logger import log_perf   # <-- CSV logger

router = APIRouter()

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "mistralai/Ministral-3-14B-Reasoning-2512"

TTS_URL = "http://localhost:7000/v1/audio/speech"
TTS_MODEL = "mistralai/Voxtral-4B-TTS-2603"
TTS_VOICE = "casual_male"

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
    t_gateway_start = now()

    body = await request.json()

    chat_messages = convert_openresponses_to_chat(body["input"])
    params = body.get("params", {})
    tts_enabled = params.get("tts", True)

    # ---- Build non-streaming vLLM payload ----
    vllm_payload = {
        "model": params.get("model", MODEL),
        "messages": chat_messages,
        "max_tokens": params.get("max_tokens", 500),
        "temperature": params.get("temperature", 0.7),
        "stream": False
    }

    # ---- vLLM timing ----
    t0 = now()
    async with httpx.AsyncClient() as client:
        vllm_response = await client.post(VLLM_URL, json=vllm_payload, timeout=120)
    t_vllm = now() - t0

    print(f"[TIMING] vLLM: {t_vllm:.3f} sec")
    log_perf(
        user_id="gateway",
        stage="vllm",
        duration=t_vllm,
        extra=f"model={vllm_payload['model']}"
    )

    print("GATEWAY RECEIVED REQUEST")

    # ---- Extract final text safely ----
    data = vllm_response.json()
    choices = data.get("choices", [])
    if not choices:
        final_text = ""
    else:
        final_text = choices[0]["message"]["content"]

    # ---- TTS ----
    audio_b64 = None
    t_tts = 0.0

    if tts_enabled and final_text.strip():
        try:
            t1 = now()
            tts_payload = {
                "input": final_text,
                "model": TTS_MODEL,
                "voice": TTS_VOICE,
                "response_format": "wav",
            }
            async with httpx.AsyncClient() as client:
                tts_resp = await client.post(TTS_URL, json=tts_payload, timeout=20.0)
                tts_resp.raise_for_status()
                audio_b64 = base64.b64encode(tts_resp.content).decode("utf-8")
            t_tts = now() - t1

            print(f"[TIMING] TTS: {t_tts:.3f} sec")
            log_perf(
                user_id="gateway",
                stage="tts",
                duration=t_tts,
                extra=f"voice={TTS_VOICE}"
            )

        except Exception as e:
            print("[TTS ERROR]", e)

    # ---- Total gateway time ----
    t_gateway_total = now() - t_gateway_start
    print(f"[TIMING] Gateway Total: {t_gateway_total:.3f} sec\n")

    log_perf(
        user_id="gateway",
        stage="gateway_total",
        duration=t_gateway_total
    )

    return {
        "response": final_text,
        "audio_base64": audio_b64
    }
