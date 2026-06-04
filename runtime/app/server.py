from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from services.langgraph_node import run_chat
from services.perf_logger import log_perf   # <-- CSV logger
import base64
import httpx
from api.openresponses_gateway import router as responses_router
import io
import wave
import uvicorn
import asyncio
import time

app = FastAPI(title="LangGraph + vLLM API")
app.include_router(responses_router)

TTS_URL = "http://localhost:7000/v1/audio/speech"
WHISPER_URL = "http://localhost:7001/transcribe"

audio_buffers: dict[str, list[bytes]] = {}

#--------------Latency report------------

def now():
    return time.perf_counter()

# ---------------- Models ----------------

class ChatRequest(BaseModel):
    user_id: str
    time_stamp: str
    message: str | None = ""
    system_prompt: str
    params: dict
    input_type: str
    tts_enabled: bool = True
    audio_base64: str | None = None

class ChatResponse(BaseModel):
    response: str
    audio_base64: str | None = None

# ---------------- TTS -------------------

def run_tts(text: str) -> str | None:
    t0 = now()
    try:
        payload = {
            "input": text,
            "model": "mistralai/Voxtral-4B-TTS-2603",
            "voice": "casual_male",
            "response_format": "wav"
        }
        resp = httpx.post(TTS_URL, json=payload, timeout=120.0)
        resp.raise_for_status()
        audio = base64.b64encode(resp.content).decode("utf-8")
        t_tts = now() - t0
        print(f"[TIMING] TTS: {t_tts:.3f} sec")

        # CSV log
        log_perf("server", "tts", t_tts)

        return audio
    except Exception as e:
        print("[TTS ERROR]", e)
        return None

# ---------------- STT -------------------

async def run_stt(audio_bytes: bytes) -> str:
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)

    wav_buf.seek(0)
    files = {"file": ("audio.wav", wav_buf.read(), "audio/wav")}

    t0 = now()
    async with httpx.AsyncClient() as client:
        resp = await client.post(WHISPER_URL, files=files, timeout=60.0)
        resp.raise_for_status()
        text = resp.json().get("text", "")
    t_stt = now() - t0

    print(f"[TIMING] STT: {t_stt:.3f} sec")
    log_perf("server", "stt", t_stt)

    return text

# ---------------- Text Handler -------------------

def handler_text_input(payload: ChatRequest) -> ChatResponse:
    t0 = now()
    output = run_chat(
        payload.user_id,
        payload.message,
        payload.time_stamp,
        payload.system_prompt,
        payload.params
    )
    t_graph = now() - t0

    print(f"[TIMING] LangGraph: {t_graph:.3f} sec")
    log_perf(payload.user_id, "langgraph_outer", t_graph)

    text = output["response"]

    audio_b64 = run_tts(text) if payload.tts_enabled else None

    return ChatResponse(response=text, audio_base64=audio_b64)

# ---------------- Audio Buffer Helpers -------------------

MAX_BUFFER_BYTES = 60 * 16000 * 2

def add_audio_chunk(user_id: str, chunk_b64: str):
    if user_id not in audio_buffers:
        audio_buffers[user_id] = []
    chunk = base64.b64decode(chunk_b64)
    audio_buffers[user_id].append(chunk)

def get_audio_buffer(user_id: str) -> bytes:
    return b"".join(audio_buffers.get(user_id, []))

def clear_audio_buffer(user_id: str):
    audio_buffers[user_id] = []

# ---------------- HTTP Audio Handlers -------------------

async def handler_audio_stream_input(payload: ChatRequest) -> ChatResponse:
    if payload.audio_base64:
        add_audio_chunk(payload.user_id, payload.audio_base64)
    return ChatResponse(response="", audio_base64=None)

async def handler_audio_finalize(payload: ChatRequest) -> ChatResponse:
    audio_bytes = get_audio_buffer(payload.user_id)
    text = await run_stt(audio_bytes)
    clear_audio_buffer(payload.user_id)
    payload.message = text
    return handler_text_input(payload)

# ---------------- HTTP Endpoint -------------------

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    match payload.input_type:
        case "text":
            return handler_text_input(payload)
        case "audio_stream":
            return await handler_audio_stream_input(payload)
        case "audio_finalize":
            return await handler_audio_finalize(payload)
        case _:
            return ChatResponse(response="Invalid input_type", audio_base64=None)

# ---------------- WebSocket Endpoint -------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")
            user_id = message.get("user_id", "unknown")

            # -------- TEXT INPUT --------
            if msg_type == "text":
                t0 = now()
                output = await asyncio.to_thread(
                    run_chat,
                    user_id,
                    message["text"],
                    message["time_stamp"],
                    message.get("system_prompt", ""),
                    message.get("params", {})
                )
                t_graph = now() - t0

                print(f"[TIMING] LangGraph: {t_graph:.3f} sec")
                log_perf(user_id, "langgraph_outer", t_graph)

                t1 = now()
                await websocket.send_json({
                    "type": "llm_response",
                    "text": output["response"],
                    "audio_base64": output.get("audio_base64")
                })
                t_ws = now() - t1

                print(f"[TIMING] WS Send: {t_ws:.3f} sec")
                log_perf(user_id, "ws_send", t_ws)

            # -------- AUDIO CHUNK --------
            elif msg_type == "audio_chunk":
                audio_buffer.extend(base64.b64decode(message["audio_base64"]))

            # -------- END OF SPEECH --------
            elif msg_type == "audio_end":
                t_total_start = now()

                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_buffer)

                wav_buf.seek(0)

                # STT
                stt_text = await run_stt(bytes(audio_buffer))
                audio_buffer = bytearray()

                await websocket.send_json({
                    "type": "stt_result",
                    "text": stt_text
                })

                synthetic_payload = ChatRequest(
                    user_id=user_id,
                    time_stamp=message["time_stamp"],
                    message=stt_text,
                    system_prompt=message["system_prompt"],
                    params=message["params"],
                    input_type="text",
                    tts_enabled=True
                )

                # LangGraph + TTS
                t1 = now()
                output = await asyncio.to_thread(handler_text_input, synthetic_payload)
                t_graph = now() - t1

                print(f"[TIMING] LangGraph: {t_graph:.3f} sec")
                log_perf(user_id, "langgraph_outer", t_graph)

                # WS send
                t2 = now()
                await websocket.send_json({
                    "type": "llm_response",
                    "text": output.response,
                    "audio_base64": output.audio_base64
                })
                t_ws = now() - t2

                print(f"[TIMING] WS Send: {t_ws:.3f} sec")
                log_perf(user_id, "ws_send", t_ws)

                # Total
                t_total = now() - t_total_start
                print(f"[TIMING] TOTAL ROUND TRIP: {t_total:.3f} sec\n")
                log_perf(user_id, "round_trip", t_total)

    except WebSocketDisconnect:
        print("WebSocket disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
