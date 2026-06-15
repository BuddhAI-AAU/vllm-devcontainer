from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from runtime.app.services.langgraph_node import run_chat
from scripts.perf_logger import log_perf
import base64
import httpx
from runtime.app.api.openresponses_gateway import router as responses_router
import io
import wave
import uvicorn
import asyncio
import time
import websockets
import json

app = FastAPI(title="LangGraph + vLLM API")
app.include_router(responses_router)

WHISPER_URL = "http://localhost:7001/transcribe"

audio_buffers: dict[str, list[bytes]] = {}

#--------------Latency report------------

def now():
    return time.perf_counter()


class SessionState:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.audio_buffer = bytearray()
        self.current_llm_task: asyncio.Task | None = None
        self.current_tts_task: asyncio.Task | None = None
        self.interrupted = False


def is_phrase_boundary(text: str) -> bool:
    return any(text.endswith(p) for p in [".", "!", "?", ":", ";"])


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


# ---------------- Text Handler (HTTP) -------------------

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


#-------------WebSocket TTS client----------------------------

async def tts_stream(phrase: str, instruct: str):
    uri = "ws://omnivoice:7002"

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "config",
            "instruct": instruct
        }))

        await ws.send(json.dumps({
            "type": "text_chunk",
            "text": phrase
        }))

        await ws.send(json.dumps({"type": "end"}))

        async for msg in ws:
            data = json.loads(msg)
            yield data
            if data.get("final"):
                break


#-------------- streaming LLM + TTS ---------------------------------

async def stream_llm_and_tts(
    websocket: WebSocket,
    session: SessionState,
    user_text: str,
    system_prompt: str,
    params: dict,
    instruct: str = "male",
):
    messages = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        }
    ]

    body = {"input": messages, "params": params or {}}

    llm_buffer = ""
    full_text = ""

    t0 = now()
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", "http://localhost:9000/responses", json=body) as resp:
                async for chunk in resp.aiter_text():
                    if session.interrupted:
                        return

                    if not chunk:
                        continue

                    llm_buffer += chunk
                    full_text += chunk

                    if len(llm_buffer) > 40 or is_phrase_boundary(llm_buffer):
                        phrase = llm_buffer
                        llm_buffer = ""

                        await websocket.send_json({
                            "type": "llm_partial",
                            "text": phrase,
                            "final": False,
                        })

                        if session.interrupted:
                            return

                        async for tts_msg in tts_stream(phrase, instruct=instruct):
                            if session.interrupted:
                                return

                            await websocket.send_json({
                                "type": "audio_out_chunk",
                                "audio_base64": tts_msg["audio_base64"],
                                "final": tts_msg["final"],
                            })

        if llm_buffer:
            await websocket.send_json({
                "type": "llm_partial",
                "text": llm_buffer,
                "final": False,
            })

            async for tts_msg in tts_stream(llm_buffer, instruct=instruct):
                await websocket.send_json({
                    "type": "audio_out_chunk",
                    "audio_base64": tts_msg["audio_base64"],
                    "final": tts_msg["final"],
                })

            full_text += llm_buffer

        await websocket.send_json({
            "type": "llm_final",
            "text": full_text,
            "final": True,
        })
        await websocket.send_json({"type": "audio_out_end"})

    except asyncio.CancelledError:
        return


#--------------ws handlers ---------------------------------

async def handle_turn_text(websocket: WebSocket, session: SessionState, message: dict):
    user_id = session.user_id

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


async def handle_turn_audio(websocket: WebSocket, session: SessionState, message: dict):
    user_id = session.user_id

    t_total_start = now()

    stt_text = await run_stt(bytes(session.audio_buffer))
    session.audio_buffer = bytearray()

    await websocket.send_json({
        "type": "stt_result",
        "text": stt_text
    })

    session.interrupted = False
    session.current_llm_task = asyncio.create_task(
        stream_llm_and_tts(
            websocket=websocket,
            session=session,
            user_text=stt_text,
            system_prompt=message.get("system_prompt", ""),
            params=message.get("params", {}),
            instruct=message.get("instruct", "male"),
        )
    )

    try:
        await session.current_llm_task
    except asyncio.CancelledError:
        print("LLM task cancelled due to interrupt")
    finally:
        session.current_llm_task = None

    if not session.interrupted:
        t_total = now() - t_total_start
        print(f"[TIMING] TOTAL ROUND TRIP: {t_total:.3f} sec\n")
        log_perf(user_id, "round_trip", t_total)


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

    session: SessionState | None = None

    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")
            user_id = message.get("user_id", "unknown")

            if session is None:
                session = SessionState(user_id)

            if msg_type == "text":
                await handle_turn_text(websocket, session, message)

            elif msg_type == "audio_chunk":
                chunk = base64.b64decode(message["audio_base64"])
                session.audio_buffer.extend(chunk)

            elif msg_type == "audio_end":
                await handle_turn_audio(websocket, session, message)

            elif msg_type == "interrupt":
                session.interrupted = True

                if session.current_llm_task and not session.current_llm_task.done():
                    session.current_llm_task.cancel()

                if session.current_tts_task and not session.current_tts_task.done():
                    session.current_tts_task.cancel()

                session.audio_buffer = bytearray()

                await websocket.send_json({"type": "interrupted"})

    except WebSocketDisconnect:
        print("WebSocket disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
