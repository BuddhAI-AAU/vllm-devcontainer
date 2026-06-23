from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
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

from runtime.app.services.memory_node import (
    build_messages_with_history,
    write_turn,
)

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


#-------------- streaming LLM only ----------------------------------

async def stream_text_only(
    websocket: WebSocket,
    session: SessionState,
    user_text: str,
    system_prompt: str,
    params: dict,
):
    # build messages with history + system prompt
    messages = build_messages_with_history(
        user_id=session.user_id,
        user_input=user_text,
        system_prompt=system_prompt,
        history_limit=10,
    )

    body = {"input": messages, "params": params or {}}

    llm_buffer = ""
    full_text = ""

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "http://localhost:9000/responses",
                json=body
            ) as resp:

                async for chunk in resp.aiter_text():

                    if session.interrupted:
                        return

                    if not chunk:
                        continue

                    # accumulate raw streamed text
                    llm_buffer += chunk

                    # flush when phrase boundary OR enough text
                    if len(llm_buffer) > 160 or is_phrase_boundary(llm_buffer):
                        phrase = llm_buffer
                        llm_buffer = ""

                        # send partial to client
                        await websocket.send_json({
                            "type": "llm_partial",
                            "text": phrase,
                            "final": False,
                        })

                        # accumulate into final output
                        full_text += phrase

        # flush any remaining text
        if llm_buffer:
            await websocket.send_json({
                "type": "llm_partial",
                "text": llm_buffer,
                "final": False,
            })
            full_text += llm_buffer

        # send final message
        await websocket.send_json({
            "type": "llm_final",
            "text": full_text,
            "final": True,
        })

        # memory write
        if not session.interrupted and full_text.strip():
            await asyncio.to_thread(
                write_turn,
                {
                    "user_id": session.user_id,
                    "input": user_text,
                    "response": full_text,
                    "time_stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "system_prompt": system_prompt,
                    "params": params,
                    "history": [],
                }
            )

    except asyncio.CancelledError:
        return


#-------------- streaming LLM + TTS ---------------------------------

async def stream_llm_and_tts(
    websocket: WebSocket,
    session: SessionState,
    user_text: str,
    system_prompt: str,
    params: dict,
    instruct: str = "male",
):

    # build messages with history + system prompt
    messages = build_messages_with_history(
        user_id=session.user_id,
        user_input=user_text,
        system_prompt=system_prompt,
        history_limit=10,
    )

    body = {"input": messages, "params": params or {}}

    llm_buffer = ""
    full_text = ""

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "http://localhost:9000/responses",
                json=body
            ) as resp:
                async for chunk in resp.aiter_text():

                    # interruption check
                    if session.interrupted:
                        return

                    if not chunk:
                        continue

                    # accumulate tokens
                    llm_buffer += chunk
                    full_text += chunk

                    # phrase boundary or chunk size threshold
                    if len(llm_buffer) > 40 or is_phrase_boundary(llm_buffer):
                        phrase = llm_buffer
                        llm_buffer = ""

                        # send partial text to client
                        await websocket.send_json({
                            "type": "llm_partial",
                            "text": phrase,
                            "final": False,
                        })

                        if session.interrupted:
                            return

                        # stream TTS for this phrase
                        async for tts_msg in tts_stream(phrase, instruct=instruct):
                            if session.interrupted:
                                return

                            await websocket.send_json({
                                "type": "audio_out_chunk",
                                "audio_base64": tts_msg["audio_base64"],
                                "final": tts_msg["final"],
                            })

        # flush any remaining text
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

        # send final text to client
        await websocket.send_json({
            "type": "llm_final",
            "text": full_text,
            "final": True,
        })

        # signal end of audio stream
        await websocket.send_json({"type": "audio_out_end"})

        # write memory
        if not session.interrupted and full_text.strip():
            await asyncio.to_thread(
                write_turn,
                {
                    "user_id": session.user_id,
                    "input": user_text,      # STT or typed text
                    "response": full_text,   # full assistant output
                    "time_stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "system_prompt": system_prompt,
                    "params": params,
                    "history": [],
                }
            )

    except asyncio.CancelledError:
        return


#--------------ws handlers ---------------------------------

async def handle_turn_text(websocket: WebSocket, session: SessionState, message: dict):
    session.interrupted = False

    session.current_llm_task = asyncio.create_task(
        stream_text_only(
            websocket=websocket,
            session=session,
            user_text=message["text"],
            system_prompt=message.get("system_prompt", ""),
            params=message.get("params", {}),
        )
    )

    try:
        await session.current_llm_task
    except asyncio.CancelledError:
        print("Text LLM task cancelled")
    finally:
        session.current_llm_task = None


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
