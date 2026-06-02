from fastapi import FastAPI
from pydantic import BaseModel
from services.langgraph_node import run_chat
import base64
import httpx
from services.vad import detect_end_of_speech
#gateway for BuddhAi el
from api.openresponses_gateway import router as responses_router
import io
import wave
#gateway for EverMemOS testing (instead of Grok we use vLLM)
#from api.evermem_test_gateway import router as evermem_responses_router

app = FastAPI(title="LangGraph + vLLM API")
app.include_router(responses_router)        #change depending on which gateway to use

TTS_URL = "http://localhost:7000/v1/audio/speech"
WHISPER_URL = "http://localhost:7001/transcribe"



# Per-user audio buffers
audio_buffers: dict[str, list[bytes]] = {}

#---------------- ingoing ------------
#what we turn the payload into
class ChatRequest(BaseModel):
    user_id: str
    time_stamp: str
    message: str | None = ""
    system_prompt: str
    params: dict
    input_type: str
    tts_enabled: bool = True
    audio_base64: str | None = None


#---------outgoing-------------------
#what we pass to the client
class ChatResponse(BaseModel):
    response: str
    audio_base64: str | None = None     #for tts

#---------- Handlers for ingoing inputs ---------------------
def handler_text_input(payload: ChatRequest) -> ChatResponse:
    output = run_chat(
        payload.user_id,
        payload.message,
        payload.time_stamp,
        payload.system_prompt,
        payload.params
    )

    text = output["response"]
    audio_b64 = run_tts(text) if payload.tts_enabled else None

    return ChatResponse(response=text, audio_base64=audio_b64)

#----------------- TTS -----------------------

def run_tts(text: str) -> str | None:
    try:
        tts_payload = {
            "input": text,
            "model": "mistralai/Voxtral-4B-TTS-2603",
            "voice": "casual_male",
            "response_format": "wav"
            }

        resp = httpx.post(TTS_URL, json=tts_payload, timeout=10.0)
        resp.raise_for_status()

        return base64.b64encode(resp.content).decode("utf-8")

    except Exception:
        # TTS failed — but text still works
        return None
    
#-------------------STT---------------------------
async def run_stt(audio_bytes: bytes) -> str:
    """
    audio_bytes should be a valid WAV file.
    If your chunks are raw PCM, we wrap them into a WAV container here.
    """

    # Example: assume 16kHz, mono, 16-bit PCM
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(16000)  # 16 kHz
        wf.writeframes(audio_bytes)

    wav_buf.seek(0)
    files = {
        "file": ("audio.wav", wav_buf.read(), "audio/wav")
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(WHISPER_URL, files=files, timeout=60.0)
        resp.raise_for_status()
        return resp.json().get("text", "")

# ------------------ Audio buffer functions --------------

#Append a base64 audio chunk to the user's buffer
MAX_BUFFER_BYTES = 60 * 16000 * 2  # ~60 seconds at 16kHz, 16-bit mono

def add_audio_chunk(user_id: str, chunk_b64: str):
    if user_id not in audio_buffers:
        audio_buffers[user_id] = []

    chunk = base64.b64decode(chunk_b64)
    audio_buffers[user_id].append(chunk)

    # Simple safety: trim if buffer too large
    total_len = sum(len(c) for c in audio_buffers[user_id])
    if total_len > MAX_BUFFER_BYTES:
        # Drop oldest chunks until under limit
        while audio_buffers[user_id] and total_len > MAX_BUFFER_BYTES:
            removed = audio_buffers[user_id].pop(0)
            total_len -= len(removed)

#Return the full audio buffer as raw bytes
def get_audio_buffer(user_id: str) -> bytes:
    if user_id not in audio_buffers:
        return b""
    return b"".join(audio_buffers[user_id])

#Clear the user's audio buffer
def clear_audio_buffer(user_id: str):
    audio_buffers[user_id] = []

# ----------------------- AUdio handlers-------------------------------

async def handler_audio_stream_input(payload: ChatRequest) -> ChatResponse:
    if payload.audio_base64:
        add_audio_chunk(payload.user_id, payload.audio_base64)

    # Use raw bytes for VAD, not the list
    raw_audio = get_audio_buffer(payload.user_id)

    if detect_end_of_speech(raw_audio):
        return await handler_audio_finalize(payload)

    return ChatResponse(
        response="(audio chunk received)",
        audio_base64=None
    )


#end of speech finalize, put together audio cunks from buffer
async def handler_audio_finalize(payload: ChatRequest) -> ChatResponse:
    audio_bytes = get_audio_buffer(payload.user_id)

    # Run STT via microservice
    text = await run_stt(audio_bytes)

    clear_audio_buffer(payload.user_id)

    # Reuse your text handler
    payload.message = text
    return handler_text_input(payload)


#------------ endpoint -----------------
#post endpoint - multimodal version
@app.post("/chat", response_model=ChatResponse)
#turn payload into ChatRequest and return as string
async def chat_endpoint(payload: ChatRequest):

    match payload.input_type:

        case "text":
            return handler_text_input(payload)

        case "audio_stream":
            return await handler_audio_stream_input(payload)
        
        case "audio_finalize":
            return await handler_audio_finalize(payload)

        case _:
            return ChatResponse(
                response="Invalid input_type",
                audio_base64=None
            )
