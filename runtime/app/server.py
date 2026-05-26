from fastapi import FastAPI
from pydantic import BaseModel
from services.langgraph_node import run_chat
import base64
import httpx

#gateway for BuddhAi
from api.openresponses_gateway import router as responses_router

#gateway for EverMemOS testing (instead of Grok we use vLLM)
#from api.evermem_test_gateway import router as evermem_responses_router


app = FastAPI(title="LangGraph + vLLM API")
app.include_router(responses_router)        #change depending on which gateway to use

TTS_URL = "http://localhost:7000/v1/audio/speech"

#what we turn the payload into
class ChatRequest(BaseModel):
    user_id: str
    time_stamp: str
    message: str
    system_prompt: str
    params: dict

#what we pass to the client
class ChatResponse(BaseModel):
    response: str
    audio_base64: str | None = None     #for tts
    


#post endpoint - multimodal version
@app.post("/chat", response_model=ChatResponse)
#turn payload into ChatRequest and return as string
def chat_endpoint(payload: ChatRequest):
    output = run_chat(
        payload.user_id,
        payload.message,
        payload.time_stamp,
        payload.system_prompt,
        payload.params,
    )

    text = output["response"] #tts conversion

    #tts conversion
    tts_payload = {
        "input": text,
        "model": "mistralai/Voxtral-4B-TTS-2603",
        "voice": "casual_male",
        "response_format": "wav",
    }

    try:
        resp = httpx.post(TTS_URL, json=tts_payload, timeout=120.0)
        resp.raise_for_status()
        audio_bytes = resp.content
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    except Exception:
        audio_b64 = None
    
    #return ChatResponse(response=output) #from before tts
    #return ChatResponse(**output)
    return ChatResponse(response=text, audio_base64=audio_b64)


