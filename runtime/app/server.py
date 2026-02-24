from fastapi import FastAPI
from pydantic import BaseModel
from services.langgraph_node import run_chat
from api.openresponses_gateway import router as responses_router


app = FastAPI(title="LangGraph + vLLM API")
app.include_router(responses_router)

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    output = run_chat(payload.user_id, payload.message)
    return ChatResponse(response=output)
