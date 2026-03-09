from fastapi import FastAPI
from pydantic import BaseModel
from services.langgraph_node import run_chat
from api.openresponses_gateway import router as responses_router


app = FastAPI(title="LangGraph + vLLM API")
app.include_router(responses_router)


#what we turn the payload into
class ChatRequest(BaseModel):
    user_id: str
    time_stamp: str
    message: str

#what we pass to the client
class ChatResponse(BaseModel):
    response: str

#post endpoint
@app.post("/chat", response_model=ChatResponse)
#turn payload into ChatRequest and return as string
def chat_endpoint(payload: ChatRequest):
    output = run_chat(payload.user_id, payload.message, payload.time_stamp)
    return ChatResponse(response=output)
