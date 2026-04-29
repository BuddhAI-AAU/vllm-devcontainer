from fastapi import FastAPI
from pydantic import BaseModel
from services.langgraph_node import run_chat

#gateway for BuddhAi
from api.openresponses_gateway import router as responses_router

#gateway for EverMemOS testing (instead of Grok we use vLLM)
#from api.evermem_test_gateway import router as evermem_responses_router


app = FastAPI(title="LangGraph + vLLM API")
app.include_router(responses_router)        #change depending on which gateway to use


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
    


#post endpoint
@app.post("/chat", response_model=ChatResponse)
#turn payload into ChatRequest and return as string
def chat_endpoint(payload: ChatRequest):
    output = run_chat(payload.user_id, payload.message, payload.time_stamp, payload.system_prompt, payload.params,)
    return ChatResponse(
        response=output
        )
