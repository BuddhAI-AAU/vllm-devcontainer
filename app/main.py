from fastapi import FastAPI
from api.openresponses_gateway import router as responses_router

app = FastAPI()

app.include_router(responses_router)
