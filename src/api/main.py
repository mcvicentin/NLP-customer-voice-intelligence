# src/api/main.py

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Customer Voice Intelligence API",
    description="NLP endpoints for sentiment, topics, and summarization.",
    version="1.0.0"
)

# ---------------------------
# Health Check Endpoint
# -------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Customer Voice Intelligence API is running."
    }

# ---------------------------
# Schema Base
#---------------------------
class TextInput(BaseModel):
    text: str


