import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.services.analysis_service import analyze_article
from models.api.schemas import AnalyzeRequest


app = FastAPI(
    title="RippleIQ",
    description="Causal intelligence engine",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent.parent


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    if not request.article.strip():
        raise HTTPException(
            status_code=400,
            detail="Article text cannot be empty.",
        )

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set.",
        )

    return analyze_article(request.article)


@app.get("/")
async def index():
    return FileResponse(
        BASE_DIR / "frontend" / "static" / "index.html"
    )


app.mount(
    "/static",
    StaticFiles(
        directory=BASE_DIR / "frontend" / "static"
    ),
    name="static",
)