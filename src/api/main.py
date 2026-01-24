import sys
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import genre, author
from .services.model_service import model_service
from .schemas.predictions import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    status = model_service.load_all_models()
    print(f"Model loading status: {status}")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Music Classification API",
    description="API for classifying music by genre and artist/author",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(genre.router)
app.include_router(author.router)


@app.get("/", summary="Root endpoint")
async def root():
    return {
        "message": "Music Classification API",
        "docs": "/docs",
        "endpoints": {
            "genre": "/genre/predict",
            "author": "/author/predict",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    return HealthResponse(
        status="healthy",
        models_loaded={
            "genre": model_service.is_genre_loaded(),
            "author": model_service.is_author_loaded()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
