import time
from fastapi import APIRouter, UploadFile, File, HTTPException

from ..schemas.predictions import AuthorResponse, AuthorPrediction, ErrorResponse
from ..services.model_service import model_service
from ..services.audio_service import audio_service


router = APIRouter(prefix="/author", tags=["Author Classification"])


@router.post(
    "/predict",
    response_model=AuthorResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Classify music artist/author",
    description="Upload an audio file to predict the artist/author"
)
async def predict_author(
    file: UploadFile = File(..., description="Audio file (MP3, WAV, etc.)"),
    top_k: int = 5
):
    if not model_service.is_author_loaded():
        raise HTTPException(status_code=503, detail="Author model not loaded")
    
    start_time = time.time()
    audio_path = None
    
    try:
        audio_path = await audio_service.save_upload_file(file)
        features = audio_service.extract_author_features(audio_path)
        predictions = model_service.predict_author(features, top_k=top_k)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AuthorResponse(
            predictions=[
                AuthorPrediction(artist=artist, confidence=conf)
                for artist, conf in predictions
            ],
            top_artist=predictions[0][0],
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if audio_path:
            audio_service.cleanup(audio_path)
