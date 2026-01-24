from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class GenrePrediction(BaseModel):
    genre: str = Field(..., description="Predicted genre")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


class GenreResponse(BaseModel):
    predictions: List[GenrePrediction] = Field(..., description="Top genre predictions")
    top_genre: str = Field(..., description="Most likely genre")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class AuthorPrediction(BaseModel):
    artist: str = Field(..., description="Predicted artist name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


class AuthorResponse(BaseModel):
    predictions: List[AuthorPrediction] = Field(..., description="Top artist predictions")
    top_artist: str = Field(..., description="Most likely artist")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class CombinedResponse(BaseModel):
    genre: GenreResponse
    author: AuthorResponse


class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
