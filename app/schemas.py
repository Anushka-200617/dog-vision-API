
"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PredictionResult(BaseModel):
    """Single prediction result"""
    breed: str = Field(..., description="Dog breed name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    confidence_percentage: str = Field(..., description="Confidence as percentage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "breed": "Golden Retriever",
                "confidence": 0.8934,
                "confidence_percentage": "89.34%"
            }
        }


class PredictionResponse(BaseModel):
    """Complete prediction response"""
    success: bool = Field(..., description="Whether prediction was successful")
    predictions: List[PredictionResult] = Field(..., description="List of top predictions")
    top_prediction: str = Field(..., description="Most likely breed")
    confidence: float = Field(..., description="Confidence of top prediction")
    timestamp: str = Field(..., description="ISO format timestamp")
    message: Optional[str] = Field(None, description="Additional message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "predictions": [
                    {
                        "breed": "Golden Retriever",
                        "confidence": 0.8934,
                        "confidence_percentage": "89.34%"
                    }
                ],
                "top_prediction": "Golden Retriever",
                "confidence": 0.8934,
                "timestamp": "2025-11-23T00:29:00.000000",
                "message": "Prediction completed successfully"
            }
        }


class BatchPredictionResult(BaseModel):
    """Result for single image in batch prediction"""
    filename: str
    success: bool
    predictions: Optional[List[PredictionResult]] = None
    top_prediction: Optional[str] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    total_images: int
    results: List[BatchPredictionResult]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    labels_loaded: bool
    num_breeds: int
    timestamp: str


class BreedsResponse(BaseModel):
    """Response for breeds list"""
    total_breeds: int
    breeds: List[str]
