"""
Main FastAPI application
Dog Breed Prediction API
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
from datetime import datetime
import os


from app.model_loader import get_predictor
from app.schemas import (
    PredictionResponse,
    PredictionResult,
    BatchPredictionResponse,
    BatchPredictionResult,
    HealthResponse,
    BreedsResponse
)
from app.utils import (
    format_breed_name,
    IMG_SIZE,
    NUM_CLASSES
)


# Create FastAPI app
app = FastAPI(
    title="Dog Breed Prediction API",
    description="Predict dog breeds from images using Transfer Learning (ResNet50)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Setup templates
templates = Jinja2Templates(directory="templates")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup event - load model once
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("=" * 60)
    print("🚀 Starting Dog Breed Prediction API...")
    print("=" * 60)
    
    try:
        # This will load the model and labels
        predictor = get_predictor()
        print("=" * 60)
        print("✅ API ready to serve predictions!")
        print("=" * 60)
    except Exception as e:
        print("=" * 60)
        print(f"❌ Failed to load model: {e}")
        print("=" * 60)
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down Dog Breed Prediction API...")


# Serve frontend
@app.get("/", tags=["Frontend"])
async def read_root(request: Request):
    """Serve the frontend HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})


# API info endpoint
@app.get("/api", tags=["Health"])
async def api_info():
    """API information endpoint"""
    return {
        "message": "🐶 Dog Breed Prediction API",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "predict_batch": "POST /predict-batch",
            "health": "GET /health",
            "breeds": "GET /breeds",
            "docs": "GET /docs"
        }
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health check endpoint"""
    try:
        predictor = get_predictor()
        model_loaded = predictor.model is not None
        labels_loaded = len(predictor.labels) > 0
        
        return HealthResponse(
            status="healthy" if (model_loaded and labels_loaded) else "unhealthy",
            model_loaded=model_loaded,
            labels_loaded=labels_loaded,
            num_breeds=len(predictor.labels),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_dog_breed(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, etc.)"),
    top_k: int = 5
):
    """
    Predict dog breed from uploaded image
    
    - **file**: Image file to analyze (JPEG, PNG, etc.)
    - **top_k**: Number of top predictions to return (default: 5, max: 120)
    
    Returns prediction results with breed names and confidence scores
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}"
        )
    
    # Validate top_k parameter
    if top_k < 1 or top_k > 120:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 120"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Basic size validation (10MB limit)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image file too large. Maximum size: 10MB"
            )
        
        # Get predictor and make prediction
        predictor = get_predictor()
        predictions = predictor.predict(image_bytes, top_k=top_k)
        
        # Format response
        prediction_results = [
            PredictionResult(
                breed=format_breed_name(breed),
                confidence=confidence,
                confidence_percentage=f"{confidence * 100:.2f}%"
            )
            for breed, confidence in predictions
        ]
        
        return PredictionResponse(
            success=True,
            predictions=prediction_results,
            top_prediction=format_breed_name(predictions[0][0]),
            confidence=predictions[0][1],
            timestamp=datetime.now().isoformat(),
            message="Prediction completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# Batch prediction endpoint
@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    top_k: int = 3
):
    """
    Predict dog breeds for multiple images
    
    - **files**: List of image files (max 10 images)
    - **top_k**: Number of top predictions per image (default: 3)
    
    Returns predictions for each image
    """
    # Validate number of files
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch request"
        )
    
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    results = []
    predictor = get_predictor()
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type or not file.content_type.startswith("image/"):
                results.append(BatchPredictionResult(
                    filename=file.filename,
                    success=False,
                    error="Not an image file"
                ))
                continue
            
            # Read image
            image_bytes = await file.read()
            
            # Basic size validation
            if len(image_bytes) > 10 * 1024 * 1024:
                results.append(BatchPredictionResult(
                    filename=file.filename,
                    success=False,
                    error="Image too large (max 10MB)"
                ))
                continue
            
            # Make prediction
            predictions = predictor.predict(image_bytes, top_k=top_k)
            
            prediction_results = [
                PredictionResult(
                    breed=format_breed_name(breed),
                    confidence=confidence,
                    confidence_percentage=f"{confidence * 100:.2f}%"
                )
                for breed, confidence in predictions
            ]
            
            results.append(BatchPredictionResult(
                filename=file.filename,
                success=True,
                predictions=prediction_results,
                top_prediction=format_breed_name(predictions[0][0])
            ))
            
        except Exception as e:
            results.append(BatchPredictionResult(
                filename=file.filename,
                success=False,
                error=str(e)
            ))
    
    return BatchPredictionResponse(
        total_images=len(files),
        results=results,
        timestamp=datetime.now().isoformat()
    )


# Get all breeds endpoint
@app.get("/breeds", response_model=BreedsResponse, tags=["Information"])
async def get_all_breeds():
    """
    Get list of all 120 supported dog breeds
    
    Returns alphabetically sorted list of breed names
    """
    try:
        predictor = get_predictor()
        breeds = [format_breed_name(label) for label in predictor.get_all_breeds()]
        
        return BreedsResponse(
            total_breeds=len(breeds),
            breeds=sorted(breeds)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load breeds: {str(e)}"
        )


# Mount static files (MUST be at the end, after all routes)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Run with: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
