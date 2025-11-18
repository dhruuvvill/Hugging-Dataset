#!/usr/bin/env python3
"""
FastAPI application for HuggingFace model popularity prediction.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controller.predict_controller import router as predict_router
from controller.feature_extraction_controller import router as feature_extraction_router
from controller.dataset_stats_controller import router as dataset_stats_router

app = FastAPI(
    title="HuggingFace Model Popularity Predictor",
    description="Predict the popularity of HuggingFace models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router, prefix="/api/v1", tags=["prediction"])
app.include_router(feature_extraction_router, prefix="/api/v1", tags=["feature-extraction"])
app.include_router(dataset_stats_router, prefix="/api/v1", tags=["dataset-stats"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "HuggingFace Model Popularity Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/v1/predict",
            "extract_features": "/api/v1/extract-features",
            "trending_datasets": "/api/v1/datasets/trending",
            "top_datasets": "/api/v1/datasets/top",
            "dataset_stats": "/api/v1/datasets/stats",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

