"""
Pydantic schemas for request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    huggingface_url: str = Field(
        ..., 
        description="HuggingFace model URL (e.g., https://huggingface.co/model-name)",
        example="https://huggingface.co/google/gemma-2-2b"
    )


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    model_id: str
    predicted_popularity: str = Field(..., description="'high' or 'low'")
    probability: float = Field(..., description="Confidence score (0-1)")
    features: dict = Field(..., description="Extracted features from the model")
    message: Optional[str] = Field(None, description="Additional information")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None


class FeatureExtractionRequest(BaseModel):
    """Request model for feature extraction endpoint."""
    huggingface_url: Optional[str] = Field(
        None,
        description="HuggingFace model URL (e.g., https://huggingface.co/model-name)",
        example="https://huggingface.co/google/gemma-2-2b"
    )
    model_id: Optional[str] = Field(
        None,
        description="HuggingFace model ID (e.g., google/gemma-2-2b)",
        example="google/gemma-2-2b"
    )
    tags: Optional[list] = Field(
        None,
        description="List of tags (if providing directly instead of URL)"
    )
    last_modified: Optional[str] = Field(
        None,
        description="Last modified date (ISO format)"
    )


class FeatureExtractionResponse(BaseModel):
    """Response model for feature extraction endpoint."""
    model_id: str
    features: dict = Field(..., description="Extracted features from the model")
    prediction: Optional[dict] = Field(None, description="Popularity prediction if model is available")
    feature_importance: Optional[list] = Field(None, description="Top feature importance scores")
    message: Optional[str] = Field(None, description="Additional information")

