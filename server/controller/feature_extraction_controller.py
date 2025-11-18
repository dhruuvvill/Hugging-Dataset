"""
Controller for feature extraction endpoints.
"""

import json
from fastapi import APIRouter, HTTPException
from models.schemas import FeatureExtractionRequest, FeatureExtractionResponse, ErrorResponse
from services.feature_extraction_service import FeatureExtractionService
from services.hf_service import HuggingFaceService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
feature_service = FeatureExtractionService()
hf_service = HuggingFaceService()


@router.post("/extract-features", response_model=FeatureExtractionResponse)
async def extract_features(request: FeatureExtractionRequest):
    """
    Extract comprehensive features from a HuggingFace model.
    
    Args:
        request: FeatureExtractionRequest containing HuggingFace URL or model ID
        
    Returns:
        FeatureExtractionResponse with extracted features and optional prediction
    """
    try:
        model_id = None
        tags = None
        last_modified = None
        
        # Get model info from HuggingFace if URL or model_id provided
        if request.huggingface_url:
            is_dataset, entity_id = hf_service.extract_model_id(request.huggingface_url)
            model_id = entity_id
            logger.info(f"Extracting features for model: {model_id}")
            
            # Get model info from HuggingFace
            model_info = hf_service.get_model_info(entity_id, is_dataset=is_dataset)
            tags = model_info.get('tags', [])
            last_modified = model_info.get('lastModified', None)
            
        elif request.model_id:
            model_id = request.model_id
            logger.info(f"Extracting features for model: {model_id}")
            
            # Get model info from HuggingFace
            is_dataset = "/datasets/" in model_id or model_id.startswith("datasets/")
            model_info = hf_service.get_model_info(model_id, is_dataset=is_dataset)
            tags = model_info.get('tags', [])
            last_modified = model_info.get('lastModified', None)
            
        elif request.tags:
            # Use provided tags directly
            model_id = "custom_model"
            tags = request.tags
            last_modified = request.last_modified
        else:
            raise ValueError("Either huggingface_url, model_id, or tags must be provided")
        
        # Convert tags to string format if it's a list
        tags_str = json.dumps(tags) if isinstance(tags, list) else str(tags) if tags else "[]"
        
        # Extract comprehensive features
        features = feature_service.extract_comprehensive_features(
            tags_str=tags_str,
            last_modified=last_modified,
            model_id=model_id
        )
        
        # Try to predict popularity if model is available
        prediction = feature_service.predict_popularity(features)
        
        # Get feature importance
        feature_importance = feature_service.get_feature_importance()
        top_features = None
        if feature_importance:
            top_features = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:30]
        
        message = "Features extracted successfully"
        if prediction:
            message += f". Predicted popularity: {'High' if prediction['is_popular'] else 'Low'} (confidence: {prediction['probability']:.2%})"
        
        return FeatureExtractionResponse(
            model_id=model_id,
            features=features,
            prediction=prediction,
            feature_importance=top_features,
            message=message
        )
        
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")


@router.get("/extract-features/{model_id}", response_model=FeatureExtractionResponse)
async def extract_features_by_id(model_id: str):
    """
    Extract comprehensive features from a HuggingFace model by ID.
    
    Args:
        model_id: The HuggingFace model ID (e.g., "google/gemma-2-2b")
        
    Returns:
        FeatureExtractionResponse with extracted features and optional prediction
    """
    try:
        logger.info(f"Extracting features for model: {model_id}")
        
        # Check if it's a dataset
        is_dataset = "/datasets/" in model_id or model_id.startswith("datasets/")
        
        # Get model info from HuggingFace
        model_info = hf_service.get_model_info(model_id, is_dataset=is_dataset)
        tags = model_info.get('tags', [])
        last_modified = model_info.get('lastModified', None)
        
        # Convert tags to string format
        tags_str = json.dumps(tags) if isinstance(tags, list) else str(tags) if tags else "[]"
        
        # Extract comprehensive features
        features = feature_service.extract_comprehensive_features(
            tags_str=tags_str,
            last_modified=last_modified,
            model_id=model_id
        )
        
        # Try to predict popularity if model is available
        prediction = feature_service.predict_popularity(features)
        
        # Get feature importance
        feature_importance = feature_service.get_feature_importance()
        top_features = None
        if feature_importance:
            top_features = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:30]
        
        message = "Features extracted successfully"
        if prediction:
            message += f". Predicted popularity: {'High' if prediction['is_popular'] else 'Low'} (confidence: {prediction['probability']:.2%})"
        
        return FeatureExtractionResponse(
            model_id=model_id,
            features=features,
            prediction=prediction,
            feature_importance=top_features,
            message=message
        )
        
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

