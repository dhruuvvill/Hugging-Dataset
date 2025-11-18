"""
Controller for prediction endpoints.
"""

from fastapi import APIRouter, HTTPException
from models.schemas import PredictRequest, PredictionResponse, ErrorResponse
from services.model_service import ModelPredictionService
from services.hf_service import HuggingFaceService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
prediction_service = ModelPredictionService()
hf_service = HuggingFaceService()


@router.post("/predict", response_model=PredictionResponse)
async def predict_popularity(request: PredictRequest):
    """
    Predict the popularity of a HuggingFace model based on its URL.
    
    Args:
        request: PredictRequest containing the HuggingFace URL
        
    Returns:
        PredictionResponse with predicted popularity and features
        
    Raises:
        HTTPException: If the model cannot be found or prediction fails
    """
    try:
        # Extract entity ID from URL (handles both models and datasets)
        is_dataset, entity_id = hf_service.extract_model_id(request.huggingface_url)
        entity_type = "dataset" if is_dataset else "model"
        logger.info(f"Extracting features for {entity_type}: {entity_id}")
        
        # Get model/dataset info from HuggingFace
        model_info = hf_service.get_model_info(entity_id, is_dataset=is_dataset)
        
        # Extract features (pass is_dataset flag for proper feature extraction)
        features = hf_service.extract_features(model_info, is_dataset=is_dataset)
        
        # Make prediction (pass is_dataset flag so it uses appropriate method)
        prediction = prediction_service.predict(features, is_dataset=is_dataset)
        
        # Format response
        predicted_popularity = "high" if prediction['prediction'] == 1 else "low"
        
        # Calculate confidence: if predicting low, use 1 - probability (since probability is for high class)
        # If predicting high, use probability directly
        if prediction['prediction'] == 0:
            # Predicting low popularity, so confidence is 1 - probability of high
            confidence = 1.0 - float(prediction['probability'])
        else:
            # Predicting high popularity, use probability directly
            confidence = float(prediction['probability'])
        
        # Custom message for datasets
        if is_dataset:
            message = f"This dataset is predicted to have {'high' if predicted_popularity == 'high' else 'low'} popularity based on its metadata."
        else:
            message = f"This model is predicted to have {'high' if predicted_popularity == 'high' else 'low'} popularity based on its metadata."
        
        return PredictionResponse(
            model_id=entity_id,
            predicted_popularity=predicted_popularity,
            probability=confidence,
            features=prediction['features'],
            message=message
        )
        
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=500, detail="Model file not found. Please train the model first.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/predict/{model_id}", response_model=PredictionResponse)
async def predict_popularity_by_id(model_id: str):
    """
    Predict the popularity of a HuggingFace model by its ID.
    
    Args:
        model_id: The HuggingFace model ID (e.g., "google/gemma-2-2b")
        
    Returns:
        PredictionResponse with predicted popularity and features
    """
    try:
        logger.info(f"Predicting popularity for entity: {model_id}")
        
        # Check if it's a dataset by parsing the ID (assume model if unknown)
        is_dataset = "/datasets/" in model_id or model_id.startswith("datasets/")
        
        # Get model/dataset info from HuggingFace
        model_info = hf_service.get_model_info(model_id, is_dataset=is_dataset)
        
        # Extract features (pass is_dataset flag for proper feature extraction)
        features = hf_service.extract_features(model_info, is_dataset=is_dataset)
        
        # Make prediction (pass is_dataset flag so it uses appropriate method)
        prediction = prediction_service.predict(features, is_dataset=is_dataset)
        
        # Format response
        predicted_popularity = "high" if prediction['prediction'] == 1 else "low"
        
        # Calculate confidence: if predicting low, use 1 - probability (since probability is for high class)
        # If predicting high, use probability directly
        if prediction['prediction'] == 0:
            # Predicting low popularity, so confidence is 1 - probability of high
            confidence = 1.0 - float(prediction['probability'])
        else:
            # Predicting high popularity, use probability directly
            confidence = float(prediction['probability'])
        
        entity_type = "dataset" if is_dataset else "model"
        message = f"This {entity_type} is predicted to have {'high' if predicted_popularity == 'high' else 'low'} popularity based on its metadata."
        
        return PredictionResponse(
            model_id=model_id,
            predicted_popularity=predicted_popularity,
            probability=confidence,
            features=prediction['features'],
            message=message
        )
        
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=500, detail="Model file not found. Please train the model first.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

