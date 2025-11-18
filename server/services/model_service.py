"""
Service for loading and using the trained model.
"""

import os
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelPredictionService:
    """Service for loading and using the trained popularity prediction model."""
    
    def __init__(self):
        self.model = None
        self.feature_order = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and feature order."""
        try:
            # Find model file - try multiple locations
            current_dir = Path(__file__).resolve().parent  # services/
            project_root = current_dir.parent.parent  # HuggingFace/
            
            possible_paths = [
                project_root / "models_popularity" / "popularity_classification_downloads.joblib",
                Path("models_popularity/popularity_classification_downloads.joblib"),
                Path("../models_popularity/popularity_classification_downloads.joblib"),
            ]
            
            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = path.resolve()
                    break
            
            if not model_path or not model_path.exists():
                logger.warning(f"Model file not found. Searched: {possible_paths}")
                logger.warning("Will use heuristic prediction. Train the model with: python3 model.py")
                return
            
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            
            # Try to load feature names
            features_path = model_path.parent / "feature_names.json"
            if features_path.exists():
                import json
                with open(features_path, 'r') as f:
                    self.feature_order = json.load(f)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Will use heuristic prediction")
            # Don't raise error, allow fallback to heuristic
    
    def predict(self, features: dict, is_dataset: bool = False) -> dict:
        """
        Make a prediction on features.
        
        Args:
            features: Dictionary of features
            is_dataset: Whether this is a dataset (True) or model (False)
            
        Returns:
            Dictionary containing prediction and probability
        """
        # For datasets, use heuristic since model was trained on models
        # Check if it looks like a dataset (has library:datasets or very few model-specific features)
        if is_dataset or (features.get('has_library_datasets', 0) > 0 and features.get('has_transformers', 0) == 0):
            logger.info("Using heuristic prediction for dataset (model was trained on models)")
            result = self._simple_predict(features, is_dataset=True)
            
            # Moderate age adjustment for datasets - age is one factor among many
            days_old = features.get('days_since_modification', 0)
            if days_old > 1200:
                # Only force override if extremely old (> 1200 days)
                age_factor = min(0.80, (days_old - 1200) / 1000)  # 0 to 0.80
                logger.info(f"Dataset is extremely old ({days_old} days, age_factor: {age_factor:.2f}) - adjusting prediction")
                
                # Force low popularity only if extremely old
                result['prediction'] = 0
                result['probability'] = max(0.05, min(0.40, result['probability'] * (1 - age_factor * 0.6)))
                result['note'] = f'Dataset is extremely old ({days_old} days) - low popularity expected'
            elif days_old > 800:
                # Moderate adjustment for old datasets - don't override completely
                age_factor = min(0.60, (days_old - 800) / 1000)  # 0 to 0.60
                logger.info(f"Dataset is old ({days_old} days, age_factor: {age_factor:.2f}) - adjusting prediction")
                
                # Reduce probability moderately for old datasets
                result['probability'] = result['probability'] * (1 - age_factor * 0.4)  # Reduced from 0.7 to 0.4
                if result['prediction'] == 1 and result['probability'] < 0.60:
                    result['prediction'] = 0
            
            return result
        
        # For models, if the model is loaded, use it but be more conservative
        # Adjust prediction threshold to reduce false positives
        
        if self.model is None:
            # Fallback to simple heuristic if model not loaded
            return self._simple_predict(features)
        
        try:
            # Convert features to DataFrame
            df = pd.DataFrame([features])
            
            # One-hot encode categorical variables (if they exist)
            categorical_cols = ['main_task', 'license_type', 'author']
            for col in categorical_cols:
                if col in df.columns:
                    # Create dummy columns for top values
                    values = df[col].unique()
                    for val in values:
                        df[f'{col}_{val}'] = (df[col] == val).astype(int)
                    df = df.drop(columns=[col])
            
            # Fill missing values
            df = df.fillna(0)
            
            # Get probability and prediction
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(df)[0]
                probability = float(probabilities[1] if len(probabilities) > 1 else probabilities[0])
                
                # CRITICAL: Check days_since_modification - it's the most important feature (0.3294)
                # Dynamic age-based adjustment: highest boost at 0-1 days, decreases with age
                days_old = features.get('days_since_modification', 0)
                
                # Age adjustment: moderate influence, balanced with model's base prediction
                # Age is one factor among many - don't override the model's judgment
                if days_old <= 1:  # Brand new (0-1 days) - moderate boost
                    # Moderate boost: increase probability slightly
                    probability = min(0.95, probability * 1.12)  # Boost by 12%, cap at 95%
                elif days_old <= 7:  # Very recent (1 week) - small boost
                    # Linear decay from 1.12x to 1.06x over 6 days
                    age_factor = (days_old - 1) / 6  # 0 to 1
                    boost = 1.12 - (age_factor * 0.06)  # 1.12 to 1.06
                    probability = min(0.95, probability * boost)
                elif days_old <= 30:  # Recent (1 month) - very small boost
                    # Linear decay from 1.06x to 1.02x over 23 days
                    age_factor = (days_old - 7) / 23  # 0 to 1
                    boost = 1.06 - (age_factor * 0.04)  # 1.06 to 1.02
                    probability = min(0.95, probability * boost)
                elif days_old <= 90:  # Fairly recent (3 months) - minimal boost
                    # Linear decay from 1.02x to 1.0x over 60 days
                    age_factor = (days_old - 30) / 60  # 0 to 1
                    boost = 1.02 - (age_factor * 0.02)  # 1.02 to 1.0
                    probability = min(0.95, probability * boost)
                elif days_old <= 180:  # Slightly old (6 months) - neutral
                    # No adjustment, use base probability
                    pass
                elif days_old <= 365:  # Moderately old (1 year) - small penalty
                    # Linear penalty: reduce by up to 5%
                    age_factor = (days_old - 180) / 185  # 0 to 1 for 180-365 days
                    probability = probability * (1 - age_factor * 0.05)
                elif days_old <= 800:  # Old (more than 1 year) - moderate penalty
                    # Linear penalty: reduce by 5% to 20%
                    age_factor = (days_old - 365) / 435  # 0 to 1 for 365-800 days
                    probability = probability * (1 - 0.05 - age_factor * 0.15)  # 5% to 20% reduction
                else:  # Very old (> 800 days) - significant but not overwhelming penalty
                    # Calculate age penalty factor (0 to 1, increasing with age)
                    # For 1000 days: factor ≈ 0.2
                    # For 1122 days: factor ≈ 0.32
                    # For 1500 days: factor ≈ 0.7
                    age_factor = min(0.90, (days_old - 800) / 1000)
                    
                    # Reduce probability based on age (moderate reduction)
                    probability = probability * (1 - age_factor * 0.5)  # Reduced from 0.8 to 0.5
                    
                    # Only force low popularity if extremely old (> 1200 days)
                    if days_old > 1200:
                        prediction = 0
                        probability = max(0.05, min(0.35, probability))
                    else:
                        prediction = 1 if probability >= 0.60 else 0
                
                # Final prediction threshold
                if days_old <= 1200:  # For items up to 1200 days old, use normal threshold
                    prediction = 1 if probability >= 0.60 else 0
                
                # Adjust probability to be more conservative
                # If predicting high, use probability but cap at 0.90
                # If predicting low, use 1 - probability
                if prediction == 1:
                    # High popularity - use probability but be conservative
                    adjusted_probability = min(0.90, max(0.60, probability * 0.95))
                else:
                    # Low popularity - use 1 - probability
                    adjusted_probability = min(0.90, max(0.50, (1.0 - probability) * 0.95))
                
                probability = adjusted_probability
            else:
                prediction = self.model.predict(df)[0]
                probability = 0.5  # Default probability
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Fallback to simple prediction
            return self._simple_predict(features)
    
    def _simple_predict(self, features: dict, is_dataset: bool = False) -> dict:
        """Simple heuristic-based prediction as fallback."""
        # Simple heuristic: models/datasets with more tags, recent updates tend to be popular
        score = 0
        
        if is_dataset:
            # Dataset-specific heuristics (more conservative to reduce false positives)
            # CRITICAL: days_since_modification is the MOST IMPORTANT feature (0.3294 importance)
            # Dynamic age-based penalty: the older, the more penalized
            days_old = features.get('days_since_modification', 0)
            
            # Age-based scoring: moderate influence, balanced with other features
            # Age is important but not the sole factor - other features matter too
            if days_old <= 1:  # Brand new (0-1 days) - moderate boost
                score += 0.3  # Moderate boost for brand new items (balanced with other features)
            elif days_old <= 7:  # Very recent (1 week) - small boost
                # Linear decay from 0.3 to 0.15 over 6 days
                age_factor = (days_old - 1) / 6  # 0 to 1
                score += 0.3 - (age_factor * 0.15)  # 0.3 to 0.15
            elif days_old <= 30:  # Recent (1 month) - very small boost
                # Linear decay from 0.15 to 0.05 over 23 days
                age_factor = (days_old - 7) / 23  # 0 to 1
                score += 0.15 - (age_factor * 0.10)  # 0.15 to 0.05
            elif days_old <= 90:  # Fairly recent (3 months) - minimal boost
                # Linear decay from 0.05 to 0 over 60 days
                age_factor = (days_old - 30) / 60  # 0 to 1
                score += 0.05 - (age_factor * 0.05)  # 0.05 to 0
            elif days_old <= 180:  # Slightly old (6 months) - neutral to small penalty
                # Small linear penalty
                score -= 0.0002 * (days_old - 90)
            elif days_old <= 365:  # Moderately old (1 year) - moderate penalty
                # Linear penalty
                score -= 0.018 + 0.0003 * (days_old - 180)
            elif days_old <= 800:  # Old (more than 1 year) - heavier penalty
                # Linear penalty
                score -= 0.074 + 0.0004 * (days_old - 365)
            else:  # Very old (> 800 days) - significant but not overwhelming penalty
                # Moderate exponential penalty (reduced from before)
                # Formula: -0.0005 * (days_old - 800)^1.3 (reduced coefficient and exponent)
                age_penalty = -0.0005 * ((days_old - 800) ** 1.3)
                score += age_penalty
            
            # Datasets with licenses are more popular
            score += features.get('has_license', 0) * 0.2
            
            # Having library:datasets tag is good
            score += features.get('has_library_datasets', 0) * 0.15
            
            # Tags help discoverability (but don't over-reward just having many tags)
            # Cap tag contribution to prevent over-scoring
            num_tags = features.get('num_tags', 0)
            if num_tags >= 3:
                # Cap at 10 tags max - having 20 tags doesn't mean 2x better than 10
                score += min(num_tags, 10) * 0.06  # Reduced from 0.08, cap at 10
            elif num_tags < 2:
                # Penalize datasets with very few tags (less discoverable)
                score -= 0.2
            
            # Task categories help
            score += features.get('has_task_category', 0) * 0.12
            
            # Size category helps
            score += features.get('has_size_category', 0) * 0.08
            
            # Modality helps
            score += features.get('has_modality', 0) * 0.08
            
            # Language support helps
            score += features.get('has_language', 0) * 0.1
            
            # Missing important features should be penalized
            # Datasets without transformers/endpoints are less popular
            if features.get('has_transformers', 0) == 0 and features.get('has_endpoints_compatible', 0) == 0:
                score -= 0.2  # Penalty for missing key features
            
            # Require multiple positive signals AND not be too old - higher threshold
            threshold = 0.7  # Increased from 0.65
        else:
            # Model-specific heuristics (more conservative to reduce false positives)
            # CRITICAL: days_since_modification is the MOST IMPORTANT feature (0.3294 importance)
            # Dynamic age-based penalty: the older, the more penalized
            days_old = features.get('days_since_modification', 0)
            
            # Age-based scoring: moderate influence, balanced with other features
            # Age is important but not the sole factor - other features matter too
            if days_old <= 1:  # Brand new (0-1 days) - moderate boost
                score += 0.35  # Moderate boost for brand new models (balanced with other features)
            elif days_old <= 7:  # Very recent (1 week) - small boost
                # Linear decay from 0.35 to 0.18 over 6 days
                age_factor = (days_old - 1) / 6  # 0 to 1
                score += 0.35 - (age_factor * 0.17)  # 0.35 to 0.18
            elif days_old <= 30:  # Recent (1 month) - very small boost
                # Linear decay from 0.18 to 0.06 over 23 days
                age_factor = (days_old - 7) / 23  # 0 to 1
                score += 0.18 - (age_factor * 0.12)  # 0.18 to 0.06
            elif days_old <= 90:  # Fairly recent (3 months) - minimal boost
                # Linear decay from 0.06 to 0 over 60 days
                age_factor = (days_old - 30) / 60  # 0 to 1
                score += 0.06 - (age_factor * 0.06)  # 0.06 to 0
            elif days_old <= 180:  # Slightly old (6 months) - neutral to small penalty
                # Small linear penalty
                score -= 0.0002 * (days_old - 90)
            elif days_old <= 365:  # Moderately old (1 year) - moderate penalty
                # Linear penalty
                score -= 0.018 + 0.0003 * (days_old - 180)
            elif days_old <= 800:  # Old (more than 1 year) - heavier penalty
                # Linear penalty
                score -= 0.074 + 0.0004 * (days_old - 365)
            else:  # Very old (> 800 days) - significant but not overwhelming penalty
                # Moderate exponential penalty (reduced from before)
                # Formula: -0.0005 * (days_old - 800)^1.3 (reduced coefficient and exponent)
                # This means: 1000 days = -0.0005 * 200^1.3 ≈ -0.75
                #            1122 days = -0.0005 * 322^1.3 ≈ -1.35
                #            1500 days = -0.0005 * 700^1.3 ≈ -3.85
                age_penalty = -0.0005 * ((days_old - 800) ** 1.3)
                score += age_penalty
            
            # Base score from tags (more tags = better discoverability)
            # Require at least 5 tags for models to be considered popular
            num_tags = features.get('num_tags', 0)
            if num_tags >= 5:
                score += min(num_tags, 12) * 0.12  # Cap at 12 tags
            elif num_tags >= 3:
                score += num_tags * 0.08
            else:
                # Penalize models with very few tags
                score -= 0.15
            
            # Research/academic features (strong signal)
            score += features.get('num_arxiv_refs', 0) * 0.25
            
            # Model-specific features (important for models)
            score += features.get('has_transformers', 0) * 0.25
            score += features.get('has_safetensors', 0) * 0.12
            score += features.get('has_endpoints_compatible', 0) * 0.15
            score += features.get('has_autotrain_compatible', 0) * 0.1
            
            # License (having a license is good)
            score += features.get('has_license', 0) * 0.08
            
            # Base model relationships (finetunes are often popular)
            score += features.get('num_base_models', 0) * 0.05
            
            # Require strong signals - higher threshold
            threshold = 1.0
        
        prediction = 1 if score > threshold else 0
        
        # Normalize probability - ensure it's meaningful and conservative
        if is_dataset:
            # For datasets, use more conservative probability calculation
            # Only give high confidence if score is well above threshold
            if score > threshold * 1.3:
                probability = min(0.90, max(0.50, score / 1.2))
            elif score > threshold:
                probability = min(0.75, max(0.50, (score - threshold) / 0.5 + 0.5))
            else:
                probability = max(0.05, min(0.50, score / threshold * 0.5))
        else:
            # For models, use conservative probability
            if score > threshold * 1.2:
                probability = min(0.90, max(0.55, score / 1.8))
            elif score > threshold:
                probability = min(0.75, max(0.50, (score - threshold) / 0.6 + 0.5))
            else:
                probability = max(0.05, min(0.50, score / threshold * 0.5))
        
        note = 'Using heuristic prediction for dataset' if is_dataset else 'Using heuristic prediction (model not loaded)'
        
        return {
            'prediction': prediction,
            'probability': probability,
            'features': features,
            'note': note
        }

