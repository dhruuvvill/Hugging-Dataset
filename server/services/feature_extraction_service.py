"""
Service for comprehensive feature extraction from HuggingFace models.
Uses the same feature extraction logic as feature_extraction_model.py
"""

import ast
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class FeatureExtractionService:
    """Service for extracting comprehensive features from model metadata."""
    
    def __init__(self):
        self.model = None
        self.feature_order = None
        self.load_model()
    
    def load_model(self):
        """Load the trained feature extraction model if available."""
        try:
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent
            
            possible_paths = [
                project_root / "models_popularity" / "popularity_classification_top_bottom_20.joblib",
                Path("models_popularity/popularity_classification_top_bottom_20.joblib"),
                Path("../models_popularity/popularity_classification_top_bottom_20.joblib"),
            ]
            
            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = path.resolve()
                    break
            
            if model_path and model_path.exists():
                logger.info(f"Loading feature extraction model from {model_path}")
                self.model = joblib.load(model_path)
                
                # Try to load feature importance
                importance_path = model_path.parent / "feature_importance_top_bottom_20.csv"
                if importance_path.exists():
                    feature_importance_df = pd.read_csv(importance_path)
                    self.feature_order = feature_importance_df['feature'].tolist()
                    logger.info(f"Loaded feature order with {len(self.feature_order)} features")
            
        except Exception as e:
            logger.warning(f"Could not load feature extraction model: {e}")
            logger.info("Feature extraction will work without model for analysis")
    
    def parse_tags(self, tags_str):
        """Parse tags string into a list of tags."""
        if pd.isna(tags_str) or not tags_str:
            return []
        try:
            # If already a list
            if isinstance(tags_str, list):
                return tags_str
            # Try JSON parsing first
            tags = json.loads(tags_str)
            return tags if isinstance(tags, list) else []
        except:
            try:
                # Try Python literal eval
                tags = ast.literal_eval(tags_str)
                return tags if isinstance(tags, list) else []
            except:
                return []
    
    def extract_comprehensive_features(self, tags_str, last_modified, model_id):
        """Extract comprehensive features from tags and metadata - prioritizing top features from model analysis."""
        tags = self.parse_tags(tags_str)
        tag_str_lower = str(tags).lower() if tags else ""
        
        # Top features from model analysis (by importance):
        # 1. num_tags (0.1679) - Most important feature
        # 2. has_transformers (0.0779)
        # 3. has_endpoints_compatible (0.0576)
        # 4. num_languages (0.0309)
        # 5. has_autotrain_compatible (0.0250)
        # 6. has_safetensors (0.0243)
        # 7. has_license (0.0185)
        # 8. num_dataset_refs (0.0125)
        # 9. has_english (0.0119)
        # 10. has_open_license (0.0118)
        # 11. num_arxiv_refs (0.0103)
        # 12. has_arxiv_ref (0.0100)
        
        features = {
            # TOP PRIORITY FEATURES (from model analysis)
            'num_tags': len(tags),  # 0.1679 importance - MOST IMPORTANT
            'has_transformers': 1 if 'transformers' in tag_str_lower else 0,  # 0.0779 importance
            'has_endpoints_compatible': 1 if 'endpoints_compatible' in tag_str_lower else 0,  # 0.0576 importance
            'num_languages': sum(1 for t in tags if str(t).lower() in ['en', 'zh', 'fr', 'de', 'es', 'ja', 'ko', 'pt', 'it', 'ar', 'hi', 'th'] or 'language:' in str(t).lower()),  # 0.0309 importance
            'has_autotrain_compatible': 1 if 'autotrain_compatible' in tag_str_lower else 0,  # 0.0250 importance
            'has_safetensors': 1 if 'safetensors' in tag_str_lower else 0,  # 0.0243 importance
            'has_license': 1 if any('license:' in str(t) for t in tags) else 0,  # 0.0185 importance
            'num_dataset_refs': sum(1 for t in tags if 'dataset:' in str(t)),  # 0.0125 importance
            'has_english': 1 if any('en' == str(t).lower() or 'language:en' in str(t).lower() for t in tags) else 0,  # 0.0119 importance
            'has_open_license': 1 if any('license:apache' in str(t).lower() or 'license:mit' in str(t).lower() for t in tags) else 0,  # 0.0118 importance
            'num_arxiv_refs': sum(1 for t in tags if 'arxiv:' in str(t)),  # 0.0103 importance
            'has_arxiv_ref': 1 if any('arxiv:' in str(t) for t in tags) else 0,  # 0.0100 importance
            
            # Additional important features
            'has_dataset_ref': 1 if any('dataset:' in str(t) for t in tags) else 0,
            'is_finetune': 1 if any('finetune:' in str(t) for t in tags) else 0,  # 0.0110 importance
            'has_diffusers': 1 if 'diffusers' in tag_str_lower else 0,  # 0.0110 importance
            'has_gguf': 1 if 'gguf' in tag_str_lower else 0,  # 0.0108 importance
            
            # Library features
            'has_sentence_transformers': 1 if 'sentence-transformers' in tag_str_lower else 0,
            'has_lora': 1 if 'lora' in tag_str_lower else 0,
            
            # Task/application features
            'has_text_generation': 1 if 'text-generation' in tag_str_lower else 0,
            'has_image_to_text': 1 if 'image-to-text' in tag_str_lower else 0,
            'has_text_to_image': 1 if 'text-to-image' in tag_str_lower else 0,
            'has_image_to_image': 1 if 'image-to-image' in tag_str_lower else 0,
            'has_text_to_video': 1 if 'text-to-video' in tag_str_lower else 0,
            'has_ocr': 1 if 'ocr' in tag_str_lower else 0,
            'has_conversational': 1 if 'conversational' in tag_str_lower else 0,
            'has_feature_extraction': 1 if 'feature-extraction' in tag_str_lower else 0,
            
            # Technical features
            'has_custom_code': 1 if 'custom_code' in tag_str_lower else 0,
            'has_text_generation_inference': 1 if 'text-generation-inference' in tag_str_lower else 0,
            
            # Base model features
            'num_base_models': sum(1 for t in tags if 'base_model:' in str(t)),
            'has_base_model': 1 if any('base_model:' in str(t) for t in tags) else 0,
            'is_adapter': 1 if any('adapter:' in str(t) for t in tags) else 0,
            'is_quantized': 1 if any('quantized:' in str(t) for t in tags) else 0,
            
            # License features
            'has_apache_license': 1 if any('license:apache' in str(t).lower() for t in tags) else 0,
            'has_mit_license': 1 if any('license:mit' in str(t).lower() for t in tags) else 0,
            
            # Language features
            'has_multilingual': 1 if 'multilingual' in tag_str_lower else 0,
            
            # Organization/author features
            'is_from_org': 1 if '/' in str(model_id) else 0,
            'org_name_length': len(str(model_id).split('/')[0]) if '/' in str(model_id) else 0,
            
            # Region features
            'has_region_us': 1 if any('region:us' in str(t) for t in tags) else 0,
        }
        
        # Extract specific license type (negative signal if unknown - 0.0205 importance)
        license_type = 'unknown'
        for tag in tags:
            tag_str = str(tag)
            if tag_str.startswith('license:'):
                license_type = tag_str.split(':', 1)[1].lower()
                break
        features['license_type'] = license_type  # 'unknown' is a negative signal
        
        # Extract pipeline tag (negative signal if 'other' - 0.0297 importance)
        pipeline_tag = 'other'
        for tag in tags:
            tag_str = str(tag)
            if tag_str in ['text-generation', 'image-to-text', 'text-to-image', 'image-to-image', 
                           'text-to-video', 'feature-extraction', 'sentence-similarity']:
                pipeline_tag = tag_str
                break
        features['pipeline_tag'] = pipeline_tag  # 'other' is a negative signal
        
        # Date features
        if pd.isna(last_modified) or not last_modified:
            features['days_since_modification'] = 0
            features['is_recent'] = 0
            features['is_very_recent'] = 0
        else:
            try:
                if 'T' in str(last_modified):
                    mod_date = datetime.fromisoformat(str(last_modified).replace('Z', '+00:00'))
                else:
                    mod_date = pd.to_datetime(last_modified)
                
                now = datetime.now()
                days_diff = (now - mod_date.replace(tzinfo=None)).days
                features['days_since_modification'] = days_diff
                features['is_recent'] = 1 if days_diff < 90 else 0
                features['is_very_recent'] = 1 if days_diff < 30 else 0
            except:
                features['days_since_modification'] = 0
                features['is_recent'] = 0
                features['is_very_recent'] = 0
        
        return features
    
    def prepare_features_for_model(self, features_dict):
        """Prepare features in the format expected by the trained model."""
        # Convert to DataFrame
        df = pd.DataFrame([features_dict])
        
        # One-hot encode categorical variables
        categorical_cols = ['license_type', 'pipeline_tag']
        for col in categorical_cols:
            if col in df.columns:
                # Get top 10 most common values (we'll use a default set)
                top_values = ['apache-2.0', 'mit', 'unknown', 'other', 'text-generation', 
                             'image-to-text', 'text-to-image', 'feature-extraction']
                for val in top_values:
                    df[f'{col}_{val}'] = (df[col] == val).astype(int)
                df = df.drop(columns=[col])
        
        # Fill NaN values
        df = df.fillna(0)
        
        # If we have feature order, reorder columns
        if self.feature_order:
            # Add missing columns with 0
            for col in self.feature_order:
                if col not in df.columns:
                    df[col] = 0
            # Reorder and select only known features
            df = df[[col for col in self.feature_order if col in df.columns]]
        
        return df
    
    def predict_popularity(self, features_dict):
        """Predict popularity using the trained model."""
        if self.model is None:
            return None
        
        try:
            df = self.prepare_features_for_model(features_dict)
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(df)[0]
                prediction = self.model.predict(df)[0]
                probability = float(probabilities[1] if len(probabilities) > 1 else probabilities[0])
            else:
                prediction = self.model.predict(df)[0]
                probability = 0.5
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'is_popular': bool(prediction == 1)
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance from the trained model."""
        if self.model is None or self.feature_order is None:
            return None
        
        try:
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent
            importance_path = project_root / "models_popularity" / "feature_importance_top_bottom_20.csv"
            
            if importance_path.exists():
                df = pd.read_csv(importance_path)
                return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error loading feature importance: {e}")
        
        return None

