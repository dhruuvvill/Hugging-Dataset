"""
Service for interacting with HuggingFace Hub API.
"""

import re
import logging
from huggingface_hub import HfApi
import os
from datetime import datetime
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HuggingFaceService:
    """Service for fetching and processing HuggingFace model information."""
    
    def __init__(self):
        # Read token from environment variable
        self.token = os.getenv("HF_TOKEN", "").strip()
        
        # Initialize API with or without token
        if self.token:
            logger.info(f"Using HF_TOKEN for authenticated access (token: {self.token[:10]}...)")
            self.api = HfApi(token=self.token)
        else:
            logger.info("Accessing HuggingFace without authentication (public repos only)")
            self.api = HfApi()  # No token, public access only
    
    def extract_model_id(self, url: str):
        """
        Extract model/dataset ID from HuggingFace URL.
        
        Args:
            url: HuggingFace URL (e.g., https://huggingface.co/owner/model-name)
            
        Returns:
            Tuple of (is_dataset: bool, entity_id: str) where entity_id is like "owner/name"
        """
        # Parse the URL to extract the path
        # Handle: https://huggingface.co/google/gemma-2-2b
        # or: https://huggingface.co/models/google/gemma-2-2b
        # or: https://huggingface.co/datasets/EleutherAI/hendrycks_math
        
        # Remove protocol and domain
        if "huggingface.co" in url:
            path = url.split("huggingface.co/")[-1].split("?")[0].strip("/")
        else:
            raise ValueError(f"Invalid HuggingFace URL: {url}")
        
        # Check if it's a dataset
        is_dataset = False
        if path.startswith("datasets/"):
            path = path[9:]  # Remove "datasets/" prefix
            is_dataset = True
        elif path.startswith("models/"):
            path = path[7:]  # Remove "models/" prefix
        
        # Split by / and reconstruct
        parts = path.split("/")
        if len(parts) >= 2:
            # Return owner/entity_name
            entity_id = f"{parts[0]}/{parts[1]}"
            return (is_dataset, entity_id)
        elif len(parts) == 1:
            # Single part - assume it's an entity ID
            return (is_dataset, parts[0])
        else:
            raise ValueError(f"Invalid HuggingFace URL: {url}. Expected format: https://huggingface.co/owner/entity-name")
    
    def get_model_info(self, model_id: str, is_dataset: bool = False) -> dict:
        """
        Get model or dataset information from HuggingFace Hub.
        
        Args:
            model_id: Model/Dataset ID (e.g., "google/gemma-2-2b")
            is_dataset: Whether this is a dataset (True) or model (False)
            
        Returns:
            Dictionary containing model/dataset information
        """
        try:
            # Try to get model or dataset info
            if is_dataset:
                info = self.api.dataset_info(model_id, files_metadata=False)
            else:
                info = self.api.model_info(model_id, files_metadata=False)
            
            # Extract the information we need
            model_data = {
                'id': model_id,
                'tags': getattr(info, 'tags', None) or [],
                'lastModified': str(getattr(info, 'lastModified', None)) if hasattr(info, 'lastModified') else None,
                'downloads': getattr(info, 'downloads', None) if hasattr(info, 'downloads') else None,
                'likes': getattr(info, 'likes', None) if hasattr(info, 'likes') else None,
            }
            
            return model_data
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific error cases
            if "Repository Not Found" in error_msg or "404" in error_msg:
                raise ValueError(
                    f"Model '{model_id}' not found on HuggingFace. "
                    "Please check the model ID (format: owner/model-name)."
                )
            
            elif "401" in error_msg or "authentication" in error_msg.lower():
                raise ValueError(
                    f"Authentication required for '{model_id}'. "
                    "This model may be private or gated. "
                    "To access it, set your HF_TOKEN environment variable."
                )
            
            elif "gated" in error_msg.lower() or "EulaAcceptedError" in error_msg:
                raise ValueError(
                    f"Model '{model_id}' requires acceptance of usage terms on HuggingFace. "
                    "Please accept the terms at https://huggingface.co/{} first.".format(model_id)
                )
            
            else:
                raise ValueError(
                    f"Could not fetch model info for '{model_id}'. "
                    f"Error: {error_msg} "
                    "Please ensure the model exists and is publicly accessible."
                )
    
    def extract_features(self, model_info: dict, is_dataset: bool = False) -> dict:
        """
        Extract features from model/dataset information using the same logic as training.
        
        Args:
            model_info: Dictionary containing model/dataset info
            is_dataset: Whether this is a dataset (True) or model (False)
            
        Returns:
            Dictionary of extracted features
        """
        # Extract tag features
        tags = model_info.get('tags', [])
        tag_features = self._extract_tag_features(tags, is_dataset)
        
        # Extract date features
        last_modified = model_info.get('lastModified', None)
        date_features = self._extract_date_features(last_modified)
        
        # Add popularity metrics as features (these are available before prediction)
        downloads = model_info.get('downloads', 0) or 0
        likes = model_info.get('likes', 0) or 0
        
        # Combine features
        features = {
            **tag_features, 
            **date_features,
            # Note: We don't include downloads/likes as features since they're the target
            # But we can use them for heuristics if model is not loaded
        }
        
        return features
    
    def _extract_tag_features(self, tags: list, is_dataset: bool = False) -> dict:
        """Extract features from tags - focusing on top important features from model analysis."""
        if not tags:
            return {
                'num_tags': 0,
                'has_library_datasets': 0,
                'has_task_category': 0,
                'has_language': 0,
                'has_license': 0,
                'has_modality': 0,
                'has_size_category': 0,
                'num_base_models': 0,
                'has_transformers': 0,
                'has_safetensors': 0,
                'num_arxiv_refs': 0,
                'has_arxiv_ref': 0,
                'has_endpoints_compatible': 0,
                'has_autotrain_compatible': 0,
                'num_languages': 0,
                'has_english': 0,
                'has_open_license': 0,
                'num_dataset_refs': 0,
                'has_dataset_ref': 0,
            }
        
        tag_str_lower = str(tags).lower()
        
        # Top features from model analysis (by importance):
        # 1. num_tags (0.1679) - Most important after days_since_modification
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
            # Top priority features
            'num_tags': len(tags),  # Most important feature (0.1679 importance)
            'has_transformers': 1 if 'transformers' in tag_str_lower else 0,  # 0.0779 importance
            'has_endpoints_compatible': 1 if 'endpoints_compatible' in tag_str_lower else 0,  # 0.0576 importance
            'has_autotrain_compatible': 1 if 'autotrain_compatible' in tag_str_lower else 0,  # 0.0250 importance
            'has_safetensors': 1 if 'safetensors' in tag_str_lower else 0,  # 0.0243 importance
            'has_license': 1 if any('license:' in str(t) for t in tags) else 0,  # 0.0185 importance
            
            # Language features
            'num_languages': sum(1 for t in tags if str(t).lower() in ['en', 'zh', 'fr', 'de', 'es', 'ja', 'ko', 'pt', 'it', 'ar', 'hi', 'th'] or 'language:' in str(t).lower()),  # 0.0309 importance
            'has_english': 1 if any('en' == str(t).lower() or 'language:en' in str(t).lower() for t in tags) else 0,  # 0.0119 importance
            
            # License features
            'has_open_license': 1 if any('license:apache' in str(t).lower() or 'license:mit' in str(t).lower() for t in tags) else 0,  # 0.0118 importance
            
            # Research features
            'num_arxiv_refs': sum(1 for t in tags if 'arxiv:' in str(t)),  # 0.0103 importance
            'has_arxiv_ref': 1 if any('arxiv:' in str(t) for t in tags) else 0,  # 0.0100 importance
            
            # Dataset features
            'num_dataset_refs': sum(1 for t in tags if 'dataset:' in str(t)),  # 0.0125 importance
            'has_dataset_ref': 1 if any('dataset:' in str(t) for t in tags) else 0,
            
            # Other features (still useful)
            'has_library_datasets': sum(1 for t in tags if 'library:' in str(t) and 'datasets' in str(t)),
            'has_task_category': sum(1 for t in tags if 'task_categories:' in str(t)),
            'has_language': sum(1 for t in tags if 'language:' in str(t)),
            'has_modality': sum(1 for t in tags if 'modality:' in str(t)),
            'has_size_category': sum(1 for t in tags if 'size_categories:' in str(t)),
            'num_base_models': sum(1 for t in tags if 'base_model:' in str(t)),
        }
        
        # For datasets, library:datasets is expected and should be counted positively
        if is_dataset:
            # Datasets typically have library:datasets tag, which is good
            features['has_library_datasets'] = 1 if any('library:' in str(t) and 'datasets' in str(t) for t in tags) else 0
        
        # Extract specific values for pipeline_tag and license_type
        main_task = None
        license_type = None
        pipeline_tag = None
        
        for tag in tags:
            tag_str = str(tag)
            if tag_str.startswith('task_categories:'):
                main_task = tag_str.split(':', 1)[1]
            elif tag_str.startswith('license:'):
                license_type = tag_str.split(':', 1)[1]
            elif tag_str in ['text-generation', 'image-to-text', 'text-to-image', 'image-to-image', 
                           'text-to-video', 'feature-extraction', 'sentence-similarity']:
                pipeline_tag = tag_str
        
        if main_task:
            features['main_task'] = main_task
        if license_type:
            features['license_type'] = license_type
        else:
            features['license_type'] = 'unknown'  # Negative signal (0.0205 importance)
        
        if pipeline_tag:
            features['pipeline_tag'] = pipeline_tag
        else:
            features['pipeline_tag'] = 'other'  # Negative signal (0.0297 importance)
        
        return features
    
    def _extract_date_features(self, last_modified: str) -> dict:
        """Extract features from last modified date."""
        if not last_modified:
            return {'days_since_modification': 0, 'is_recent': 0}
        
        try:
            # Parse ISO format
            if 'T' in str(last_modified):
                mod_date = datetime.fromisoformat(str(last_modified).replace('Z', '+00:00'))
            else:
                mod_date = pd.to_datetime(last_modified)
            
            now = datetime.now()
            days_diff = (now - mod_date.replace(tzinfo=None)).days
            
            return {
                'days_since_modification': days_diff,
                'is_recent': 1 if days_diff < 90 else 0,
            }
        except:
            return {'days_since_modification': 0, 'is_recent': 0}

