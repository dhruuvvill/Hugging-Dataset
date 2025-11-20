#!/usr/bin/env python3
"""
Model to extract features that make Hugging Face models popular or unpopular.
Uses top 20% and bottom 20% based on likes and downloads.
"""

import pandas as pd
import numpy as np
import ast
import json
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


def parse_tags(tags_str):
    """Parse tags string into a list of tags."""
    if pd.isna(tags_str):
        return []
    try:
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


def extract_comprehensive_features(tags_str, last_modified, model_id):
    """Extract comprehensive features from tags and metadata."""
    tags = parse_tags(tags_str)
    tag_str_lower = str(tags).lower() if tags else ""
    
    features = {
        # Basic tag counts
        'num_tags': len(tags),
        
        # Library features
        'has_transformers': 1 if 'transformers' in tag_str_lower else 0,
        'has_safetensors': 1 if 'safetensors' in tag_str_lower else 0,
        'has_diffusers': 1 if 'diffusers' in tag_str_lower else 0,
        'has_sentence_transformers': 1 if 'sentence-transformers' in tag_str_lower else 0,
        'has_gguf': 1 if 'gguf' in tag_str_lower else 0,
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
        'has_autotrain_compatible': 1 if 'autotrain_compatible' in tag_str_lower else 0,
        'has_endpoints_compatible': 1 if 'endpoints_compatible' in tag_str_lower else 0,
        'has_text_generation_inference': 1 if 'text-generation-inference' in tag_str_lower else 0,
        
        # Base model features
        'num_base_models': sum(1 for t in tags if 'base_model:' in str(t)),
        'has_base_model': 1 if any('base_model:' in str(t) for t in tags) else 0,
        'is_finetune': 1 if any('finetune:' in str(t) for t in tags) else 0,
        'is_adapter': 1 if any('adapter:' in str(t) for t in tags) else 0,
        'is_quantized': 1 if any('quantized:' in str(t) for t in tags) else 0,
        
        # Research/academic features
        'num_arxiv_refs': sum(1 for t in tags if 'arxiv:' in str(t)),
        'has_arxiv_ref': 1 if any('arxiv:' in str(t) for t in tags) else 0,
        
        # License features
        'has_license': 1 if any('license:' in str(t) for t in tags) else 0,
        'has_apache_license': 1 if any('license:apache' in str(t).lower() for t in tags) else 0,
        'has_mit_license': 1 if any('license:mit' in str(t).lower() for t in tags) else 0,
        'has_open_license': 1 if any('license:apache' in str(t).lower() or 'license:mit' in str(t).lower() for t in tags) else 0,
        
        # Language features
        'has_multilingual': 1 if 'multilingual' in tag_str_lower else 0,
        'has_english': 1 if any('en' == str(t).lower() or 'language:en' in str(t).lower() for t in tags) else 0,
        'num_languages': sum(1 for t in tags if str(t).lower() in ['en', 'zh', 'fr', 'de', 'es', 'ja', 'ko', 'pt', 'it', 'ar', 'hi', 'th'] or 'language:' in str(t).lower()),
        
        # Dataset features
        'num_dataset_refs': sum(1 for t in tags if 'dataset:' in str(t)),
        'has_dataset_ref': 1 if any('dataset:' in str(t) for t in tags) else 0,
        
        # Organization/author features
        'is_from_org': 1 if '/' in str(model_id) else 0,
        'org_name_length': len(str(model_id).split('/')[0]) if '/' in str(model_id) else 0,
        
        # Region features
        'has_region_us': 1 if any('region:us' in str(t) for t in tags) else 0,
    }
    
    # Extract specific license type
    for tag in tags:
        tag_str = str(tag)
        if tag_str.startswith('license:'):
            license_type = tag_str.split(':', 1)[1].lower()
            features['license_type'] = license_type
            break
    else:
        features['license_type'] = 'unknown'
    
    # Extract pipeline tag
    for tag in tags:
        tag_str = str(tag)
        if tag_str in ['text-generation', 'image-to-text', 'text-to-image', 'image-to-image', 
                       'text-to-video', 'feature-extraction', 'sentence-similarity']:
            features['pipeline_tag'] = tag_str
            break
    else:
        features['pipeline_tag'] = 'other'
    
    # Date features
    if pd.isna(last_modified):
        features['days_since_modification'] = 0
        features['is_recent'] = 0
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


def prepare_features_df(df):
    """Prepare feature dataframe from raw data."""
    print("Extracting features from tags and metadata...")
    
    feature_list = []
    for idx, row in df.iterrows():
        features = extract_comprehensive_features(
            row.get('tags', ''),
            row.get('lastModified', None),
            row.get('id', '')
        )
        feature_list.append(features)
    
    features_df = pd.DataFrame(feature_list)
    
    # One-hot encode categorical variables
    categorical_cols = ['license_type', 'pipeline_tag']
    for col in categorical_cols:
        if col in features_df.columns:
            # Take top 10 most common values
            top_values = features_df[col].value_counts().head(10).index
            for val in top_values:
                features_df[f'{col}_{val}'] = (features_df[col] == val).astype(int)
            features_df = features_df.drop(columns=[col])
    
    # Fill NaN values
    features_df = features_df.fillna(0)
    
    return features_df


def main():
    print("="*70)
    print("Hugging Face Model Popularity Feature Extraction")
    print("="*70)
    print()
    
    # Load data
    csv_path = "/Users/dhruvil1904/Desktop/HuggingFace/hf_features/index_models.csv"
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} models")
    print()
    
    # Clean data - remove rows with missing likes or downloads
    df = df.dropna(subset=['likes', 'downloads'])
    df = df[(df['likes'] >= 0) & (df['downloads'] >= 0)]
    print(f"After cleaning: {len(df)} models with valid likes and downloads")
    print()
    
    # Create a combined popularity score
    # Normalize likes and downloads to 0-1 scale, then combine
    likes_normalized = (df['likes'] - df['likes'].min()) / (df['likes'].max() - df['likes'].min() + 1)
    downloads_normalized = (df['downloads'] - df['downloads'].min()) / (df['downloads'].max() - df['downloads'].min() + 1)
    
    # Combined score (weighted average: 40% likes, 60% downloads)
    df['popularity_score'] = 0.4 * likes_normalized + 0.6 * downloads_normalized
    
    # Sort by popularity score
    df = df.sort_values('popularity_score', ascending=False).reset_index(drop=True)
    
    # Get top 20% and bottom 20%
    n_total = len(df)
    n_top = int(0.2 * n_total)
    n_bottom = int(0.2 * n_total)
    
    top_20 = df.head(n_top).copy()
    bottom_20 = df.tail(n_bottom).copy()
    
    print(f"Top 20%: {n_top} models")
    print(f"  Likes range: {top_20['likes'].min():.0f} - {top_20['likes'].max():.0f}")
    print(f"  Downloads range: {top_20['downloads'].min():.0f} - {top_20['downloads'].max():.0f}")
    print()
    print(f"Bottom 20%: {n_bottom} models")
    print(f"  Likes range: {bottom_20['likes'].min():.0f} - {bottom_20['likes'].max():.0f}")
    print(f"  Downloads range: {bottom_20['downloads'].min():.0f} - {bottom_20['downloads'].max():.0f}")
    print()
    
    # Create binary labels: 1 for popular (top 20%), 0 for unpopular (bottom 20%)
    top_20['is_popular'] = 1
    bottom_20['is_popular'] = 0
    
    # Combine top and bottom
    analysis_df = pd.concat([top_20, bottom_20], ignore_index=True)
    
    print(f"Total samples for analysis: {len(analysis_df)}")
    print(f"  Popular (1): {len(top_20)}")
    print(f"  Unpopular (0): {len(bottom_20)}")
    print()
    
    # Extract features
    X = prepare_features_df(analysis_df)
    y = analysis_df['is_popular']
    
    print(f"Extracted {X.shape[1]} features")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()
    
    # Train Random Forest Classifier
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print()
    print("="*70)
    print("Model Performance")
    print("="*70)
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Unpopular', 'Popular']))
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    
    # Feature importance analysis
    print("="*70)
    print("Top 30 Most Important Features for Popularity")
    print("="*70)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(30).iterrows():
        print(f"{row['feature']:50s}: {row['importance']:.4f}")
    
    print()
    
    # Analyze feature differences between popular and unpopular
    print("="*70)
    print("Feature Comparison: Popular vs Unpopular Models")
    print("="*70)
    
    popular_features = X[analysis_df['is_popular'] == 1]
    unpopular_features = X[analysis_df['is_popular'] == 0]
    
    comparison = []
    for col in X.columns:
        pop_mean = popular_features[col].mean()
        unpop_mean = unpopular_features[col].mean()
        diff = pop_mean - unpop_mean
        importance = feature_importance[feature_importance['feature'] == col]['importance'].values[0]
        
        comparison.append({
            'feature': col,
            'popular_mean': pop_mean,
            'unpopular_mean': unpop_mean,
            'difference': diff,
            'importance': importance
        })
    
    comparison_df = pd.DataFrame(comparison).sort_values('importance', ascending=False)
    
    print("\nTop 20 Features with Largest Differences:")
    print("-" * 100)
    print(f"{'Feature':<50} {'Popular':<12} {'Unpopular':<12} {'Diff':<12} {'Importance':<12}")
    print("-" * 100)
    for _, row in comparison_df.head(20).iterrows():
        print(f"{row['feature']:<50} {row['popular_mean']:<12.4f} {row['unpopular_mean']:<12.4f} "
              f"{row['difference']:<12.4f} {row['importance']:<12.4f}")
    
    print()
    
    # Save model
    os.makedirs('models_popularity', exist_ok=True)
    model_path = 'models_popularity/popularity_classification_top_bottom_20.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save feature importance
    feature_importance_path = 'models_popularity/feature_importance_top_bottom_20.csv'
    feature_importance.to_csv(feature_importance_path, index=False)
    print(f"Feature importance saved to: {feature_importance_path}")
    
    # Save comparison
    comparison_path = 'models_popularity/feature_comparison_top_bottom_20.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Feature comparison saved to: {comparison_path}")
    
    print()
    print("="*70)
    print("✅ Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()