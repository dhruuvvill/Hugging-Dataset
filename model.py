#!/usr/bin/env python3
"""
Predict Hugging Face dataset popularity using metadata features.
Supports both regression (predicting downloads/likes) and classification (high/low popularity).
"""

import pandas as pd
import numpy as np
import ast
import json
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib


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


def extract_tags_features(tags_str):
    """Extract features from tags (works for both models and datasets)."""
    tags = parse_tags(tags_str)
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
        }
    
    features = {
        'num_tags': len(tags),
        'has_library_datasets': sum(1 for t in tags if 'library:' in str(t) and 'datasets' in str(t)),
        'has_task_category': sum(1 for t in tags if 'task_categories:' in str(t)),
        'has_language': sum(1 for t in tags if 'language:' in str(t)),
        'has_license': sum(1 for t in tags if 'license:' in str(t)),
        'has_modality': sum(1 for t in tags if 'modality:' in str(t)),
        'has_size_category': sum(1 for t in tags if 'size_categories:' in str(t)),
        'num_base_models': sum(1 for t in tags if 'base_model:' in str(t)),
        'has_transformers': 1 if 'transformers' in str(tags).lower() else 0,
        'has_safetensors': 1 if 'safetensors' in str(tags).lower() else 0,
        'num_arxiv_refs': sum(1 for t in tags if 'arxiv:' in str(t)),
    }
    
    # Extract specific values
    for tag in tags:
        tag_str = str(tag)
        if tag_str.startswith('task_categories:'):
            features['main_task'] = tag_str.split(':', 1)[1]
        elif tag_str.startswith('license:'):
            features['license_type'] = tag_str.split(':', 1)[1]
    
    return features


def extract_date_features(last_modified):
    """Extract features from last modified date."""
    if pd.isna(last_modified):
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


def extract_description_features(description):
    """Extract features from description text."""
    if pd.isna(description):
        return {'description_length': 0, 'has_code_example': 0, 'has_link': 0}
    
    desc_str = str(description).lower()
    
    return {
        'description_length': len(description) if not pd.isna(description) else 0,
        'has_code_example': 1 if ('```' in desc_str or 'code' in desc_str) else 0,
        'has_link': 1 if ('http' in desc_str or 'github' in desc_str) else 0,
    }


def prepare_features(df):
    """Prepare all features for training."""
    # Copy only the feature columns we need (avoid data leakage)
    feature_df = df.copy()
    
    # Extract tag features
    print("Extracting tag features...")
    tag_features = feature_df['tags'].apply(extract_tags_features).tolist()
    tag_df = pd.DataFrame(tag_features)
    
    # Extract date features
    print("Extracting date features...")
    date_features = feature_df['lastModified'].apply(extract_date_features).tolist()
    date_df = pd.DataFrame(date_features)
    
    # Extract description features (if available)
    print("Extracting description features...")
    if 'description' in feature_df.columns:
        desc_features = feature_df.get('description', pd.Series()).apply(extract_description_features).tolist()
        desc_df = pd.DataFrame(desc_features)
    else:
        # Create empty dataframe with default values
        desc_df = pd.DataFrame({
            'description_length': [0] * len(feature_df),
            'has_code_example': [0] * len(feature_df),
            'has_link': [0] * len(feature_df)
        })
    
    # Combine all features
    combined_features = pd.concat([
        tag_df,
        date_df,
        desc_df
    ], axis=1)
    
    # Add author as categorical feature
    if 'author' in feature_df.columns:
        combined_features['author'] = feature_df['author']
    
    return combined_features


def prepare_features_advanced(df):
    """Prepare features with one-hot encoding for categorical variables."""
    features = prepare_features(df)
    
    # One-hot encode categorical columns (only those that exist)
    categorical_cols = ['main_task', 'license_type', 'author']
    categorical_cols = [col for col in categorical_cols if col in features.columns]
    
    for col in categorical_cols:
        # Take top 20 most common values to avoid too many features
        top_values = features[col].value_counts().head(20).index
        for val in top_values:
            features[f'{col}_{val}'] = (features[col] == val).astype(int)
        features = features.drop(columns=[col])
    
    # Fill NaN values
    features = features.fillna(0)
    
    return features


def train_popularity_model(data_path, model_type='regression', target='downloads'):
    """
    Train a model to predict dataset popularity.
    
    Args:
        data_path: Path to CSV file
        model_type: 'regression' (predict exact values) or 'classification' (high/low)
        target: 'downloads' or 'likes'
    """
    print(f"\n{'='*60}")
    print(f"Loading data from: {data_path}")
    print(f"{'='*60}\n")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Filter out missing targets
    df = df[df[target].notna() & (df[target] > 0)]
    
    print(f"Loaded {len(df)} datasets with {target} data")
    print(f"Target distribution:\n{df[target].describe()}\n")
    
    # Explicitly exclude target columns and other leakage columns
    # Make sure we only use metadata that should be available before the dataset becomes popular
    leakage_columns = ['downloads', 'likes', 'trendingScore', '_id', 'sha', 'key']
    
    # Create a clean df without leakage
    clean_df = df.copy()
    
    # Prepare features (this will extract from tags, lastModified, description, author)
    X = prepare_features_advanced(clean_df)
    y = df[target]
    
    # Debug: check for any feature that might be leaking the target
    print(f"Number of features: {X.shape[1]}")
    print(f"Features: {list(X.columns)[:20]}...")  # Print first 20 feature names
    
    # For classification, create binary target (top 20% vs others)
    if model_type == 'classification':
        threshold = np.percentile(y, 80)
        y_binary = (y >= threshold).astype(int)
        
        # Check class distribution
        class_counts = np.bincount(y_binary)
        print(f"Classification threshold: {threshold:.0f}")
        print(f"Class distribution - Low (0): {class_counts[0]}, High (1): {class_counts[1]}")
        
        # If we have too few samples or imbalanced classes, adjust
        if len(class_counts) < 2 or min(class_counts) < 50:
            print("⚠️ Warning: Very imbalanced classes. Using 50th percentile instead.")
            threshold = np.median(y)
            y_binary = (y >= threshold).astype(int)
            class_counts = np.bincount(y_binary)
            print(f"New threshold: {threshold:.0f}")
            print(f"Class distribution - Low (0): {class_counts[0]}, High (1): {class_counts[1]}")
        
        y = y_binary
        print()
    
    # Split data with stratification for classification
    if model_type == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")
    
    # Train model
    if model_type == 'regression':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"{'='*60}")
        print("Regression Results:")
        print(f"{'='*60}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"\nTarget mean: {y_test.mean():.2f}")
        print(f"Target std: {y_test.std():.2f}")
        
    else:  # classification
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
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
        
        print(f"{'='*60}")
        print("Classification Results:")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low', 'High']))
    
    # Feature importance
    print(f"\n{'='*60}")
    print("Top 10 Most Important Features:")
    print(f"{'='*60}")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:40s}: {row['importance']:.4f}")
    
    # Save model
    model_path = f'models_popularity/popularity_{model_type}_{target}.joblib'
    import os
    os.makedirs('models_popularity', exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model, X_test, y_test, y_pred


if __name__ == "__main__":
    # Train a single classification model to predict model popularity
    data_path = "hf_features/index_models.csv"
    
    print("="*70)
    print("Training HuggingFace Model Popularity Predictor")
    print("="*70)
    
    print("\nTraining classification model to predict high vs low popularity...")
    train_popularity_model(
        data_path, 
        model_type='classification', 
        target='downloads'
    )
    
    print("\n" + "="*70)
    print("✅ Model trained successfully!")
    print("="*70)