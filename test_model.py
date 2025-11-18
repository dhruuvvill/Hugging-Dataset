#!/usr/bin/env python3
"""
Test the model with randomly generated data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from model import prepare_features_advanced
import ast
import json

def generate_random_tags(num_tags):
    """Generate random tags for testing."""
    library_options = ['transformers', 'diffusers', 'safetensors', 'comfyUI']
    task_options = ['text-generation', 'image-to-text', 'text-to-image', 'image-classification']
    license_options = ['mit', 'apache-2.0', 'other']
    language_options = ['en', 'zh', 'multilingual']
    
    tags = []
    
    # Randomly add library
    if np.random.random() > 0.3:
        tags.append(np.random.choice(library_options))
    
    # Randomly add task
    if np.random.random() > 0.4:
        tags.append(f"task_categories:{np.random.choice(task_options)}")
    
    # Randomly add license
    if np.random.random() > 0.2:
        tags.append(f"license:{np.random.choice(license_options)}")
    
    # Randomly add language
    if np.random.random() > 0.3:
        tags.append(np.random.choice(language_options))
    
    # Randomly add other tags
    other_tags = ['arxiv:12345.67890', 'region:us', 'conversational']
    for tag in other_tags:
        if np.random.random() > 0.5:
            tags.append(tag)
    
    return json.dumps(tags)

def generate_random_data(n_samples=1000):
    """Generate random model data for testing."""
    data = []
    
    # Generate random IDs, dates, and popularity metrics
    for i in range(n_samples):
        # Generate downloads with some structure (popular models get more)
        base_downloads = np.random.lognormal(5, 2)
        downloads = int(max(1, base_downloads))
        
        # Likes correlated with downloads
        likes = int(max(1, downloads * np.random.uniform(0.01, 0.05)))
        
        # Generate random dates (last 2 years)
        days_ago = np.random.randint(0, 730)
        last_modified = f"2023-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}T{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}+00:00"
        
        # Generate ID
        model_id = f"test_model_{i}/model-{i}"
        
        # Generate tags
        tags = generate_random_tags(np.random.randint(3, 10))
        
        data.append({
            'id': model_id,
            'downloads': downloads,
            'likes': likes,
            'lastModified': last_modified,
            'tags': tags
        })
    
    return pd.DataFrame(data)

def main():
    print("=" * 70)
    print("Testing Model with Randomly Generated Data")
    print("=" * 70)
    
    # Generate random data
    print("\n1. Generating random data...")
    df = generate_random_data(n_samples=1000)
    print(f"   Created {len(df)} random model records")
    
    # Prepare features
    print("\n2. Extracting features...")
    X = prepare_features_advanced(df)
    y = df['downloads']
    
    # Create binary classification
    threshold = np.percentile(y, 80)
    y_binary = (y >= threshold).astype(int)
    
    print(f"\n3. Classification threshold: {threshold:.0f} downloads")
    print(f"   Class distribution - Low: {sum(y_binary == 0)}, High: {sum(y_binary == 1)}")
    
    # Split and train
    print("\n4. Training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    model = RandomForestClassifier(
        n_estimators=100,  # Increased from 50
        max_depth=12,      # Increased from 8
        min_samples_split=5,  # Decreased from 10
        min_samples_leaf=2,  # Decreased from 5
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n5. Results:")
    print(f"\n   Accuracy: {accuracy:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'High']))
    
    # Feature importance
    print("\n   Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        print(f"     {i+1}. {row['feature']:40s}: {row['importance']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Test completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()

