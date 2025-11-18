#!/usr/bin/env python3
"""
Quick demo to test the model with a subset of data.
"""

import pandas as pd
import numpy as np
from model import prepare_features_advanced, train_popularity_model

def demo():
    """Run a quick demo on a small subset of data."""
    print("=" * 70)
    print("DEMO: Testing Popularity Prediction Model")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv("hf_features/index_models.csv")
    
    # Filter to valid data
    df = df[df['downloads'].notna() & (df['downloads'] > 0)]
    
    # Use a sample for faster testing
    df_sample = df.sample(min(5000, len(df)), random_state=42)
    
    print(f"   Loaded {len(df)} total datasets")
    print(f"   Using {len(df_sample)} datasets for demo\n")
    
    # Prepare features
    print("2. Extracting features...")
    X = prepare_features_advanced(df_sample)
    y = df_sample['downloads']
    
    print(f"   Created {X.shape[1]} features from metadata\n")
    
    # Create binary classification
    print("3. Creating popularity classes...")
    threshold = np.percentile(y, 80)
    y_binary = (y >= threshold).astype(int)
    
    print(f"   Popularity threshold: {threshold:.0f} downloads")
    print(f"   Low popularity: {sum(y_binary == 0)} datasets")
    print(f"   High popularity: {sum(y_binary == 1)} datasets\n")
    
    # Split and train
    print("4. Training model...")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced for faster demo
        max_depth=8,      # Reduced to prevent overfitting
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("5. Results:")
    print(f"\n   Accuracy: {accuracy:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'High']))
    
    # Feature importance
    print("\n   Top 5 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.head(5).iterrows():
        print(f"     • {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Demo completed successfully!")
    print("=" * 70)
    print("\nTo train the full model, run: python model.py")

if __name__ == "__main__":
    try:
        demo()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

