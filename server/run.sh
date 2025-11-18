#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv/bin" ]; then
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "../models_popularity/popularity_classification_downloads.joblib" ]; then
    echo "⚠️  Model not found. Training model..."
    cd ..
    python3 model.py
    cd server
fi

# Start the server
echo "🚀 Starting FastAPI server..."
python app.py

