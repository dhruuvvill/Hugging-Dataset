# Quick Start Guide

## 1. Train the Model First

From the HuggingFace directory:

```bash
python3 model.py
```

This creates: `models_popularity/popularity_classification_downloads.joblib`

## 2. Install Dependencies

```bash
cd server
pip install -r requirements.txt
```

## 3. Start the Server

```bash
python app.py
```

Or use the run script:
```bash
./run.sh
```

## 4. Test the API

### Using curl:

**Health check:**
```bash
curl http://localhost:8000/health
```

**Predict by URL:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "huggingface_url": "https://huggingface.co/google/gemma-2-2b"
  }'
```

**Predict by model ID:**
```bash
curl http://localhost:8000/api/v1/predict/google/gemma-2-2b
```

## 5. Example Response

```json
{
  "model_id": "google/gemma-2-2b",
  "predicted_popularity": "high",
  "probability": 0.87,
  "features": {
    "num_tags": 15,
    "has_transformers": 1,
    "has_safetensors": 1,
    "num_arxiv_refs": 2,
    "days_since_modification": 30,
    "is_recent": 1
  },
  "message": "This model is predicted to have high popularity based on its metadata."
}
```

## Project Structure

```
HuggingFace/
├── model.py                    # Training script
├── models_popularity/          # Trained models (after training)
│   └── popularity_classification_downloads.joblib
├── server/                     # FastAPI backend
│   ├── app.py                  # Main application
│   ├── models/                 # Data models (Pydantic)
│   │   └── schemas.py
│   ├── controller/             # API controllers
│   │   └── predict_controller.py
│   ├── services/               # Business logic
│   │   ├── model_service.py    # Model loading
│   │   └── hf_service.py       # HF API calls
│   └── requirements.txt
└── hf_features/
    └── index_models.csv        # Training data
```

## API Documentation

Visit: http://localhost:8000/docs (Swagger UI)

## Troubleshooting

### Model not found error:
```bash
# Make sure you trained the model first
python3 model.py
```

### Module not found:
```bash
# Install dependencies
pip install -r requirements.txt
```

### Can't connect to HuggingFace:
```bash
# Set your token (optional)
export HF_TOKEN=hf_your_token_here
```

