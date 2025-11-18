# HuggingFace Model Popularity Predictor API

FastAPI backend for predicting HuggingFace model popularity using trained ML models.

## Project Structure

```
server/
├── app.py                 # FastAPI application
├── models/                # Data models (Pydantic schemas)
│   ├── __init__.py
│   └── schemas.py
├── controller/            # Request handlers
│   ├── __init__.py
│   └── predict_controller.py
├── services/              # Business logic
│   ├── __init__.py
│   ├── model_service.py  # Model loading and prediction
│   └── hf_service.py      # HuggingFace API interaction
├── utils/                  # Utility functions
│   └── __init__.py
└── requirements.txt
```

## Setup

1. **Install dependencies:**
```bash
cd server
pip install -r requirements.txt
```

2. **Train the model first** (from parent directory):
```bash
cd ..
python3 model.py
```

This creates `models_popularity/popularity_classification_downloads.joblib`

3. **Optional: Set HuggingFace token:**
```bash
export HF_TOKEN=hf_your_token_here
```

## Running the Server

```bash
cd server
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Predict by URL
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "huggingface_url": "https://huggingface.co/google/gemma-2-2b"
  }'
```

### 3. Predict by Model ID
```bash
curl http://localhost:8000/api/v1/predict/google/gemma-2-2b
```

## Response Format

```json
{
  "model_id": "google/gemma-2-2b",
  "predicted_popularity": "high",
  "probability": 0.87,
  "features": {
    "num_tags": 15,
    "has_transformers": 1,
    "days_since_modification": 30,
    ...
  },
  "message": "This model is predicted to have high popularity based on its metadata."
}
```

## Example Usage

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"huggingface_url": "https://huggingface.co/google/gemma-2-2b"}
)

result = response.json()
print(f"Predicted: {result['predicted_popularity']}")
print(f"Confidence: {result['probability']:.2%}")
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    huggingface_url: 'https://huggingface.co/google/gemma-2-2b'
  })
});

const result = await response.json();
console.log(`Predicted: ${result.predicted_popularity}`);
```

## MVC Architecture

- **Models** (`models/`): Pydantic schemas for request/response validation
- **View** (`controller/`): API routes and request handling
- **Services** (`services/`): Business logic (model prediction, HF API calls)

