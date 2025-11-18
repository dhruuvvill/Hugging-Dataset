# API Integration Guide
## Frontend → Backend Integration

This guide shows how to integrate the FastAPI backend with any frontend framework.

---

## 1. Backend Setup

### Start the Server
```bash
cd server
python3 app.py
```

Server runs on: `http://localhost:8000`

### API Documentation
Visit: `http://localhost:8000/docs` (Swagger UI)

---

## 2. API Endpoints

### 2.1 POST /api/v1/predict
Predict model popularity by URL

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"huggingface_url": "https://huggingface.co/google/gemma-2-2b"}'
```

**Response:**
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
  "message": "This model is predicted to have high popularity..."
}
```

### 2.2 GET /api/v1/predict/{model_id}
Predict by model ID directly

**Request:**
```bash
curl "http://localhost:8000/api/v1/predict/google/gemma-2-2b"
```

**Response:** Same as above

### 2.3 GET /health
Health check endpoint

**Request:**
```bash
curl "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy"
}
```

---

## 3. Frontend Integration Examples

### 3.1 React/Next.js with Axios

```typescript
// api.js or utils/api.ts
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const predictModel = async (url: string) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/api/v1/predict`, {
      huggingface_url: url
    });
    return response.data;
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
};

export const getModelPrediction = async (modelId: string) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/v1/predict/${modelId}`);
    return response.data;
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
};
```

### 3.2 React Component Example

```typescript
import React, { useState } from 'react';
import { predictModel } from './api';

const ModelPredictor: React.FC = () => {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const prediction = await predictModel(url);
      setResult(prediction);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Enter HuggingFace URL"
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="result">
          <h2>{result.model_id}</h2>
          <div className={`badge ${result.predicted_popularity}`}>
            {result.predicted_popularity.toUpperCase()}
          </div>
          <p>Confidence: {(result.probability * 100).toFixed(1)}%</p>
        </div>
      )}
    </div>
  );
};
```

### 3.3 Vue.js Example

```vue
<template>
  <div>
    <form @submit.prevent="predict">
      <input v-model="url" placeholder="Enter HuggingFace URL" />
      <button :disabled="loading">
        {{ loading ? 'Predicting...' : 'Predict' }}
      </button>
    </form>
    
    <div v-if="error" class="error">{{ error }}</div>
    
    <div v-if="result" class="result">
      <h2>{{ result.model_id }}</h2>
      <div :class="['badge', result.predicted_popularity]">
        {{ result.predicted_popularity.toUpperCase() }}
      </div>
      <p>Confidence: {{ (result.probability * 100).toFixed(1) }}%</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const url = ref('');
const loading = ref(false);
const result = ref(null);
const error = ref(null);

const predict = async () => {
  loading.value = true;
  error.value = null;
  
  try {
    const response = await fetch('http://localhost:8000/api/v1/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ huggingface_url: url.value })
    });
    
    const data = await response.json();
    result.value = data;
  } catch (err) {
    error.value = 'An error occurred';
  } finally {
    loading.value = false;
  }
};
</script>
```

### 3.4 Vanilla JavaScript

```javascript
async function predictModel(url) {
  try {
    const response = await fetch('http://localhost:8000/api/v1/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        huggingface_url: url
      })
    });
    
    if (!response.ok) {
      throw new Error('Prediction failed');
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}

// Usage
const result = await predictModel('https://huggingface.co/google/gemma-2-2b');
console.log(result);
```

---

## 4. Error Handling

### Common Errors

#### 1. Invalid URL Format
```json
{
  "detail": "Invalid HuggingFace URL: https://invalid-url"
}
```

#### 2. Model Not Found
```json
{
  "detail": "Could not fetch model info for invalid/model"
}
```

#### 3. Server Error
```json
{
  "detail": "Prediction failed: [error message]"
}
```

### Error Handling Example

```typescript
try {
  const result = await predictModel(url);
  // Handle success
} catch (error: any) {
  if (error.response?.status === 400) {
    // Bad request - invalid input
    alert('Invalid URL format');
  } else if (error.response?.status === 404) {
    // Model not found
    alert('Model not found on HuggingFace');
  } else if (error.response?.status === 500) {
    // Server error
    alert('Server error. Please try again later.');
  } else {
    // Unknown error
    alert('An unexpected error occurred');
  }
}
```

---

## 5. Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | Model identifier (e.g., "google/gemma-2-2b") |
| `predicted_popularity` | string | "high" or "low" |
| `probability` | float | Confidence score (0.0 - 1.0) |
| `features` | object | Extracted feature values |
| `message` | string | Human-readable prediction summary |

### Features Object

```typescript
{
  num_tags: number;
  has_library_datasets: number;
  has_task_category: number;
  has_language: number;
  has_license: number;
  has_modality: number;
  has_size_category: number;
  num_base_models: number;
  has_transformers: number;
  has_safetensors: number;
  num_arxiv_refs: number;
  days_since_modification: number;
  is_recent: number;
  // ... plus one-hot encoded features
}
```

---

## 6. CORS Configuration

The backend already has CORS enabled for all origins. If you need to restrict it:

```python
# In server/app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 7. Testing

### Test URLs

**High Popularity:**
- `https://huggingface.co/google/gemma-2-2b`
- `https://huggingface.co/meta-llama/Llama-3.1-8B`
- `https://huggingface.co/openai/whisper-large-v3`

**Low Popularity:**
- `https://huggingface.co/JunhaoZhuang/FlashVSR`
- `https://huggingface.co/nvidia/omnivinci`
- `https://huggingface.co/QingyanBai/Ditto_models`

### Manual Testing

```javascript
// Test in browser console
fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    huggingface_url: 'https://huggingface.co/google/gemma-2-2b'
  })
})
.then(r => r.json())
.then(console.log);
```

---

## 8. Next Steps

1. **Choose a frontend framework** (React, Vue, or vanilla JS)
2. **Set up the project** using the README in `frontend/`
3. **Implement the UI** based on wireframes in `PRD.md`
4. **Connect to the API** using examples above
5. **Add error handling** and loading states
6. **Style with Tailwind CSS** or your preferred CSS framework
7. **Test thoroughly** with different model URLs
8. **Deploy** to Vercel, Netlify, or your preferred platform

---

## 9. Resources

- **Backend Code**: `server/` directory
- **API Docs**: http://localhost:8000/docs
- **PRD**: `PRD.md`
- **Model Training**: `model.py`
- **Examples**: `server/examples_low_popularity.md`

---

## 10. Support

For issues or questions:
1. Check API logs: `server/` directory
2. Review error messages in response
3. Test with curl commands first
4. Verify backend is running: `curl http://localhost:8000/health`

