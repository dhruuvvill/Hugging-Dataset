# HuggingFace Datasets Support

The API now supports both **Models** and **Datasets**!

## ✅ What's Supported

### Models
- Example: `https://huggingface.co/google/gemma-2-2b`
- Example: `https://huggingface.co/meta-llama/Llama-3.1-8B`

### Datasets  
- Example: `https://huggingface.co/datasets/EleutherAI/hendrycks_math`
- Example: `https://huggingface.co/datasets/sentence-transformers/all-MiniLM-L6-v2`

## 🔧 How It Works

The API automatically detects whether you're providing:
1. A **model** URL
2. A **dataset** URL

Then it:
1. Extracts the entity ID
2. Fetches info using the appropriate API (`model_info` or `dataset_info`)
3. Extracts features (same for both)
4. Makes a popularity prediction

## 📝 Examples

### Using the API

**Predict a model:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"huggingface_url": "https://huggingface.co/google/gemma-2-2b"}'
```

**Predict a dataset:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"huggingface_url": "https://huggingface.co/datasets/EleutherAI/hendrycks_math"}'
```

## 🎯 Test URLs

### Datasets to Try

**High Popularity:**
- `datasets/sentence-transformers/all-MiniLM-L6-v2`
- `datasets/glue`
- `datasets/squad`

**Low Popularity:**
- `datasets/EleutherAI/hendrycks_math`
- `datasets/my-repo/my-dataset`

## 🔄 Changes Made

1. **Updated `hf_service.py`**:
   - `extract_model_id()` now returns `(is_dataset, entity_id)` tuple
   - `get_model_info()` accepts `is_dataset` parameter
   - Uses `dataset_info()` for datasets and `model_info()` for models

2. **Updated `predict_controller.py`**:
   - Handles both models and datasets
   - Provides appropriate messages for each type
   - Better error handling

## ⚠️ Important Notes

- **Same features extracted** - tags, downloads, likes, lastModified
- **Same ML model used** - trained on 604K+ models
- **Prediction applies to both** - popularity is popularity regardless of type
- **No authentication needed** - works for public repos

## 🚀 Try It Now!

The dataset URL `https://huggingface.co/datasets/EleutherAI/hendrycks_math` is **now valid** and will work!

Test it:
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"huggingface_url": "https://huggingface.co/datasets/EleutherAI/hendrycks_math"}'
```

Expected response:
```json
{
  "model_id": "EleutherAI/hendrycks_math",
  "predicted_popularity": "low",
  "probability": 0.XX,
  "features": {...},
  "message": "This dataset is predicted to have low popularity based on its metadata."
}
```

---

**Updated**: October 2025  
**Status**: ✅ Datasets fully supported!

