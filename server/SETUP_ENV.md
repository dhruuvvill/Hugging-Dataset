# Environment Setup Guide

## Environment Variables

The backend automatically reads the `HF_TOKEN` environment variable for authentication.

## Setup Options

### 1. No Authentication (Default) ✅

Works for all public models:

```bash
cd server
python3 app.py
```

**Available for:**
- All public HuggingFace models
- Example: `google/gemma-2-2b`, `meta-llama/Llama-3.1-8B`

### 2. With Authentication 🔒

For private or gated models:

**Step 1:** Get your token from https://huggingface.co/settings/tokens

**Step 2:** Set environment variable:

```bash
# Create .env file (optional)
cd server
echo "HF_TOKEN=hf_your_token_here" > .env

# Or export directly
export HF_TOKEN=hf_your_token_here

# Start server
python3 app.py
```

**Now supports:**
- Private models
- Gated models (after accepting terms)
- Your own models

## How It Works

The code reads from environment variables:

```python
# In server/services/hf_service.py
self.token = os.getenv("HF_TOKEN", "").strip()

if self.token:
    self.api = HfApi(token=self.token)
else:
    self.api = HfApi()  # Public access
```

## Quick Test

### Test without token (public model):
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"huggingface_url": "https://huggingface.co/google/gemma-2-2b"}'
```

### Test with token (private model):
```bash
export HF_TOKEN=hf_...
# Same curl command works
```

## Verification

Check server logs:

**With token:**
```
INFO:service:Using HF_TOKEN for authenticated access (token: hf_abc123...)
```

**Without token:**
```
INFO:service:Accessing HuggingFace without authentication (public repos only)
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `401 Authentication Error` | Set `HF_TOKEN` environment variable |
| `Model not found` | Check model ID format (owner/model-name) |
| `Gated model` | Accept terms on HuggingFace website first |
| `.env not loaded` | Restart server after creating `.env` |

## Security

✅ Token is stored in environment variable (secure)  
✅ Never committed to git (`.gitignore` protects it)  
✅ Only used when explicitly set  
✅ Public models work without token  

---

**That's it!** The system automatically handles authentication based on whether `HF_TOKEN` is set.

