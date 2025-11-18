# HuggingFace Authentication Setup

## For Public Models (Default)

**No authentication required!** The API will work with public models automatically.

Simply use the API:
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"huggingface_url": "https://huggingface.co/google/gemma-2-2b"}'
```

## For Private or Gated Models

Some models on HuggingFace require authentication or gated access.

### 1. Get Your HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Copy the token (starts with `hf_`)

### 2. Set the Token

**On Mac/Linux:**
```bash
export HF_TOKEN=hf_your_token_here
cd server
python3 app.py
```

**On Windows:**
```cmd
set HF_TOKEN=hf_your_token_here
cd server
python3 app.py
```

**Or in your code:**
```bash
HF_TOKEN=hf_your_token_here python3 server/app.py
```

### 3. Test with Authenticated Access

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"huggingface_url": "https://huggingface.co/your-private-model"}'
```

## Common Error Messages

### "Model not found"
- Check the model ID format: `owner/model-name`
- Ensure the model exists on HuggingFace

### "Authentication required"
- Model is private or gated
- Set HF_TOKEN environment variable

### "requires acceptance of usage terms"
- Visit the model page on HuggingFace
- Accept the terms/gating
- Then try again

## Public Models (No Auth Needed)

These models work without authentication:
- `google/gemma-2-2b`
- `meta-llama/Llama-3.1-8B`
- `openai/whisper-large-v3`
- `EleutherAI/gpt-j-6b`
- Most popular open-source models

## Private Models (Need Auth)

These may require authentication:
- User's private models
- Gated commercial models
- Research-only models

## Troubleshooting

### Token not working
```bash
# Check if token is set
echo $HF_TOKEN  # Mac/Linux
echo %HF_TOKEN% # Windows

# Restart the server after setting token
```

### Still getting 401 errors
1. Verify token is valid
2. Check token has read permissions
3. Visit model page - may need to accept terms first

