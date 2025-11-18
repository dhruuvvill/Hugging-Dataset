# Environment Variables Setup

## Quick Start

### Option 1: Using Environment Variables (Recommended)

**Create a `.env` file in the `server/` directory:**

```bash
cd server
touch .env
```

**Add to `.env`:**
```bash
HF_TOKEN=hf_your_token_here
```

**Then start the server:**
```bash
python3 app.py
```

### Option 2: Using export (Linux/Mac)

```bash
export HF_TOKEN=hf_your_token_here
cd server
python3 app.py
```

### Option 3: Using set (Windows)

```cmd
set HF_TOKEN=hf_your_token_here
cd server
python3 app.py
```

### Option 4: Inline (Temporary)

```bash
HF_TOKEN=hf_your_token_here python3 server/app.py
```

## Get Your HuggingFace Token

1. Visit: https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "Model Predictor API")
4. Select scope: **Read** (sufficient for fetching model info)
5. Click "Generate token"
6. Copy the token (starts with `hf_`)
7. Add it to your `.env` file

## Token Usage

### Public Models (No Token Needed)
Most models work without authentication:
- `google/gemma-2-2b` ✅
- `meta-llama/Llama-3.1-8B` ✅
- `openai/whisper-large-v3` ✅

### Private/Gated Models (Token Required)
Some models need authentication:
- Your own private models 🔒
- Gated commercial models 🔒
- Research-only models 🔒

When token is set, the API automatically uses it for all requests.

## Verify Token is Loaded

Check the server logs when starting:
```bash
INFO:service:Using HF_TOKEN for authenticated access  # ✅ Token loaded
INFO:service:Accessing HuggingFace without authentication  # ⚠️ No token
```

## Troubleshooting

### Token not being used?
1. Check `.env` file is in `server/` directory
2. Restart the server after adding token
3. Check server logs for confirmation

### Still getting auth errors?
1. Verify token is valid at https://huggingface.co/settings/tokens
2. Make sure token has "Read" permissions
3. Some models require manual acceptance of terms on website first

## Security Notes

⚠️ **Never commit your `.env` file to git!**

Add to `.gitignore`:
```
.env
```

Your token is kept safe and secure.

