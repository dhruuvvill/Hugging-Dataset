# Examples of Low Popularity HuggingFace Models

These are models with very few downloads (0-100), indicating low popularity:

## Examples:

### 1. **Phr00t/Qwen-Image-Edit-Rapid-AIO**
- **URL**: https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO
- **Downloads**: 0
- **Likes**: 405
- **Tags**: Image editing, ComfyUI model

### 2. **JunhaoZhuang/FlashVSR**
- **URL**: https://huggingface.co/JunhaoZhuang/FlashVSR
- **Downloads**: 0
- **Likes**: 93
- **Tags**: Video super-resolution, recent research

### 3. **facebook/MobileLLM-Pro**
- **URL**: https://huggingface.co/facebook/MobileLLM-Pro
- **Downloads**: 179
- **Likes**: 119
- **Tags**: Mobile LLM, Meta/Facebook

### 4. **nvidia/omnivinci**
- **URL**: https://huggingface.co/nvidia/omnivinci
- **Downloads**: 95
- **Likes**: 61
- **Tags**: NVIDIA, vision model

### 5. **QingyanBai/Ditto_models**
- **URL**: https://huggingface.co/QingyanBai/Ditto_models
- **Downloads**: 0
- **Likes**: 41
- **Tags**: Video-to-video model

### 6. **xiabs/DreamOmni2**
- **URL**: https://huggingface.co/xiabs/DreamOmni2
- **Downloads**: 0
- **Likes**: 66
- **Tags**: Safetensors format

### 7. **meituan-longcat/LongCat-Audio-Codec**
- **URL**: https://huggingface.co/meituan-longcat/LongCat-Audio-Codec
- **Downloads**: 0
- **Likes**: 28
- **Tags**: Audio codec, research paper

### 8. **lrzjason/QwenImage-Rebalance**
- **URL**: https://huggingface.co/lrzjason/QwenImage-Rebalance
- **Downloads**: 0
- **Likes**: 22
- **Tags**: ComfyUI, image generation

### 9. **WithAnyone/WithAnyone**
- **URL**: https://huggingface.co/WithAnyone/WithAnyone
- **Downloads**: 0
- **Likes**: 20
- **Tags**: Text-to-image, novel research

### 10. **allenai/olmOCR-2-7B-1025**
- **URL**: https://huggingface.co/allenai/olmOCR-2-7B-1025
- **Downloads**: 242
- **Likes**: 18
- **Tags**: OCR model, AllenAI

## Common Characteristics of Low Popularity Models:

1. **No downloads** (0) - hasn't been used
2. **Specialized/niche use cases** - very specific domains
3. **Recent uploads** - not enough time to gain traction
4. **Research prototypes** - experimental models
5. **Alternative formats** - specialized model formats

## Test These URLs:

```bash
# Example 1
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"huggingface_url": "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO"}'

# Example 2
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"huggingface_url": "https://huggingface.co/JunhaoZhuang/FlashVSR"}'

# Example 3
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"huggingface_url": "https://huggingface.co/nvidia/omnivinci"}'
```

These should typically return **"predicted_popularity": "low"** with lower probability scores.

