# SmolVLM-500M RunPod Serverless Deployment

Fast, lightweight vision-language model for real-time coaching analysis.

## Quick Deploy (2 Options)

### Option 1: GitHub Deploy (Recommended)

1. Push this folder to a GitHub repo
2. In RunPod Console → Serverless → New Endpoint
3. Click "Import GitHub Repository"
4. Connect your GitHub and select the repo
5. RunPod will auto-detect the Dockerfile and build

### Option 2: Docker Hub Deploy

```bash
# Build locally
docker build -t yourusername/smolvlm-runpod:latest .

# Push to Docker Hub
docker push yourusername/smolvlm-runpod:latest
```

Then in RunPod:
1. Serverless → New Endpoint
2. Select "Custom" template
3. Enter your Docker image: `yourusername/smolvlm-runpod:latest`

## RunPod Settings

| Setting | Recommended Value |
|---------|-------------------|
| GPU | RTX 3080 / RTX 4080 / A4000 (any 10GB+ VRAM) |
| Max Workers | 3-5 |
| Idle Timeout | 5 seconds |
| Flash Boot | Enabled |

## API Usage

### Endpoint URL
```
https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/runsync
```

### Request Format
```json
{
  "input": {
    "prompt": "Analyze the guitar hand position in these frames...",
    "images": ["base64_image1", "base64_image2"],
    "max_tokens": 150,
    "temperature": 0.3
  }
}
```

### Single Image
```json
{
  "input": {
    "prompt": "What do you see?",
    "image": "base64_encoded_jpeg"
  }
}
```

### Response
```json
{
  "output": {
    "status": "success",
    "output": "The person is holding a guitar with..."
  }
}
```

## Test Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run with test server
python handler.py --rp_serve_api

# Test endpoint at http://localhost:8000
```

## Cost Estimate

- **Model**: SmolVLM-500M (~1.2GB VRAM)
- **GPU**: RTX 4080 = ~$0.39/hr on RunPod
- **Latency**: ~50-100ms per image
- **Cost**: ~$0.0001 per request (very cheap!)

## Model Info

- **Model**: [HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
- **Size**: 500M parameters
- **VRAM**: 1.23GB for single image
- **Speed**: 50-100ms inference
