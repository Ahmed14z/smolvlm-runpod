# RunPod Serverless Dockerfile for SmolVLM-500M
# Optimized for fast cold starts

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model to bake into image (faster cold starts)
RUN python -c "from transformers import AutoProcessor, AutoModelForVision2Seq; \
    AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-500M-Instruct'); \
    AutoModelForVision2Seq.from_pretrained('HuggingFaceTB/SmolVLM-500M-Instruct')"

# Copy handler
COPY handler.py .

# Run handler
CMD ["python", "-u", "handler.py"]
