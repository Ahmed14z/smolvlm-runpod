# RunPod Serverless Dockerfile for SmolVLM-500M
# Optimized for fast cold starts

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install flash-attn from pre-built wheel (MUCH faster than building from source)
# Wheel for: CUDA 12.x, PyTorch 2.2, Python 3.10
# Find wheels at: https://github.com/Dao-AILab/flash-attention/releases
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install other dependencies
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
