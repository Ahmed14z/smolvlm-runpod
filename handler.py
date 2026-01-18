"""
RunPod Serverless Handler for SmolVLM2-2.2B-Instruct
Optimized for real-time coaching visual analysis

SmolVLM2 is significantly more capable than SmolVLM-500M for:
- Screen content understanding
- UI element detection
- Complex scene analysis
"""

import os
import base64
import io
import logging
import torch
from PIL import Image

import runpod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model state (loaded once at container startup)
MODEL = None
PROCESSOR = None
DEVICE = None


def load_model():
    """Load SmolVLM2 model at container startup."""
    global MODEL, PROCESSOR, DEVICE

    # SmolVLM2 uses AutoModelForImageTextToText (not AutoModelForVision2Seq)
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = os.getenv("MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model: {model_id}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load processor
    PROCESSOR = AutoProcessor.from_pretrained(model_id)

    # Load model with optimizations
    # SmolVLM2-2.2B needs ~5GB VRAM in bfloat16
    MODEL = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)

    MODEL.eval()
    logger.info("Model loaded successfully!")


def decode_base64_image(image_data: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    # Remove data URL prefix if present
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def handler(job):
    """
    Handle inference requests.

    Input format:
    {
        "prompt": "Analyze this screen activity...",
        "images": ["base64_image1", "base64_image2"],  # or single "image": "base64"
        "max_tokens": 150,
        "temperature": 0.3
    }
    """
    job_input = job.get("input", {})

    try:
        # Get prompt
        prompt = job_input.get("prompt", "Describe what you see in this image.")

        # Get images (support both single and multiple)
        images_b64 = job_input.get("images", [])
        if not images_b64:
            single_image = job_input.get("image")
            if single_image:
                images_b64 = [single_image]

        if not images_b64:
            return {"error": "No images provided. Use 'image' or 'images' field."}

        # Log request info
        logger.info(f"Processing {len(images_b64)} image(s), prompt length: {len(prompt)}")

        # Decode images
        images = [decode_base64_image(img) for img in images_b64]

        # Log image sizes
        for i, img in enumerate(images):
            logger.debug(f"Image {i}: {img.size}")

        # Build message format for SmolVLM2
        # SmolVLM2 uses the same chat template format
        content = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # Process input
        text_prompt = PROCESSOR.apply_chat_template(messages, add_generation_prompt=True)
        inputs = PROCESSOR(text=text_prompt, images=images, return_tensors="pt").to(DEVICE)

        # Generate
        max_tokens = job_input.get("max_tokens", 150)
        temperature = job_input.get("temperature", 0.3)

        with torch.no_grad():
            outputs = MODEL.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        # Decode response - only decode the generated tokens (not the input)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = PROCESSOR.decode(generated_tokens, skip_special_tokens=True)

        # Clean up response
        response = response.strip()

        logger.info(f"Generated response: {response[:100]}...")

        return {
            "status": "success",
            "output": response,
        }

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
        }


# Load model at startup
load_model()

# Start serverless handler
runpod.serverless.start({"handler": handler})
