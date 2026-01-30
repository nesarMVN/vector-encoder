#!/usr/bin/env python3
"""
RunPod Serverless Handler for Vector Encoding.

Supports:
- Text encoding via Sentence Transformer
- Image encoding via OpenCLIP
- Single and batch requests
"""

import runpod
import torch
from sentence_transformers import SentenceTransformer
import open_clip
from PIL import Image
import requests
from io import BytesIO
import time

print("=" * 70)
print("INITIALIZING VECTOR ENCODING SERVICE")
print("=" * 70)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Load Sentence Transformer
print("\n[1/2] Loading Sentence Transformer...")
start = time.time()
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_model = sentence_model.to(device)
print(f"✓ Loaded in {time.time() - start:.2f}s")
print(f"  Dimensions: {sentence_model.get_sentence_embedding_dimension()}")

# Load OpenCLIP
print("\n[2/2] Loading OpenCLIP...")
start = time.time()
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='laion2b_s34b_b79k'
)
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model.eval()
clip_model = clip_model.to(device)
print(f"✓ Loaded in {time.time() - start:.2f}s")

print("\n" + "=" * 70)
print("SERVICE READY")
print("=" * 70 + "\n")


def encode_text(text):
    """
    Encode single text using Sentence Transformer.

    Args:
        text: String to encode

    Returns:
        Dict with vector, dimensions, model info
    """
    start = time.time()

    embedding = sentence_model.encode(text, convert_to_numpy=True)

    latency_ms = (time.time() - start) * 1000

    return {
        "vector": embedding.tolist(),
        "dimensions": len(embedding),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "text",
        "latency_ms": round(latency_ms, 2)
    }


def encode_image(image_url):
    """
    Encode single image using CLIP.

    Args:
        image_url: URL of image to encode

    Returns:
        Dict with vector, dimensions, model info
    """
    start = time.time()

    # Download image
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert('RGB')

    # Preprocess
    img_tensor = clip_preprocess(img).unsqueeze(0).to(device)

    # Encode
    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    vector = embedding.cpu().numpy()[0]

    latency_ms = (time.time() - start) * 1000

    return {
        "vector": vector.tolist(),
        "dimensions": len(vector),
        "model": "openclip/ViT-B-32",
        "type": "image",
        "latency_ms": round(latency_ms, 2)
    }


def encode_batch_text(texts):
    """
    Encode multiple texts using Sentence Transformer.

    Args:
        texts: List of strings to encode

    Returns:
        Dict with vectors, count, dimensions, model info
    """
    start = time.time()

    embeddings = sentence_model.encode(texts, convert_to_numpy=True)

    latency_ms = (time.time() - start) * 1000

    return {
        "vectors": embeddings.tolist(),
        "count": len(texts),
        "dimensions": embeddings.shape[1],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "text",
        "latency_ms": round(latency_ms, 2),
        "avg_latency_per_item_ms": round(latency_ms / len(texts), 2)
    }


def encode_batch_images(image_urls):
    """
    Encode multiple images using CLIP.

    Args:
        image_urls: List of image URLs to encode

    Returns:
        Dict with vectors, count, dimensions, model info
    """
    start = time.time()

    # Download and preprocess all images
    images = []
    for url in image_urls:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        images.append(clip_preprocess(img))

    # Stack into batch
    img_batch = torch.stack(images).to(device)

    # Encode batch
    with torch.no_grad():
        embeddings = clip_model.encode_image(img_batch)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    vectors = embeddings.cpu().numpy()

    latency_ms = (time.time() - start) * 1000

    return {
        "vectors": vectors.tolist(),
        "count": len(image_urls),
        "dimensions": vectors.shape[1],
        "model": "openclip/ViT-B-32",
        "type": "image",
        "latency_ms": round(latency_ms, 2),
        "avg_latency_per_item_ms": round(latency_ms / len(image_urls), 2)
    }


def handler(event):
    """
    RunPod serverless handler function.

    Expected input formats:
    - Single text: {"input": {"text": "some text"}}
    - Single image: {"input": {"image_url": "https://..."}}
    - Batch text: {"input": {"texts": ["text1", "text2", ...]}}
    - Batch images: {"input": {"image_urls": ["url1", "url2", ...]}}

    Returns:
        Dict with vectors and metadata
    """

    try:
        input_data = event.get("input", {})

        # Validate input
        if not input_data:
            return {
                "error": "No input provided",
                "usage": {
                    "single_text": {"text": "your text here"},
                    "single_image": {"image_url": "https://example.com/image.jpg"},
                    "batch_text": {"texts": ["text1", "text2"]},
                    "batch_images": {"image_urls": ["url1", "url2"]}
                }
            }

        # Route to appropriate encoder
        if "text" in input_data:
            return encode_text(input_data["text"])

        elif "image_url" in input_data:
            return encode_image(input_data["image_url"])

        elif "texts" in input_data:
            if not isinstance(input_data["texts"], list):
                return {"error": "'texts' must be a list"}
            if len(input_data["texts"]) == 0:
                return {"error": "'texts' list cannot be empty"}
            return encode_batch_text(input_data["texts"])

        elif "image_urls" in input_data:
            if not isinstance(input_data["image_urls"], list):
                return {"error": "'image_urls' must be a list"}
            if len(input_data["image_urls"]) == 0:
                return {"error": "'image_urls' list cannot be empty"}
            return encode_batch_images(input_data["image_urls"])

        else:
            return {
                "error": "Invalid input format",
                "received_keys": list(input_data.keys()),
                "expected_keys": ["text", "image_url", "texts", "image_urls"]
            }

    except requests.exceptions.RequestException as e:
        return {
            "error": f"Failed to download image: {str(e)}"
        }

    except Exception as e:
        return {
            "error": f"Processing failed: {str(e)}",
            "error_type": type(e).__name__
        }


# Start RunPod serverless handler
if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
