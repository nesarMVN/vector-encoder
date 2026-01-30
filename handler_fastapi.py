#!/usr/bin/env python3
"""
FastAPI Handler for Vector Encoding (Load Balancer mode).

Supports:
- Text encoding via Sentence Transformer
- Image encoding via OpenCLIP
- Single and batch requests
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
import open_clip
from PIL import Image
import requests
from io import BytesIO
import time
import uvicorn

app = FastAPI(title="Vector Encoder API")

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
clip_model.eval()
clip_model = clip_model.to(device)
print(f"✓ Loaded in {time.time() - start:.2f}s")

print("\n" + "=" * 70)
print("SERVICE READY")
print("=" * 70 + "\n")


# Request models
class TextRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    image_url: str

class BatchTextRequest(BaseModel):
    texts: List[str]

class BatchImageRequest(BaseModel):
    image_urls: List[str]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ready",
        "device": device,
        "models": {
            "text": "sentence-transformers/all-MiniLM-L6-v2",
            "image": "openclip/ViT-B-32"
        }
    }


@app.post("/encode/text")
async def encode_text(request: TextRequest):
    """Encode single text"""
    try:
        start = time.time()
        embedding = sentence_model.encode(request.text, convert_to_numpy=True)
        latency_ms = (time.time() - start) * 1000

        return {
            "vector": embedding.tolist(),
            "dimensions": len(embedding),
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "type": "text",
            "latency_ms": round(latency_ms, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode/image")
async def encode_image(request: ImageRequest):
    """Encode single image"""
    try:
        start = time.time()

        # Download image
        response = requests.get(request.image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')

        # Preprocess and encode
        img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
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
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode/batch/text")
async def encode_batch_text(request: BatchTextRequest):
    """Encode multiple texts"""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="texts list cannot be empty")

        start = time.time()
        embeddings = sentence_model.encode(request.texts, convert_to_numpy=True)
        latency_ms = (time.time() - start) * 1000

        return {
            "vectors": embeddings.tolist(),
            "count": len(request.texts),
            "dimensions": embeddings.shape[1],
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "type": "text",
            "latency_ms": round(latency_ms, 2),
            "avg_latency_per_item_ms": round(latency_ms / len(request.texts), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode/batch/image")
async def encode_batch_images(request: BatchImageRequest):
    """Encode multiple images"""
    try:
        if not request.image_urls:
            raise HTTPException(status_code=400, detail="image_urls list cannot be empty")

        start = time.time()

        # Download and preprocess all images
        images = []
        for url in request.image_urls:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
            images.append(clip_preprocess(img))

        # Stack and encode
        img_batch = torch.stack(images).to(device)
        with torch.no_grad():
            embeddings = clip_model.encode_image(img_batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        vectors = embeddings.cpu().numpy()
        latency_ms = (time.time() - start) * 1000

        return {
            "vectors": vectors.tolist(),
            "count": len(request.image_urls),
            "dimensions": vectors.shape[1],
            "model": "openclip/ViT-B-32",
            "type": "image",
            "latency_ms": round(latency_ms, 2),
            "avg_latency_per_item_ms": round(latency_ms / len(request.image_urls), 2)
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
