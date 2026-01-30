#!/usr/bin/env python3
"""
Pre-download models during Docker build.
This caches models in the image layer so containers start fast.
"""

import os
import sys

print("=" * 70)
print("DOWNLOADING MODELS FOR CACHING")
print("=" * 70)

# Download Sentence Transformer
print("\n[1/2] Downloading Sentence Transformer model...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Sentence Transformer downloaded successfully")
    print(f"  Model dimensions: {model.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"✗ Failed to download Sentence Transformer: {e}")
    sys.exit(1)

# Download OpenCLIP
print("\n[2/2] Downloading OpenCLIP model...")
try:
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='laion2b_s34b_b79k'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    print("✓ OpenCLIP downloaded successfully")
    print(f"  Model: ViT-B-32")
    print(f"  Pretrained: laion2b_s34b_b79k")
except Exception as e:
    print(f"✗ Failed to download OpenCLIP: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL MODELS CACHED SUCCESSFULLY")
print("=" * 70)
