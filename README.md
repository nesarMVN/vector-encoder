# RunPod Vector Encoding Microservice

Serverless microservice for encoding text and images into vector embeddings.

## Features

- **Text encoding** using Sentence Transformers (all-MiniLM-L6-v2)
- **Image encoding** using OpenCLIP (ViT-B/32)
- **Single and batch processing**
- **GPU accelerated** (CUDA support)
- **Auto-scaling** via RunPod serverless
- **Pre-cached models** for fast cold starts

## Models

| Type | Model | Dimensions | Use Case |
|------|-------|------------|----------|
| Text | sentence-transformers/all-MiniLM-L6-v2 | 384 | Semantic search, similarity |
| Image | openclip/ViT-B-32 | 512 | Image search, classification |

## Deployment

### Option A: Automated Build (GitHub Actions)

The Docker image is automatically built and pushed to GitHub Container Registry on every push to `main`.

**Push code:**
```bash
git push origin main
```

**Image available at:**
```
ghcr.io/nesarmvn/vector-encoder:latest
```

Build takes ~5-8 minutes on GitHub runners.

### Option B: Manual Build

```bash
docker build -t ghcr.io/nesarmvn/vector-encoder:latest .
docker push ghcr.io/nesarmvn/vector-encoder:latest
```

### Create RunPod Serverless Endpoint

1. Go to https://runpod.io/console/serverless
2. Click "New Endpoint"
3. Configure:
   - **Name**: vector-encoder
   - **Container Image**: `ghcr.io/nesarmvn/vector-encoder:latest`
   - **GPU Type**: T4 (cheapest) or RTX 3090 (better value)
   - **Min Workers**: 0 (pay only when used) or 1 (keep warm, ~$70/month)
   - **Max Workers**: 10 (adjust based on load)
   - **Idle Timeout**: 5 seconds
   - **Regions**: Select all or choose Asia regions for Bangladesh
4. Click "Deploy"

### 3. Get Endpoint URL

After deployment, you'll receive:
- Endpoint ID: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
- API URL: `https://api.runpod.ai/v2/{endpoint_id}/run`

## Usage

### Authentication

All requests require RunPod API key in header:

```bash
Authorization: Bearer YOUR_RUNPOD_API_KEY
```

Get your API key from: https://runpod.io/console/user/settings

### API Endpoints

Base URL: `https://api.runpod.ai/v2/{endpoint_id}/run`

### Single Text Encoding

**Request:**
```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "red leather shoes"
    }
  }'
```

**Response:**
```json
{
  "id": "request-id",
  "status": "COMPLETED",
  "output": {
    "vector": [0.123, -0.456, 0.789, ...],
    "dimensions": 384,
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "type": "text",
    "latency_ms": 12.5
  }
}
```

### Single Image Encoding

**Request:**
```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_url": "https://example.com/product.jpg"
    }
  }'
```

**Response:**
```json
{
  "id": "request-id",
  "status": "COMPLETED",
  "output": {
    "vector": [0.234, -0.567, 0.890, ...],
    "dimensions": 512,
    "model": "openclip/ViT-B-32",
    "type": "image",
    "latency_ms": 25.3
  }
}
```

### Batch Text Encoding

**Request:**
```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "texts": [
        "red leather shoes",
        "blue cotton jacket",
        "black leather bag"
      ]
    }
  }'
```

**Response:**
```json
{
  "id": "request-id",
  "status": "COMPLETED",
  "output": {
    "vectors": [
      [0.1, -0.2, ...],
      [0.3, -0.4, ...],
      [0.5, -0.6, ...]
    ],
    "count": 3,
    "dimensions": 384,
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "type": "text",
    "latency_ms": 18.7,
    "avg_latency_per_item_ms": 6.2
  }
}
```

### Batch Image Encoding

**Request:**
```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_urls": [
        "https://example.com/product1.jpg",
        "https://example.com/product2.jpg"
      ]
    }
  }'
```

**Response:**
```json
{
  "id": "request-id",
  "status": "COMPLETED",
  "output": {
    "vectors": [
      [0.4, -0.5, ...],
      [0.6, -0.7, ...]
    ],
    "count": 2,
    "dimensions": 512,
    "model": "openclip/ViT-B-32",
    "type": "image",
    "latency_ms": 45.2,
    "avg_latency_per_item_ms": 22.6
  }
}
```

## Performance

### Latency

**Cold start (first request):**
- Container boot: 3-10 seconds
- Model loading: Already cached in image
- Total: 3-10 seconds

**Warm requests:**
- Text (single): ~10-15ms
- Image (single): ~20-30ms
- Text (batch of 10): ~30-50ms
- Image (batch of 10): ~80-120ms

### Cost Estimation (RunPod T4 GPU)

**Pricing:** ~$0.00029/second

**For 1 million requests/day:**
- Average processing: 15ms per request
- Total compute: 15,000 seconds/day
- Daily cost: $4.35
- Monthly cost: ~$130

**With batch processing (10 items per batch):**
- Average: 5ms per item
- Total compute: 5,000 seconds/day
- Daily cost: $1.45
- Monthly cost: ~$43

## Error Handling

### Common Errors

**Invalid input:**
```json
{
  "error": "Invalid input format",
  "received_keys": ["invalid_key"],
  "expected_keys": ["text", "image_url", "texts", "image_urls"]
}
```

**Image download failed:**
```json
{
  "error": "Failed to download image: 404 Not Found"
}
```

**Empty batch:**
```json
{
  "error": "'texts' list cannot be empty"
}
```

## Monitoring

Check RunPod console for:
- Request count
- Average latency
- Error rate
- Cost per request
- Active workers

## Optimization Tips

1. **Use batch endpoints** for better throughput
2. **Keep min_workers > 0** if you need low latency (costs more)
3. **Increase idle timeout** for periodic traffic patterns
4. **Choose regions** closest to your users
5. **Monitor cold starts** and adjust worker settings

## Local Testing

Test locally before deploying:

```bash
# Build image
docker build -t vector-encoder .

# Run container
docker run --rm --gpus all -p 8000:8000 vector-encoder

# Test (in another terminal)
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "test"}}'
```

## Troubleshooting

**Container fails to start:**
- Check Docker logs in RunPod console
- Verify GPU availability
- Check model download succeeded

**High latency:**
- Check selected regions (closer is faster)
- Increase min_workers for consistent performance
- Use batch processing for higher throughput

**Out of memory:**
- Reduce batch size
- Use smaller GPU (T4 instead of A10)
- Process images sequentially instead of batching

## Support

- RunPod Documentation: https://docs.runpod.io
- RunPod Discord: https://discord.gg/runpod
- Sentence Transformers: https://www.sbert.net
- OpenCLIP: https://github.com/mlfoundations/open_clip

## License

MIT
