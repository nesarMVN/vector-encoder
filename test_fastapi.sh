#!/bin/bash
# Test FastAPI Load Balancer endpoints

ENDPOINT_URL=$1

if [ -z "$ENDPOINT_URL" ]; then
    echo "Usage: ./test_fastapi.sh <endpoint_url>"
    echo "Example: ./test_fastapi.sh https://su6k3l03fjqhjy-8000.proxy.runpod.net"
    exit 1
fi

echo "Testing Vector Encoder FastAPI Endpoints"
echo "========================================"

# Health check
echo -e "\n1. Health Check (GET /)"
curl -s "$ENDPOINT_URL/" | python3 -m json.tool

# Single text encoding
echo -e "\n2. Single Text Encoding (POST /encode/text)"
curl -s "$ENDPOINT_URL/encode/text" \
  -H "Content-Type: application/json" \
  -d '{"text":"red leather shoes"}' | python3 -m json.tool | head -20

# Single image encoding
echo -e "\n3. Single Image Encoding (POST /encode/image)"
curl -s "$ENDPOINT_URL/encode/image" \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400"}' | python3 -m json.tool | head -20

# Batch text encoding
echo -e "\n4. Batch Text Encoding (POST /encode/batch/text)"
curl -s "$ENDPOINT_URL/encode/batch/text" \
  -H "Content-Type: application/json" \
  -d '{"texts":["red leather shoes","blue cotton jacket","black leather bag"]}' | python3 -m json.tool | head -20

# Batch image encoding
echo -e "\n5. Batch Image Encoding (POST /encode/batch/image)"
curl -s "$ENDPOINT_URL/encode/batch/image" \
  -H "Content-Type: application/json" \
  -d '{"image_urls":["https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400","https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=400"]}' | python3 -m json.tool | head -20

echo -e "\n========================================"
echo "All tests completed"
