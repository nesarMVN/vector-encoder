#!/usr/bin/env python3
"""
Test script for RunPod vector encoding endpoint.

Usage:
    python test_endpoint.py <endpoint_id> <api_key>
"""

import requests
import json
import sys
import time

def test_single_text(endpoint_url, api_key):
    """Test single text encoding"""
    print("\n" + "="*70)
    print("TEST 1: Single Text Encoding")
    print("="*70)

    payload = {
        "input": {
            "text": "red leather shoes"
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print(f"Request: {json.dumps(payload, indent=2)}")
    start = time.time()
    response = requests.post(endpoint_url, json=payload, headers=headers)
    latency = (time.time() - start) * 1000

    print(f"\nStatus: {response.status_code}")
    print(f"Total latency: {latency:.2f}ms")

    if response.status_code == 200:
        result = response.json()
        print(f"\nResponse:")
        print(f"  Status: {result.get('status')}")
        if result.get('status') == 'COMPLETED':
            output = result.get('output', {})
            print(f"  Dimensions: {output.get('dimensions')}")
            print(f"  Model: {output.get('model')}")
            print(f"  Processing latency: {output.get('latency_ms')}ms")
            print(f"  Vector (first 5): {output.get('vector', [])[:5]}")
            return True
    else:
        print(f"Error: {response.text}")
        return False


def test_single_image(endpoint_url, api_key):
    """Test single image encoding"""
    print("\n" + "="*70)
    print("TEST 2: Single Image Encoding")
    print("="*70)

    payload = {
        "input": {
            "image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400"
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print(f"Request: {json.dumps(payload, indent=2)}")
    start = time.time()
    response = requests.post(endpoint_url, json=payload, headers=headers)
    latency = (time.time() - start) * 1000

    print(f"\nStatus: {response.status_code}")
    print(f"Total latency: {latency:.2f}ms")

    if response.status_code == 200:
        result = response.json()
        print(f"\nResponse:")
        print(f"  Status: {result.get('status')}")
        if result.get('status') == 'COMPLETED':
            output = result.get('output', {})
            print(f"  Dimensions: {output.get('dimensions')}")
            print(f"  Model: {output.get('model')}")
            print(f"  Processing latency: {output.get('latency_ms')}ms")
            print(f"  Vector (first 5): {output.get('vector', [])[:5]}")
            return True
    else:
        print(f"Error: {response.text}")
        return False


def test_batch_text(endpoint_url, api_key):
    """Test batch text encoding"""
    print("\n" + "="*70)
    print("TEST 3: Batch Text Encoding")
    print("="*70)

    payload = {
        "input": {
            "texts": [
                "red leather shoes",
                "blue cotton jacket",
                "black leather bag"
            ]
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print(f"Request: {json.dumps(payload, indent=2)}")
    start = time.time()
    response = requests.post(endpoint_url, json=payload, headers=headers)
    latency = (time.time() - start) * 1000

    print(f"\nStatus: {response.status_code}")
    print(f"Total latency: {latency:.2f}ms")

    if response.status_code == 200:
        result = response.json()
        print(f"\nResponse:")
        print(f"  Status: {result.get('status')}")
        if result.get('status') == 'COMPLETED':
            output = result.get('output', {})
            print(f"  Count: {output.get('count')}")
            print(f"  Dimensions: {output.get('dimensions')}")
            print(f"  Model: {output.get('model')}")
            print(f"  Processing latency: {output.get('latency_ms')}ms")
            print(f"  Avg per item: {output.get('avg_latency_per_item_ms')}ms")
            return True
    else:
        print(f"Error: {response.text}")
        return False


def test_batch_images(endpoint_url, api_key):
    """Test batch image encoding"""
    print("\n" + "="*70)
    print("TEST 4: Batch Image Encoding")
    print("="*70)

    payload = {
        "input": {
            "image_urls": [
                "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400",
                "https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=400"
            ]
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print(f"Request: {json.dumps(payload, indent=2)}")
    start = time.time()
    response = requests.post(endpoint_url, json=payload, headers=headers)
    latency = (time.time() - start) * 1000

    print(f"\nStatus: {response.status_code}")
    print(f"Total latency: {latency:.2f}ms")

    if response.status_code == 200:
        result = response.json()
        print(f"\nResponse:")
        print(f"  Status: {result.get('status')}")
        if result.get('status') == 'COMPLETED':
            output = result.get('output', {})
            print(f"  Count: {output.get('count')}")
            print(f"  Dimensions: {output.get('dimensions')}")
            print(f"  Model: {output.get('model')}")
            print(f"  Processing latency: {output.get('latency_ms')}ms")
            print(f"  Avg per item: {output.get('avg_latency_per_item_ms')}ms")
            return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_endpoint.py <endpoint_id> <api_key>")
        print("\nExample:")
        print("  python test_endpoint.py abc123-def456-ghi789 your-api-key-here")
        sys.exit(1)

    endpoint_id = sys.argv[1]
    api_key = sys.argv[2]

    endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"

    print("="*70)
    print("RUNPOD VECTOR ENCODING ENDPOINT TEST")
    print("="*70)
    print(f"\nEndpoint URL: {endpoint_url}")
    print(f"API Key: {api_key[:10]}...{api_key[-10:]}")

    results = {
        "single_text": False,
        "single_image": False,
        "batch_text": False,
        "batch_images": False
    }

    # Run all tests
    try:
        results["single_text"] = test_single_text(endpoint_url, api_key)
        time.sleep(1)

        results["single_image"] = test_single_image(endpoint_url, api_key)
        time.sleep(1)

        results["batch_text"] = test_batch_text(endpoint_url, api_key)
        time.sleep(1)

        results["batch_images"] = test_batch_images(endpoint_url, api_key)

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nTests failed with error: {e}")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:20s}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Endpoint is working correctly.")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
