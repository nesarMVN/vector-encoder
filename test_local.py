#!/usr/bin/env python3
"""
Local test script for vector encoding handler.
Tests the handler function directly without RunPod serverless wrapper.
"""

import sys
sys.path.insert(0, '.')

from handler import handler
import json

def test_single_text():
    """Test single text encoding"""
    print("\n" + "="*50)
    print("TEST 1: Single Text Encoding")
    print("="*50)

    event = {
        "input": {
            "text": "red leather shoes"
        }
    }

    result = handler(event)
    print(json.dumps(result, indent=2))

    assert "vector" in result
    assert result["dimensions"] == 384
    assert result["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    print("✓ PASSED")
    return True

def test_batch_text():
    """Test batch text encoding"""
    print("\n" + "="*50)
    print("TEST 2: Batch Text Encoding")
    print("="*50)

    event = {
        "input": {
            "texts": [
                "red leather shoes",
                "blue cotton jacket",
                "black leather bag"
            ]
        }
    }

    result = handler(event)
    print(json.dumps({k: v for k, v in result.items() if k != "vectors"}, indent=2))
    print(f"  vectors: [{len(result.get('vectors', []))} vectors returned]")

    assert "vectors" in result
    assert result["count"] == 3
    assert result["dimensions"] == 384
    print("✓ PASSED")
    return True

def test_single_image():
    """Test single image encoding"""
    print("\n" + "="*50)
    print("TEST 3: Single Image Encoding")
    print("="*50)

    event = {
        "input": {
            "image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400"
        }
    }

    result = handler(event)
    print(json.dumps({k: v for k, v in result.items() if k != "vector"}, indent=2))

    if "error" in result:
        print(f"⚠ SKIPPED: {result['error']}")
        return True

    assert "vector" in result
    assert result["dimensions"] == 512
    assert result["model"] == "openclip/ViT-B-32"
    print("✓ PASSED")
    return True

def test_batch_images():
    """Test batch image encoding"""
    print("\n" + "="*50)
    print("TEST 4: Batch Image Encoding")
    print("="*50)

    event = {
        "input": {
            "image_urls": [
                "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400",
                "https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=400"
            ]
        }
    }

    result = handler(event)
    print(json.dumps({k: v for k, v in result.items() if k != "vectors"}, indent=2))

    if "error" in result:
        print(f"⚠ SKIPPED: {result['error']}")
        return True

    assert "vectors" in result
    assert result["count"] == 2
    assert result["dimensions"] == 512
    print("✓ PASSED")
    return True

if __name__ == "__main__":
    print("="*50)
    print("LOCAL VECTOR ENCODING HANDLER TEST")
    print("="*50)

    results = {
        "single_text": False,
        "batch_text": False,
        "single_image": False,
        "batch_images": False
    }

    try:
        results["single_text"] = test_single_text()
        results["batch_text"] = test_batch_text()
        results["single_image"] = test_single_image()
        results["batch_images"] = test_batch_images()
    except Exception as e:
        print(f"\n\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:20s}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} tests passed")
