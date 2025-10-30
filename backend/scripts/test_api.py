#!/usr/bin/env python3
"""
Test script for Equipment Verification API
"""
import requests
import json
import time
import argparse
from pathlib import Path


def test_health_endpoint(base_url):
    """Test health check endpoint"""
    print("🔄 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_checklists_endpoint(base_url):
    """Test checklists endpoint"""
    print("🔄 Testing checklists endpoint...")
    try:
        response = requests.get(f"{base_url}/checklists")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Checklists retrieved: {len(data)} available")
            for name, checklist in data.items():
                print(f"  - {name}: {checklist['name']}")
            return True
        else:
            print(f"❌ Checklists check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Checklists check error: {e}")
        return False


def test_detection_endpoint(base_url, image_path):
    """Test detection endpoint with sample image"""
    print(f"🔄 Testing detection endpoint with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'checklist': 'construction',
                'confidence_threshold': 0.5
            }
            
            response = requests.post(f"{base_url}/detect", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Detection successful:")
                print(f"  - Detections: {len(result['detections'])}")
                print(f"  - Verification: {result['verification']['status']}")
                print(f"  - Processing time: {result['processing_time']:.2f}s")
                
                # Print detected objects
                for detection in result['detections']:
                    print(f"    - {detection['class_name']}: {detection['bounding_box']['confidence']:.2f}")
                
                return True
            else:
                print(f"❌ Detection failed: {response.status_code}")
                print(f"  Error: {response.text}")
                return False
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return False


def test_verification_endpoint(base_url, image_path):
    """Test verification-only endpoint"""
    print(f"🔄 Testing verification endpoint with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'checklist': 'medical',
                'confidence_threshold': 0.3
            }
            
            response = requests.post(f"{base_url}/verify", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Verification successful:")
                print(f"  - Status: {result['status']}")
                print(f"  - Present: {result['present_equipment']}")
                print(f"  - Missing: {result['missing_equipment']}")
                print(f"  - Confidence: {result['confidence']:.2f}")
                return True
            else:
                print(f"❌ Verification failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False


def create_sample_image():
    """Create a sample test image"""
    try:
        import cv2
        import numpy as np
        
        # Create a simple test image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Draw a simple rectangle (simulating a person)
        cv2.rectangle(img, (200, 100), (400, 400), (0, 100, 200), -1)
        
        # Add some text
        cv2.putText(img, "Test Image", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save image
        sample_path = "sample_test_image.jpg"
        cv2.imwrite(sample_path, img)
        print(f"📷 Created sample image: {sample_path}")
        return sample_path
    except ImportError:
        print("❌ OpenCV not available, cannot create sample image")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test Equipment Verification API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--create-sample", action="store_true", help="Create sample test image")
    
    args = parser.parse_args()
    
    print("🧪 Testing Equipment Verification API...")
    print(f"📍 API URL: {args.url}")
    
    # Create sample image if requested
    if args.create_sample:
        sample_path = create_sample_image()
        if sample_path:
            args.image = sample_path
    
    # Test endpoints
    tests = [
        ("Health Check", lambda: test_health_endpoint(args.url)),
        ("Checklists", lambda: test_checklists_endpoint(args.url)),
    ]
    
    if args.image:
        tests.extend([
            ("Detection", lambda: test_detection_endpoint(args.url, args.image)),
            ("Verification", lambda: test_verification_endpoint(args.url, args.image)),
        ])
    else:
        print("⚠️  No test image provided, skipping detection tests")
        print("   Use --image <path> or --create-sample to test detection")
    
    # Run tests
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)


