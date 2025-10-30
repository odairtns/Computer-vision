#!/usr/bin/env python3
"""
Setup script for Equipment Verification System
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        "data/images",
        "data/labels", 
        "data/checklists",
        "models",
        "temp_annotated",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies"""
    commands = [
        ("pip install -r requirements.txt", "Installing main dependencies"),
        ("pip install -r requirements-training.txt", "Installing training dependencies")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True


def download_pretrained_model():
    """Download pretrained YOLOv8 model"""
    print("🔄 Downloading pretrained YOLOv8 model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ Pretrained model downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False


def setup_environment():
    """Setup environment variables"""
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write("PYTHONPATH=.\n")
            f.write("PYTHONUNBUFFERED=1\n")
            f.write("LOG_LEVEL=INFO\n")
        print("📝 Created .env file")
    else:
        print("📝 .env file already exists")


def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    # Test imports
    try:
        import torch
        import cv2
        import numpy as np
        from ultralytics import YOLO
        from fastapi import FastAPI
        print("✅ All required packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test model loading
    try:
        from app.models.detector import EquipmentDetector
        detector = EquipmentDetector()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup Equipment Verification System")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-model", action="store_true", help="Skip model download")
    parser.add_argument("--test-only", action="store_true", help="Only run tests")
    
    args = parser.parse_args()
    
    print("🚀 Setting up Equipment Verification System...")
    
    if args.test_only:
        return test_installation()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            print("❌ Dependency installation failed")
            return False
    
    # Download model
    if not args.skip_model:
        if not download_pretrained_model():
            print("❌ Model download failed")
            return False
    
    # Setup environment
    setup_environment()
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        return False
    
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your training data to the data/ directory")
    print("2. Run: python -m uvicorn app.main:app --reload")
    print("3. Open frontend/index.html in your browser")
    print("4. See TRAINING_GUIDE.md for training instructions")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


