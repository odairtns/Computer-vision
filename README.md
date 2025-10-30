# 🔍 Equipment Verification System

A **production-ready computer vision solution** that detects people and equipment in images, then verifies if all required equipment is present based on configurable checklists.

## ✨ Key Features

- 🎯 **YOLOv8-based Detection** - State-of-the-art object detection
- 🔧 **Configurable Checklists** - Construction, medical, laboratory safety
- 🌐 **Modern Web Interface** - Drag-and-drop image upload with real-time results
- 🚀 **REST API** - Easy integration with existing systems
- 🐳 **Docker Support** - One-command deployment
- 📚 **Retrainable** - Add new equipment types and use cases
- 🏭 **Production Ready** - Monitoring, logging, scaling support

## 🏗️ Architecture

```
equipment-verification/
├── backend/                    # FastAPI Backend
│   ├── app/                   # Main Application
│   │   ├── main.py           # FastAPI server
│   │   ├── models/           # Detection & verification logic
│   │   ├── schemas/          # API data models
│   │   └── utils/            # Image processing utilities
│   ├── training/             # Training Pipeline
│   │   ├── train.py          # Training script
│   │   ├── dataset.py        # Dataset handling
│   │   └── augmentations.py  # Data augmentation
│   ├── data/                 # Training Data & Checklists
│   ├── scripts/              # Utility scripts
│   └── requirements.txt      # Dependencies
├── frontend/                  # Web Interface
│   ├── index.html           # Main page
│   ├── style.css            # Modern styling
│   └── script.js            # Frontend logic
├── docker-compose.yml        # Docker deployment
├── README.md                 # This file
├── QUICK_START.md           # 30-second setup
├── TRAINING_GUIDE.md        # Custom model training
└── DEPLOYMENT_GUIDE.md      # Production deployment
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone and start
git clone <repository-url>
cd equipment-verification
docker-compose up --build

# Access the application
# Frontend: http://localhost
# API: http://localhost:8000
```

### Option 2: Manual Setup
```bash
# Backend
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
# Open frontend/index.html in your browser
```

## 🎯 How It Works

1. **Upload Image** - User uploads image via web interface
2. **Detect Objects** - YOLOv8 identifies people and equipment
3. **Verify Compliance** - System checks against selected checklist
4. **Display Results** - Shows annotated image with verification status

## 📊 Supported Equipment

### Construction Safety
- Hard hats, safety vests, safety boots, safety glasses
- Gloves, ear protection, respirators

### Medical Equipment  
- Scrubs, stethoscopes, face masks, gloves
- Safety goggles, aprons, shoe covers

### Laboratory Safety
- Lab coats, safety goggles, face masks, gloves
- Safety shoes, aprons, hair nets

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/detect` | POST | Full detection with verification |
| `/verify` | POST | Equipment verification only |
| `/checklists` | GET | List available checklists |
| `/classes` | GET | List detectable classes |

## 📚 Documentation

- **[Quick Start](QUICK_START.md)** - Get running in 30 seconds
- **[Training Guide](TRAINING_GUIDE.md)** - Train custom models
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production deployment
- **[API Docs](http://localhost:8000/docs)** - Interactive API documentation
