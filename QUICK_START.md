# 🚀 Equipment Verification System - Quick Start

Get up and running with the Equipment Verification System in minutes!

## ⚡ 30-Second Setup

```bash
# 1. Clone and navigate
git clone <repository-url>
cd equipment-verification

# 2. Start with Docker (Recommended)
docker-compose up --build

# 3. Open your browser
# Frontend: http://localhost
# API: http://localhost:8000
```

## 🛠️ Manual Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
# Open frontend/index.html in your browser
# Or serve with: python -m http.server 8080
```

## 🧪 Test the System

```bash
# Test API
cd backend
python scripts/test_api.py --create-sample

# Test with your own image
python scripts/test_api.py --image /path/to/your/image.jpg
```

## 📁 Project Structure

```
equipment-verification/
├── backend/                 # FastAPI backend
│   ├── app/                # Main application
│   ├── training/           # Training pipeline
│   ├── data/               # Training data & checklists
│   ├── scripts/            # Utility scripts
│   └── requirements.txt    # Dependencies
├── frontend/               # Web interface
│   ├── index.html         # Main page
│   ├── style.css          # Styling
│   └── script.js          # Frontend logic
├── docker-compose.yml     # Docker setup
└── README.md              # Full documentation
```

## 🎯 What It Does

1. **Detects People** in uploaded images
2. **Identifies Equipment** (hard hats, safety vests, etc.)
3. **Verifies Compliance** against configurable checklists
4. **Shows Results** with annotated images and verification status

## 🔧 Key Features

- ✅ **YOLOv8-based detection** - State-of-the-art object detection
- ✅ **Configurable checklists** - Construction, medical, laboratory
- ✅ **Web interface** - Easy image upload and results display
- ✅ **REST API** - Integrate with other systems
- ✅ **Docker support** - Easy deployment
- ✅ **Retrainable** - Add new equipment types
- ✅ **Production ready** - Monitoring, logging, scaling

## 📊 Default Equipment Classes

- **Safety Equipment**: hard_hat, safety_vest, safety_boots, safety_glasses
- **Medical Equipment**: scrubs, stethoscope, face_mask, gloves
- **Laboratory Equipment**: lab_coat, safety_goggles, apron, hair_net
- **And more...**

## 🚀 Next Steps

1. **Try it out** - Upload some images and see the results
2. **Customize checklists** - Edit `backend/data/checklists/`
3. **Add your data** - Train on your specific equipment
4. **Deploy** - Use Docker or cloud services
5. **Extend** - Add new equipment types and features

## 📚 Documentation

- **[Full README](README.md)** - Complete project overview
- **[Training Guide](TRAINING_GUIDE.md)** - Train custom models
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production deployment
- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs

## 🆘 Need Help?

- Check the logs: `docker-compose logs -f`
- Test the API: `python backend/scripts/test_api.py`
- See troubleshooting in [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 🎉 You're Ready!

The system is now running and ready to verify equipment compliance. Upload an image and see the magic happen!


