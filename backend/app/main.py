"""
FastAPI backend service for equipment verification system
"""
import os
import logging
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
from PIL import Image
import io

from app.models.detector import EquipmentDetector
from app.models.verifier import EquipmentVerifier
from app.schemas.models import (
    DetectionResponse, 
    HealthResponse, 
    ErrorResponse,
    ChecklistInfo,
    BoundingBox,
    Detection,
    VerificationResult
)
from app.utils.image_utils import draw_bounding_boxes, image_to_base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Equipment Verification API",
    description="Computer vision system for detecting people and equipment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model instances
detector: Optional[EquipmentDetector] = None
verifier: Optional[EquipmentVerifier] = None

# Equipment classes that can be detected
EQUIPMENT_CLASSES = [
    "hard_hat", "safety_vest", "safety_boots", "safety_glasses",
    "gloves", "ear_protection", "respirator", "scrubs", "stethoscope",
    "face_mask", "lab_coat", "safety_goggles", "apron", "shoe_covers",
    "hair_net", "safety_shoes"
]


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global detector, verifier
    
    try:
        # Initialize detector
        detector = EquipmentDetector(device="auto")
        logger.info("Detector initialized successfully")
        
        # Initialize verifier
        verifier = EquipmentVerifier()
        logger.info("Verifier initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise


def get_detector() -> EquipmentDetector:
    """Dependency to get detector instance"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    return detector


def get_verifier() -> EquipmentVerifier:
    """Dependency to get verifier instance"""
    if verifier is None:
        raise HTTPException(status_code=500, detail="Verifier not initialized")
    return verifier


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if detector and verifier else "unhealthy",
        model_loaded=detector is not None and detector.is_model_loaded(),
        version="1.0.0"
    )


@app.get("/checklists", response_model=dict)
async def get_checklists(verifier_instance: EquipmentVerifier = Depends(get_verifier)):
    """Get available equipment checklists"""
    return verifier_instance.get_available_checklists()


@app.post("/detect", response_model=DetectionResponse)
async def detect_equipment(
    file: UploadFile = File(...),
    checklist: Optional[str] = "construction",
    confidence_threshold: float = 0.5,
    detector_instance: EquipmentDetector = Depends(get_detector),
    verifier_instance: EquipmentVerifier = Depends(get_verifier)
):
    """
    Detect people and equipment in uploaded image
    
    Args:
        file: Uploaded image file
        checklist: Equipment checklist to use
        confidence_threshold: Minimum confidence for detections
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Set confidence threshold
        detector_instance.set_confidence_threshold(confidence_threshold)
        
        # Detect people
        person_detections = detector_instance.detect_people(image_np)
        
        # Detect equipment
        equipment_detections = detector_instance.detect_equipment(image_np, EQUIPMENT_CLASSES)
        
        # Combine all detections
        all_detections = person_detections + equipment_detections
        
        # Convert to API response format
        detections = []
        for det in all_detections:
            bbox = BoundingBox(
                x1=float(det['bbox'][0]),
                y1=float(det['bbox'][1]),
                x2=float(det['bbox'][2]),
                y2=float(det['bbox'][3]),
                confidence=float(det['confidence'])
            )
            
            detection = Detection(
                class_name=det['class_name'],
                bounding_box=bbox,
                equipment_id=None  # Could be added for specific equipment tracking
            )
            detections.append(detection)
        
        # Verify equipment
        verification_result = verifier_instance.verify_equipment(
            all_detections, 
            checklist
        )
        
        # Create verification result
        verification = VerificationResult(
            status=verification_result['status'],
            missing_equipment=verification_result['missing_equipment'],
            present_equipment=verification_result['present_equipment'],
            confidence=verification_result['confidence']
        )
        
        # Create annotated image
        annotated_image = draw_bounding_boxes(
            image_np, 
            all_detections, 
            detector_instance.get_class_names(),
            confidence_threshold
        )
        
        # Save annotated image
        annotated_path = f"temp_annotated_{int(time.time())}.jpg"
        cv2.imwrite(annotated_path, annotated_image)
        
        processing_time = time.time() - start_time
        
        return DetectionResponse(
            detections=detections,
            verification=verification,
            annotated_image_path=annotated_path,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/detect/annotated/{filename}")
async def get_annotated_image(filename: str):
    """Get annotated image file"""
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Annotated image not found")
    
    return FileResponse(filename, media_type="image/jpeg")


@app.post("/verify", response_model=VerificationResult)
async def verify_equipment_only(
    file: UploadFile = File(...),
    checklist: str = "construction",
    confidence_threshold: float = 0.5,
    detector_instance: EquipmentDetector = Depends(get_detector),
    verifier_instance: EquipmentVerifier = Depends(get_verifier)
):
    """
    Verify equipment without returning full detection details
    """
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Detect equipment
        detector_instance.set_confidence_threshold(confidence_threshold)
        equipment_detections = detector_instance.detect_equipment(image_np, EQUIPMENT_CLASSES)
        
        # Verify equipment
        verification_result = verifier_instance.verify_equipment(
            equipment_detections, 
            checklist
        )
        
        return VerificationResult(
            status=verification_result['status'],
            missing_equipment=verification_result['missing_equipment'],
            present_equipment=verification_result['present_equipment'],
            confidence=verification_result['confidence']
        )
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.post("/checklist/{checklist_name}")
async def set_checklist(
    checklist_name: str,
    verifier_instance: EquipmentVerifier = Depends(get_verifier)
):
    """Set active checklist"""
    try:
        verifier_instance.set_checklist(checklist_name)
        return {"message": f"Checklist set to: {checklist_name}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/classes")
async def get_detectable_classes(detector_instance: EquipmentDetector = Depends(get_detector)):
    """Get list of detectable classes"""
    return {
        "all_classes": detector_instance.get_class_names(),
        "equipment_classes": EQUIPMENT_CLASSES
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


