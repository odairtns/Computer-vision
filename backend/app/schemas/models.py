"""
Pydantic models for API request/response schemas
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates and confidence"""
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    confidence: float = Field(..., description="Detection confidence (0-1)")


class Detection(BaseModel):
    """Single object detection"""
    class_name: str = Field(..., description="Detected class name")
    bounding_box: BoundingBox = Field(..., description="Bounding box coordinates")
    equipment_id: Optional[str] = Field(None, description="Equipment ID if available")


class VerificationResult(BaseModel):
    """Equipment verification result"""
    status: str = Field(..., description="PASS or FAIL")
    missing_equipment: List[str] = Field(default_factory=list, description="List of missing equipment")
    present_equipment: List[str] = Field(default_factory=list, description="List of present equipment")
    confidence: float = Field(..., description="Overall verification confidence")


class DetectionResponse(BaseModel):
    """API response for detection endpoint"""
    detections: List[Detection] = Field(..., description="List of detected objects")
    verification: VerificationResult = Field(..., description="Equipment verification result")
    annotated_image_path: Optional[str] = Field(None, description="Path to annotated image")
    processing_time: float = Field(..., description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Application version")


class ChecklistInfo(BaseModel):
    """Equipment checklist information"""
    name: str = Field(..., description="Checklist name")
    description: str = Field(..., description="Checklist description")
    required_equipment: List[str] = Field(..., description="List of required equipment")
    optional_equipment: List[str] = Field(default_factory=list, description="List of optional equipment")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

