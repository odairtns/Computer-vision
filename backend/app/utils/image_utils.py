"""
Image processing utilities for the equipment verification system
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
import io
import base64


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image as numpy array
        target_size: Target size (width, height)
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate padding offsets
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[dict],
    class_names: List[str],
    confidence_threshold: float = 0.5
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        class_names: List of class names
        confidence_threshold: Minimum confidence to display
    
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    # Define colors for different classes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    for i, detection in enumerate(detections):
        if detection['confidence'] < confidence_threshold:
            continue
            
        # Get bounding box coordinates
        x1, y1, x2, y2 = detection['bbox']
        class_id = detection['class_id']
        confidence = detection['confidence']
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Prepare label
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        label = f"{class_name}: {confidence:.2f}"
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(
            annotated,
            (int(x1), int(y1) - text_height - baseline),
            (int(x1) + text_width, int(y1)),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label,
            (int(x1), int(y1) - baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
    
    return annotated


def image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """
    Convert numpy image to base64 string
    
    Args:
        image: Input image as numpy array
        format: Image format (JPEG, PNG)
    
    Returns:
        Base64 encoded string
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/{format.lower()};base64,{img_str}"


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for model inference
    
    Args:
        image: Input image
    
    Returns:
        Preprocessed image
    """
    # Convert to RGB if BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image


def postprocess_detections(
    raw_detections: np.ndarray,
    class_names: List[str],
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.4
) -> List[dict]:
    """
    Post-process raw model detections
    
    Args:
        raw_detections: Raw detection output from model
        class_names: List of class names
        confidence_threshold: Minimum confidence threshold
        nms_threshold: Non-maximum suppression threshold
    
    Returns:
        List of processed detections
    """
    detections = []
    
    for detection in raw_detections:
        # Extract detection components
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        
        if confidence < confidence_threshold:
            continue
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': float(confidence),
            'class_id': int(class_id),
            'class_name': class_names[int(class_id)] if int(class_id) < len(class_names) else f"Class_{int(class_id)}"
        })
    
    # Apply Non-Maximum Suppression
    if detections:
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        class_ids = np.array([d['class_id'] for d in detections])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            confidence_threshold,
            nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            detections = [detections[i] for i in indices]
    
    return detections

