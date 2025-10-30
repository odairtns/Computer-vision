"""
YOLOv8-based object detection model for equipment verification
"""
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EquipmentDetector:
    """
    YOLOv8-based detector for people and equipment
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the detector
        
        Args:
            model_path: Path to custom model weights (optional)
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        self.model = None
        self.class_names = []
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # Load model
        self._load_model(model_path)
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load YOLOv8 model"""
        try:
            if model_path and os.path.exists(model_path):
                # Load custom model
                self.model = YOLO(model_path)
                logger.info(f"Loaded custom model from {model_path}")
            else:
                # Load pretrained YOLOv8 model
                self.model = YOLO('yolov8n.pt')  # nano version for speed
                logger.info("Loaded pretrained YOLOv8n model")
            
            # Move model to device
            self.model.to(self.device)
            
            # Get class names
            self.class_names = self.model.names
            logger.info(f"Model loaded on {self.device} with {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, iou=self.nms_threshold)
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i in range(len(boxes)):
                        detection = {
                            'bbox': boxes[i].tolist(),
                            'confidence': float(confidences[i]),
                            'class_id': int(class_ids[i]),
                            'class_name': self.class_names[class_ids[i]]
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    def detect_people(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect only people in the image
        
        Args:
            image: Input image
            
        Returns:
            List of person detections
        """
        all_detections = self.detect(image)
        
        # Filter for person class (class_id 0 in COCO dataset)
        person_detections = [
            det for det in all_detections 
            if det['class_id'] == 0 and det['class_name'] == 'person'
        ]
        
        return person_detections
    
    def detect_equipment(self, image: np.ndarray, equipment_classes: List[str]) -> List[Dict[str, Any]]:
        """
        Detect specific equipment classes using heuristics and available COCO classes
        
        Args:
            image: Input image
            equipment_classes: List of equipment class names to detect
            
        Returns:
            List of equipment detections
        """
        all_detections = self.detect(image)
        
        # First, try to find exact matches in available classes
        equipment_detections = [
            det for det in all_detections 
            if det['class_name'] in equipment_classes
        ]
        
        # If no exact matches found, use heuristics to detect safety equipment
        if not equipment_detections:
            equipment_detections = self._detect_safety_equipment_heuristics(image, all_detections, equipment_classes)
        
        return equipment_detections
    
    def _detect_safety_equipment_heuristics(
        self, 
        image: np.ndarray, 
        detections: List[Dict[str, Any]], 
        equipment_classes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Use heuristics to detect safety equipment based on person detection and image analysis
        
        Args:
            image: Input image
            detections: List of all detections
            equipment_classes: List of equipment classes to detect
            
        Returns:
            List of detected equipment
        """
        equipment_detections = []
        
        # Find person detections
        person_detections = [det for det in detections if det['class_name'] == 'person']
        
        if person_detections:
            # If we have person detections, analyze each person's region
            for person in person_detections:
                person_bbox = person['bbox']
                
                # Extract person region with some padding
                x1, y1, x2, y2 = person_bbox
                padding = 50  # pixels
                h, w = image.shape[:2]
                
                # Ensure coordinates are within image bounds
                x1 = max(0, int(x1 - padding))
                y1 = max(0, int(y1 - padding))
                x2 = min(w, int(x2 + padding))
                y2 = min(h, int(y2 + padding))
                
                person_region = image[y1:y2, x1:x2]
                
                if person_region.size == 0:
                    continue
                
                # Detect safety equipment in person region
                detected_equipment = self._analyze_person_for_safety_equipment(
                    person_region, person_bbox, equipment_classes
                )
                
                equipment_detections.extend(detected_equipment)
        else:
            # If no person detected, analyze the entire image for safety equipment patterns
            # This is a fallback for cases where person detection fails
            equipment_detections = self._analyze_entire_image_for_safety_equipment(
                image, equipment_classes
            )
        
        return equipment_detections
    
    def _analyze_person_for_safety_equipment(
        self, 
        person_region: np.ndarray, 
        person_bbox: List[float], 
        equipment_classes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Analyze person region for safety equipment using color and shape analysis
        
        Args:
            person_region: Image region containing the person
            person_bbox: Bounding box of the person
            equipment_classes: List of equipment classes to detect
            
        Returns:
            List of detected equipment
        """
        equipment_detections = []
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
        
        # Detect hard hat (bright colors in upper region)
        if 'hard_hat' in equipment_classes:
            hard_hat = self._detect_hard_hat(person_region, hsv, person_bbox)
            if hard_hat:
                equipment_detections.append(hard_hat)
        
        # Detect safety vest (bright orange/yellow colors)
        if 'safety_vest' in equipment_classes:
            safety_vest = self._detect_safety_vest(person_region, hsv, person_bbox)
            if safety_vest:
                equipment_detections.append(safety_vest)
        
        # Detect safety glasses (transparent/reflective regions)
        if 'safety_glasses' in equipment_classes:
            safety_glasses = self._detect_safety_glasses(person_region, hsv, person_bbox)
            if safety_glasses:
                equipment_detections.append(safety_glasses)
        
        # Detect safety boots (dark colors in lower region)
        if 'safety_boots' in equipment_classes:
            safety_boots = self._detect_safety_boots(person_region, hsv, person_bbox)
            if safety_boots:
                equipment_detections.append(safety_boots)
        
        return equipment_detections
    
    def _analyze_entire_image_for_safety_equipment(
        self, 
        image: np.ndarray, 
        equipment_classes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Analyze entire image for safety equipment patterns when no person is detected
        
        Args:
            image: Input image
            equipment_classes: List of equipment classes to detect
            
        Returns:
            List of detected equipment
        """
        equipment_detections = []
        h, w = image.shape[:2]
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect hard hat (bright colors in upper region of image)
        if 'hard_hat' in equipment_classes:
            hard_hat = self._detect_hard_hat_global(image, hsv)
            if hard_hat:
                equipment_detections.append(hard_hat)
        
        # Detect safety vest (bright orange/yellow colors in middle region)
        if 'safety_vest' in equipment_classes:
            safety_vest = self._detect_safety_vest_global(image, hsv)
            if safety_vest:
                equipment_detections.append(safety_vest)
        
        # Detect safety glasses (edge patterns in upper region)
        if 'safety_glasses' in equipment_classes:
            safety_glasses = self._detect_safety_glasses_global(image, hsv)
            if safety_glasses:
                equipment_detections.append(safety_glasses)
        
        # Detect safety boots (dark colors in lower region)
        if 'safety_boots' in equipment_classes:
            safety_boots = self._detect_safety_boots_global(image, hsv)
            if safety_boots:
                equipment_detections.append(safety_boots)
        
        return equipment_detections
    
    def _detect_hard_hat_global(self, image: np.ndarray, hsv: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect hard hat in entire image using color analysis in upper region"""
        h, w = image.shape[:2]
        
        # Focus on upper 30% of image (head area)
        upper_region = hsv[:int(h*0.3), :]
        
        # Look for bright colors (white, yellow, orange hard hats)
        white_mask = cv2.inRange(upper_region, np.array([0, 0, 200]), np.array([180, 30, 255]))
        yellow_mask = cv2.inRange(upper_region, np.array([20, 100, 100]), np.array([30, 255, 255]))
        
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        bright_pixels = cv2.countNonZero(combined_mask)
        total_pixels = upper_region.shape[0] * upper_region.shape[1]
        
        if bright_pixels > total_pixels * 0.05:  # Lower threshold for global detection
            # Create bounding box for hard hat (upper region)
            hat_height = int(h * 0.15)
            
            return {
                'bbox': [0, 0, w, hat_height],
                'confidence': min(0.75, bright_pixels / total_pixels * 3),
                'class_id': -1,
                'class_name': 'hard_hat'
            }
        
        return None
    
    def _detect_safety_vest_global(self, image: np.ndarray, hsv: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect safety vest in entire image using bright orange/yellow color analysis"""
        h, w = image.shape[:2]
        
        # Focus on middle 60% of image (torso area)
        middle_start = int(h*0.2)
        middle_end = int(h*0.8)
        middle_region = hsv[middle_start:middle_end, :]
        
        # Look for bright orange/yellow colors
        orange_mask = cv2.inRange(middle_region, np.array([5, 100, 100]), np.array([25, 255, 255]))
        
        orange_pixels = cv2.countNonZero(orange_mask)
        total_pixels = middle_region.shape[0] * middle_region.shape[1]
        
        if orange_pixels > total_pixels * 0.08:  # Lower threshold for global detection
            # Create bounding box for safety vest (middle region)
            vest_height = int(h * 0.4)
            vest_y1 = int(h * 0.2)
            
            return {
                'bbox': [0, vest_y1, w, vest_y1 + vest_height],
                'confidence': min(0.70, orange_pixels / total_pixels * 3),
                'class_id': -2,
                'class_name': 'safety_vest'
            }
        
        return None
    
    def _detect_safety_glasses_global(self, image: np.ndarray, hsv: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect safety glasses in entire image using enhanced edge detection and shape analysis"""
        h, w = image.shape[:2]
        
        # Focus on upper 30% of image (face area) - increased range
        upper_region = image[:int(h*0.3), :]
        gray_upper = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
        
        # Use multiple edge detection methods
        edges1 = cv2.Canny(gray_upper, 30, 100)  # Lower thresholds for more sensitive detection
        edges2 = cv2.Canny(gray_upper, 50, 150)  # Standard thresholds
        
        # Combine edge detections
        combined_edges = cv2.bitwise_or(edges1, edges2)
        
        # Look for horizontal lines (glasses frames)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        horizontal_lines = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Look for circular/oval patterns (lens shapes)
        circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 10))
        circular_patterns = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, circular_kernel)
        
        # Combine horizontal lines and circular patterns
        glasses_mask = cv2.bitwise_or(horizontal_lines, circular_patterns)
        
        # Find contours to detect glasses-like shapes
        contours, _ = cv2.findContours(glasses_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        glasses_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (w * h * 0.001):  # At least 0.1% of image area
                # Check aspect ratio (glasses are typically wider than tall)
                x, y, w_cont, h_cont = cv2.boundingRect(contour)
                aspect_ratio = w_cont / h_cont if h_cont > 0 else 0
                if aspect_ratio > 2.0:  # Glasses are typically much wider than tall
                    glasses_contours.append(contour)
        
        if glasses_contours:
            # Calculate total glasses area
            total_glasses_area = sum(cv2.contourArea(contour) for contour in glasses_contours)
            total_pixels = upper_region.shape[0] * upper_region.shape[1]
            
            if total_glasses_area > total_pixels * 0.002:  # At least 0.2% of upper region
                glasses_height = int(h * 0.1)  # Increased height
                glasses_y1 = int(h * 0.05)
                
                return {
                    'bbox': [0, glasses_y1, w, glasses_y1 + glasses_height],
                    'confidence': min(0.80, total_glasses_area / total_pixels * 20),
                    'class_id': -3,
                    'class_name': 'safety_glasses'
                }
        
        # Fallback: simple line detection with lower threshold
        line_pixels = cv2.countNonZero(horizontal_lines)
        total_pixels = upper_region.shape[0] * upper_region.shape[1]
        
        if line_pixels > total_pixels * 0.005:  # Lower threshold
            glasses_height = int(h * 0.1)
            glasses_y1 = int(h * 0.05)
            
            return {
                'bbox': [0, glasses_y1, w, glasses_y1 + glasses_height],
                'confidence': min(0.70, line_pixels / total_pixels * 20),
                'class_id': -3,
                'class_name': 'safety_glasses'
            }
        
        return None
    
    def _detect_safety_boots_global(self, image: np.ndarray, hsv: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect safety boots in entire image using enhanced color and shape analysis in lower region"""
        h, w = image.shape[:2]
        
        # Focus on lower 25% of image (feet area) - increased range
        lower_region = hsv[int(h*0.75):, :]
        lower_region_bgr = image[int(h*0.75):, :]
        
        # Look for dark colors (black, brown, dark gray boots)
        dark_mask = cv2.inRange(lower_region, np.array([0, 0, 0]), np.array([180, 255, 120]))
        
        # Also look for brown colors (common boot color)
        brown_mask = cv2.inRange(lower_region, np.array([10, 50, 20]), np.array([20, 255, 100]))
        
        # Combine masks
        combined_mask = cv2.bitwise_or(dark_mask, brown_mask)
        
        # Use morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours to detect boot-like shapes
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boot_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (w * h * 0.01):  # At least 1% of image area
                # Check aspect ratio (boots are typically wider than tall)
                x, y, w_cont, h_cont = cv2.boundingRect(contour)
                aspect_ratio = w_cont / h_cont if h_cont > 0 else 0
                if aspect_ratio > 1.2:  # Boots are typically wider than tall
                    boot_contours.append(contour)
        
        if boot_contours:
            # Calculate total boot area
            total_boot_area = sum(cv2.contourArea(contour) for contour in boot_contours)
            total_pixels = lower_region.shape[0] * lower_region.shape[1]
            
            if total_boot_area > total_pixels * 0.05:  # At least 5% of lower region
                # Create bounding box for safety boots (lower region)
                boot_height = int(h * 0.2)  # Increased height
                boot_y1 = int(h * 0.8)  # Start higher
                
                return {
                    'bbox': [0, boot_y1, w, h],
                    'confidence': min(0.75, total_boot_area / total_pixels * 4),
                    'class_id': -4,
                    'class_name': 'safety_boots'
                }
        
        # Fallback: simple dark pixel detection with lower threshold
        dark_pixels = cv2.countNonZero(combined_mask)
        total_pixels = lower_region.shape[0] * lower_region.shape[1]
        
        if dark_pixels > total_pixels * 0.08:  # Lower threshold
            boot_height = int(h * 0.2)
            boot_y1 = int(h * 0.8)
            
            return {
                'bbox': [0, boot_y1, w, h],
                'confidence': min(0.65, dark_pixels / total_pixels * 3),
                'class_id': -4,
                'class_name': 'safety_boots'
            }
        
        return None
    
    def _detect_hard_hat(self, person_region: np.ndarray, hsv: np.ndarray, person_bbox: List[float]) -> Optional[Dict[str, Any]]:
        """Detect hard hat using color analysis in upper region"""
        h, w = person_region.shape[:2]
        
        # Focus on upper 30% of person region (head area)
        upper_region = hsv[:int(h*0.3), :]
        
        # Look for bright colors (white, yellow, orange hard hats)
        # White hard hats
        white_mask = cv2.inRange(upper_region, np.array([0, 0, 200]), np.array([180, 30, 255]))
        # Yellow/orange hard hats
        yellow_mask = cv2.inRange(upper_region, np.array([20, 100, 100]), np.array([30, 255, 255]))
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Check if we have enough bright pixels
        bright_pixels = cv2.countNonZero(combined_mask)
        total_pixels = upper_region.shape[0] * upper_region.shape[1]
        
        if bright_pixels > total_pixels * 0.1:  # At least 10% bright pixels
            # Create bounding box for hard hat (upper region of person)
            x1, y1, x2, y2 = person_bbox
            hat_height = int((y2 - y1) * 0.15)  # Hard hat is about 15% of person height
            
            return {
                'bbox': [x1, y1, x2, y1 + hat_height],
                'confidence': min(0.85, bright_pixels / total_pixels * 2),  # Scale confidence
                'class_id': -1,  # Custom class
                'class_name': 'hard_hat'
            }
        
        return None
    
    def _detect_safety_vest(self, person_region: np.ndarray, hsv: np.ndarray, person_bbox: List[float]) -> Optional[Dict[str, Any]]:
        """Detect safety vest using bright orange/yellow color analysis"""
        h, w = person_region.shape[:2]
        
        # Focus on middle 60% of person region (torso area)
        middle_start = int(h*0.2)
        middle_end = int(h*0.8)
        middle_region = hsv[middle_start:middle_end, :]
        
        # Look for bright orange/yellow colors (typical safety vest colors)
        orange_mask = cv2.inRange(middle_region, np.array([5, 100, 100]), np.array([25, 255, 255]))
        
        # Check if we have enough orange pixels
        orange_pixels = cv2.countNonZero(orange_mask)
        total_pixels = middle_region.shape[0] * middle_region.shape[1]
        
        if orange_pixels > total_pixels * 0.15:  # At least 15% orange pixels
            # Create bounding box for safety vest (middle region of person)
            x1, y1, x2, y2 = person_bbox
            vest_height = int((y2 - y1) * 0.4)  # Vest covers about 40% of person height
            vest_y1 = y1 + int((y2 - y1) * 0.2)
            
            return {
                'bbox': [x1, vest_y1, x2, vest_y1 + vest_height],
                'confidence': min(0.80, orange_pixels / total_pixels * 2),
                'class_id': -2,  # Custom class
                'class_name': 'safety_vest'
            }
        
        return None
    
    def _detect_safety_glasses(self, person_region: np.ndarray, hsv: np.ndarray, person_bbox: List[float]) -> Optional[Dict[str, Any]]:
        """Detect safety glasses using enhanced edge detection and shape analysis"""
        h, w = person_region.shape[:2]
        
        # Focus on upper 30% of person region (face area) - increased range
        upper_region = person_region[:int(h*0.3), :]
        gray_upper = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
        
        # Use multiple edge detection methods
        edges1 = cv2.Canny(gray_upper, 30, 100)  # Lower thresholds for more sensitive detection
        edges2 = cv2.Canny(gray_upper, 50, 150)  # Standard thresholds
        
        # Combine edge detections
        combined_edges = cv2.bitwise_or(edges1, edges2)
        
        # Look for horizontal lines (glasses frames)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal_lines = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Look for circular/oval patterns (lens shapes)
        circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 8))
        circular_patterns = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, circular_kernel)
        
        # Combine horizontal lines and circular patterns
        glasses_mask = cv2.bitwise_or(horizontal_lines, circular_patterns)
        
        # Find contours to detect glasses-like shapes
        contours, _ = cv2.findContours(glasses_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        glasses_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (w * h * 0.001):  # At least 0.1% of person region area
                # Check aspect ratio (glasses are typically wider than tall)
                x, y, w_cont, h_cont = cv2.boundingRect(contour)
                aspect_ratio = w_cont / h_cont if h_cont > 0 else 0
                if aspect_ratio > 1.5:  # Glasses are typically wider than tall
                    glasses_contours.append(contour)
        
        if glasses_contours:
            # Calculate total glasses area
            total_glasses_area = sum(cv2.contourArea(contour) for contour in glasses_contours)
            total_pixels = upper_region.shape[0] * upper_region.shape[1]
            
            if total_glasses_area > total_pixels * 0.005:  # At least 0.5% of upper region
                # Create bounding box for glasses (small region in upper face)
                x1, y1, x2, y2 = person_bbox
                glasses_height = int((y2 - y1) * 0.1)  # Increased height
                glasses_y1 = y1 + int((y2 - y1) * 0.05)
                
                return {
                    'bbox': [x1, glasses_y1, x2, glasses_y1 + glasses_height],
                    'confidence': min(0.80, total_glasses_area / total_pixels * 10),
                    'class_id': -3,
                    'class_name': 'safety_glasses'
                }
        
        # Fallback: simple line detection with lower threshold
        line_pixels = cv2.countNonZero(horizontal_lines)
        total_pixels = upper_region.shape[0] * upper_region.shape[1]
        
        if line_pixels > total_pixels * 0.01:  # Lower threshold
            x1, y1, x2, y2 = person_bbox
            glasses_height = int((y2 - y1) * 0.1)
            glasses_y1 = y1 + int((y2 - y1) * 0.05)
            
            return {
                'bbox': [x1, glasses_y1, x2, glasses_y1 + glasses_height],
                'confidence': min(0.75, line_pixels / total_pixels * 10),
                'class_id': -3,
                'class_name': 'safety_glasses'
            }
        
        return None
    
    def _detect_safety_boots(self, person_region: np.ndarray, hsv: np.ndarray, person_bbox: List[float]) -> Optional[Dict[str, Any]]:
        """Detect safety boots using enhanced color and shape analysis in lower region"""
        h, w = person_region.shape[:2]
        
        # Focus on lower 25% of person region (feet area) - increased range
        lower_region = hsv[int(h*0.75):, :]
        lower_region_bgr = person_region[int(h*0.75):, :]
        
        # Look for dark colors (black, brown, dark gray boots)
        dark_mask = cv2.inRange(lower_region, np.array([0, 0, 0]), np.array([180, 255, 120]))
        
        # Also look for brown colors (common boot color)
        brown_mask = cv2.inRange(lower_region, np.array([10, 50, 20]), np.array([20, 255, 100]))
        
        # Combine masks
        combined_mask = cv2.bitwise_or(dark_mask, brown_mask)
        
        # Use morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours to detect boot-like shapes
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boot_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (w * h * 0.005):  # At least 0.5% of person region area
                # Check aspect ratio (boots are typically wider than tall)
                x, y, w_cont, h_cont = cv2.boundingRect(contour)
                aspect_ratio = w_cont / h_cont if h_cont > 0 else 0
                if aspect_ratio > 1.0:  # Boots are typically wider than tall
                    boot_contours.append(contour)
        
        if boot_contours:
            # Calculate total boot area
            total_boot_area = sum(cv2.contourArea(contour) for contour in boot_contours)
            total_pixels = lower_region.shape[0] * lower_region.shape[1]
            
            if total_boot_area > total_pixels * 0.1:  # At least 10% of lower region
                # Create bounding box for safety boots (lower region of person)
                x1, y1, x2, y2 = person_bbox
                boot_height = int((y2 - y1) * 0.2)  # Increased height
                boot_y1 = y2 - boot_height
                
                return {
                    'bbox': [x1, boot_y1, x2, y2],
                    'confidence': min(0.80, total_boot_area / total_pixels * 2),
                    'class_id': -4,
                    'class_name': 'safety_boots'
                }
        
        # Fallback: simple dark pixel detection with lower threshold
        dark_pixels = cv2.countNonZero(combined_mask)
        total_pixels = lower_region.shape[0] * lower_region.shape[1]
        
        if dark_pixels > total_pixels * 0.15:  # Lower threshold
            x1, y1, x2, y2 = person_bbox
            boot_height = int((y2 - y1) * 0.2)
            boot_y1 = y2 - boot_height
            
            return {
                'bbox': [x1, boot_y1, x2, y2],
                'confidence': min(0.70, dark_pixels / total_pixels * 1.5),
                'class_id': -4,
                'class_name': 'safety_boots'
            }
        
        return None
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for detections"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold set to {self.confidence_threshold}")
    
    def set_nms_threshold(self, threshold: float):
        """Set NMS threshold for detections"""
        self.nms_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"NMS threshold set to {self.nms_threshold}")
    
    def get_class_names(self) -> List[str]:
        """Get list of available class names"""
        return list(self.class_names.values())
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None


class CustomEquipmentDetector(EquipmentDetector):
    """
    Custom detector trained on specific equipment classes
    """
    
    def __init__(self, model_path: str, equipment_classes: List[str], device: str = "auto"):
        """
        Initialize custom detector
        
        Args:
            model_path: Path to custom trained model
            equipment_classes: List of equipment class names
            device: Device to run inference on
        """
        self.equipment_classes = equipment_classes
        super().__init__(model_path, device)
    
    def detect_equipment(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect equipment using custom model
        
        Args:
            image: Input image
            
        Returns:
            List of equipment detections
        """
        all_detections = self.detect(image)
        
        # Filter for equipment classes
        equipment_detections = [
            det for det in all_detections 
            if det['class_name'] in self.equipment_classes
        ]
        
        return equipment_detections
    
    def get_equipment_classes(self) -> List[str]:
        """Get list of equipment classes this model can detect"""
        return self.equipment_classes.copy()

