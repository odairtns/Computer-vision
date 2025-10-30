"""
Equipment verification logic for checking required equipment presence
"""
import json
import os
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EquipmentVerifier:
    """
    Verifies if all required equipment is present based on detections
    """
    
    def __init__(self, checklist_path: Optional[str] = None):
        """
        Initialize verifier with equipment checklist
        
        Args:
            checklist_path: Path to equipment checklist JSON file
        """
        self.checklist = {}
        self.available_checklists = {}
        
        if checklist_path and os.path.exists(checklist_path):
            self.load_checklist(checklist_path)
        else:
            self._load_default_checklists()
    
    def _load_default_checklists(self):
        """Load default equipment checklists"""
        default_checklists = {
            "construction": {
                "name": "Construction Safety Equipment",
                "description": "Required safety equipment for construction workers",
                "required_equipment": [
                    "hard_hat",
                    "safety_vest",
                    "safety_boots",
                    "safety_glasses"
                ],
                "optional_equipment": [
                    "gloves",
                    "ear_protection",
                    "respirator"
                ]
            },
            "medical": {
                "name": "Medical Equipment",
                "description": "Required medical equipment for healthcare workers",
                "required_equipment": [
                    "scrubs",
                    "stethoscope",
                    "face_mask",
                    "gloves"
                ],
                "optional_equipment": [
                    "safety_goggles",
                    "apron",
                    "shoe_covers"
                ]
            },
            "laboratory": {
                "name": "Laboratory Safety Equipment",
                "description": "Required safety equipment for laboratory workers",
                "required_equipment": [
                    "lab_coat",
                    "safety_goggles",
                    "gloves",
                    "face_mask"
                ],
                "optional_equipment": [
                    "safety_shoes",
                    "apron",
                    "hair_net"
                ]
            }
        }
        
        self.available_checklists = default_checklists
        logger.info(f"Loaded {len(default_checklists)} default checklists")
    
    def load_checklist(self, checklist_path: str):
        """
        Load equipment checklist from JSON file
        
        Args:
            checklist_path: Path to checklist JSON file
        """
        try:
            with open(checklist_path, 'r') as f:
                checklist_data = json.load(f)
            
            if isinstance(checklist_data, dict):
                if "name" in checklist_data:
                    # Single checklist
                    self.checklist = checklist_data
                else:
                    # Multiple checklists
                    self.available_checklists = checklist_data
            else:
                raise ValueError("Invalid checklist format")
            
            logger.info(f"Loaded checklist from {checklist_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checklist from {checklist_path}: {e}")
            raise
    
    def set_checklist(self, checklist_name: str):
        """
        Set active checklist by name
        
        Args:
            checklist_name: Name of the checklist to use
        """
        if checklist_name in self.available_checklists:
            self.checklist = self.available_checklists[checklist_name]
            logger.info(f"Set active checklist to: {checklist_name}")
        else:
            raise ValueError(f"Checklist '{checklist_name}' not found")
    
    def verify_equipment(
        self, 
        detections: List[Dict[str, Any]], 
        checklist_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify if all required equipment is present
        
        Args:
            detections: List of detected objects
            checklist_name: Name of checklist to use (optional)
        
        Returns:
            Verification result dictionary
        """
        if checklist_name:
            self.set_checklist(checklist_name)
        
        if not self.checklist:
            raise ValueError("No checklist loaded")
        
        # Extract detected equipment classes
        detected_classes = set()
        for detection in detections:
            class_name = detection.get('class_name', '').lower()
            detected_classes.add(class_name)
        
        # Get required and optional equipment
        required_equipment = set(
            item.lower() for item in self.checklist.get('required_equipment', [])
        )
        optional_equipment = set(
            item.lower() for item in self.checklist.get('optional_equipment', [])
        )
        
        # Check for missing required equipment
        missing_equipment = required_equipment - detected_classes
        
        # Check for present equipment
        present_equipment = detected_classes.intersection(required_equipment.union(optional_equipment))
        
        # Determine verification status
        status = "PASS" if len(missing_equipment) == 0 else "FAIL"
        
        # Calculate confidence based on detection confidences
        confidence = self._calculate_verification_confidence(detections, present_equipment)
        
        result = {
            "status": status,
            "missing_equipment": list(missing_equipment),
            "present_equipment": list(present_equipment),
            "confidence": confidence,
            "required_equipment": list(required_equipment),
            "optional_equipment": list(optional_equipment),
            "detected_equipment": list(detected_classes)
        }
        
        logger.info(f"Verification result: {status}, Missing: {missing_equipment}")
        
        return result
    
    def _calculate_verification_confidence(
        self, 
        detections: List[Dict[str, Any]], 
        present_equipment: Set[str]
    ) -> float:
        """
        Calculate overall verification confidence
        
        Args:
            detections: List of detections
            present_equipment: Set of present equipment classes
        
        Returns:
            Confidence score (0-1)
        """
        if not detections:
            return 0.0
        
        # Calculate average confidence of detected equipment
        equipment_detections = [
            det for det in detections 
            if det.get('class_name', '').lower() in present_equipment
        ]
        
        if not equipment_detections:
            return 0.0
        
        avg_confidence = sum(det.get('confidence', 0) for det in equipment_detections) / len(equipment_detections)
        
        # Apply penalty for missing required equipment
        required_equipment = set(
            item.lower() for item in self.checklist.get('required_equipment', [])
        )
        missing_count = len(required_equipment - present_equipment)
        penalty = missing_count * 0.1  # 10% penalty per missing item
        
        final_confidence = max(0.0, avg_confidence - penalty)
        
        return min(1.0, final_confidence)
    
    def get_available_checklists(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available checklists"""
        return self.available_checklists.copy()
    
    def get_current_checklist(self) -> Dict[str, Any]:
        """Get current active checklist"""
        return self.checklist.copy()
    
    def add_equipment_class(self, class_name: str, is_required: bool = True):
        """
        Add equipment class to current checklist
        
        Args:
            class_name: Name of equipment class
            is_required: Whether this equipment is required
        """
        if not self.checklist:
            self.checklist = {
                "name": "Custom Checklist",
                "description": "Custom equipment checklist",
                "required_equipment": [],
                "optional_equipment": []
            }
        
        if is_required:
            if class_name not in self.checklist["required_equipment"]:
                self.checklist["required_equipment"].append(class_name)
        else:
            if class_name not in self.checklist["optional_equipment"]:
                self.checklist["optional_equipment"].append(class_name)
        
        logger.info(f"Added {'required' if is_required else 'optional'} equipment: {class_name}")
    
    def remove_equipment_class(self, class_name: str):
        """
        Remove equipment class from current checklist
        
        Args:
            class_name: Name of equipment class to remove
        """
        if not self.checklist:
            return
        
        if class_name in self.checklist["required_equipment"]:
            self.checklist["required_equipment"].remove(class_name)
        elif class_name in self.checklist["optional_equipment"]:
            self.checklist["optional_equipment"].remove(class_name)
        
        logger.info(f"Removed equipment: {class_name}")
    
    def save_checklist(self, file_path: str):
        """
        Save current checklist to file
        
        Args:
            file_path: Path to save checklist
        """
        if not self.checklist:
            raise ValueError("No checklist to save")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.checklist, f, indent=2)
            
            logger.info(f"Saved checklist to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checklist to {file_path}: {e}")
            raise

