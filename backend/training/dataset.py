"""
Dataset handling for equipment detection training
"""
import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class EquipmentDataset:
    """
    Dataset class for equipment detection training
    """
    
    def __init__(self, data_dir: str, classes: List[str]):
        """
        Initialize dataset
        
        Args:
            data_dir: Path to dataset directory
            classes: List of class names
        """
        self.data_dir = Path(data_dir)
        self.classes = classes
        self.class_to_id = {cls: idx for idx, cls in enumerate(classes)}
        self.id_to_class = {idx: cls for idx, cls in enumerate(classes)}
        
        # Dataset structure
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        
        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.samples = self._load_dataset()
        
        logger.info(f"Loaded dataset with {len(self.samples)} samples and {len(classes)} classes")
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset samples"""
        samples = []
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for image_path in self.images_dir.iterdir():
            if image_path.suffix.lower() in image_extensions:
                # Find corresponding label file
                label_path = self.labels_dir / f"{image_path.stem}.txt"
                
                if label_path.exists():
                    samples.append({
                        'image_path': str(image_path),
                        'label_path': str(label_path)
                    })
                else:
                    logger.warning(f"No label file found for {image_path}")
        
        return samples
    
    def get_sample(self, index: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Get sample by index
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (image, annotations)
        """
        if index >= len(self.samples):
            raise IndexError(f"Index {index} out of range")
        
        sample = self.samples[index]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise ValueError(f"Could not load image: {sample['image_path']}")
        
        # Load annotations
        annotations = self._load_annotations(sample['label_path'])
        
        return image, annotations
    
    def _load_annotations(self, label_path: str) -> List[Dict[str, Any]]:
        """
        Load YOLO format annotations
        
        Args:
            label_path: Path to label file
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to absolute coordinates
                bbox = {
                    'class_id': class_id,
                    'class_name': self.id_to_class.get(class_id, f"class_{class_id}"),
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                }
                
                annotations.append(bbox)
        
        except Exception as e:
            logger.error(f"Failed to load annotations from {label_path}: {e}")
        
        return annotations
    
    def save_annotation(self, image_path: str, annotations: List[Dict[str, Any]]):
        """
        Save annotations in YOLO format
        
        Args:
            image_path: Path to image file
            annotations: List of annotation dictionaries
        """
        # Get label file path
        image_name = Path(image_path).stem
        label_path = self.labels_dir / f"{image_name}.txt"
        
        try:
            with open(label_path, 'w') as f:
                for ann in annotations:
                    line = f"{ann['class_id']} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}\n"
                    f.write(line)
            
            logger.info(f"Saved annotations to {label_path}")
        
        except Exception as e:
            logger.error(f"Failed to save annotations to {label_path}: {e}")
    
    def create_yolo_config(self, output_path: str):
        """
        Create YOLO dataset configuration file
        
        Args:
            output_path: Path to save config file
        """
        config = {
            'path': str(self.data_dir.absolute()),
            'train': 'images',
            'val': 'images',  # Using same directory for train/val in this example
            'nc': len(self.classes),
            'names': self.classes
        }
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Created YOLO config at {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to create YOLO config: {e}")
    
    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """
        Split dataset into train/validation/test sets
        
        Args:
            train_ratio: Ratio of training samples
            val_ratio: Ratio of validation samples
            test_ratio: Ratio of test samples
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Shuffle samples
        import random
        random.shuffle(self.samples)
        
        # Calculate split indices
        n_samples = len(self.samples)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # Split samples
        train_samples = self.samples[:train_end]
        val_samples = self.samples[train_end:val_end]
        test_samples = self.samples[val_end:]
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (self.data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copy files to split directories
        splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        for split_name, samples in splits.items():
            for sample in samples:
                # Copy image
                src_image = Path(sample['image_path'])
                dst_image = self.data_dir / split_name / 'images' / src_image.name
                dst_image.write_bytes(src_image.read_bytes())
                
                # Copy label
                src_label = Path(sample['label_path'])
                dst_label = self.data_dir / split_name / 'labels' / src_label.name
                dst_label.write_bytes(src_label.read_bytes())
        
        logger.info(f"Split dataset: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in dataset"""
        class_counts = {cls: 0 for cls in self.classes}
        
        for sample in self.samples:
            annotations = self._load_annotations(sample['label_path'])
            for ann in annotations:
                class_name = ann['class_name']
                if class_name in class_counts:
                    class_counts[class_name] += 1
        
        return class_counts
    
    def visualize_sample(self, index: int, save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize sample with bounding boxes
        
        Args:
            index: Sample index
            save_path: Path to save visualization (optional)
        
        Returns:
            Annotated image
        """
        image, annotations = self.get_sample(index)
        annotated = image.copy()
        
        # Define colors
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        h, w = image.shape[:2]
        
        for ann in annotations:
            # Convert normalized coordinates to absolute
            x_center = int(ann['x_center'] * w)
            y_center = int(ann['y_center'] * h)
            width = int(ann['width'] * w)
            height = int(ann['height'] * h)
            
            # Calculate bounding box corners
            x1 = x_center - width // 2
            y1 = y_center - height // 2
            x2 = x_center + width // 2
            y2 = y_center + height // 2
            
            # Get color
            color = colors[ann['class_id'] % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{ann['class_name']}"
            cv2.putText(annotated, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if save_path:
            cv2.imwrite(save_path, annotated)
            logger.info(f"Saved visualization to {save_path}")
        
        return annotated
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.get_sample(index)


