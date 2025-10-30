"""
Data augmentation strategies for equipment detection training
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)


class EquipmentAugmentation:
    """
    Data augmentation pipeline for equipment detection
    """
    
    def __init__(self, image_size: Tuple[int, int] = (640, 640)):
        """
        Initialize augmentation pipeline
        
        Args:
            image_size: Target image size (width, height)
        """
        self.image_size = image_size
        self.setup_augmentations()
    
    def setup_augmentations(self):
        """Setup augmentation pipelines"""
        
        # Training augmentations (strong)
        self.train_transform = A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.1),
            A.Rotate(limit=15, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.3
            ),
            
            # Perspective and distortion
            A.Perspective(scale=(0.05, 0.1), p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1),
            
            # Color and brightness
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.3
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.MotionBlur(blur_limit=7, p=0.2),
            
            # Weather effects
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=1,
                brightness_coefficient=0.7,
                rain_type="drizzle",
                p=0.1
            ),
            A.RandomShadow(p=0.1),
            
            # Cutout and mixup
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.2
            ),
            
            # Resize and normalize
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
        
        # Validation augmentations (light)
        self.val_transform = A.Compose([
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def augment_image(self, image: np.ndarray, bboxes: List[List[float]], 
                     class_labels: List[int], is_training: bool = True) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Apply augmentations to image and bounding boxes
        
        Args:
            image: Input image
            bboxes: List of bounding boxes in YOLO format [x_center, y_center, width, height]
            class_labels: List of class labels
            is_training: Whether to use training or validation augmentations
        
        Returns:
            Tuple of (augmented_image, augmented_bboxes, class_labels)
        """
        transform = self.train_transform if is_training else self.val_transform
        
        try:
            augmented = transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            return (
                augmented['image'],
                augmented['bboxes'],
                augmented['class_labels']
            )
        
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}, returning original")
            return image, bboxes, class_labels
    
    def create_mosaic(self, images: List[np.ndarray], bboxes_list: List[List[List[float]]], 
                     class_labels_list: List[List[int]]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Create mosaic augmentation (4 images in one)
        
        Args:
            images: List of 4 images
            bboxes_list: List of bbox lists for each image
            class_labels_list: List of class label lists for each image
        
        Returns:
            Tuple of (mosaic_image, combined_bboxes, combined_labels)
        """
        if len(images) != 4:
            raise ValueError("Mosaic requires exactly 4 images")
        
        h, w = self.image_size[1], self.image_size[0]
        mosaic = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Calculate positions for 2x2 grid
        positions = [
            (0, 0, w//2, h//2),           # Top-left
            (w//2, 0, w, h//2),           # Top-right
            (0, h//2, w//2, h),           # Bottom-left
            (w//2, h//2, w, h)            # Bottom-right
        ]
        
        combined_bboxes = []
        combined_labels = []
        
        for i, (image, bboxes, class_labels) in enumerate(zip(images, bboxes_list, class_labels_list)):
            x1, y1, x2, y2 = positions[i]
            
            # Resize image to fit position
            resized_image = cv2.resize(image, (x2-x1, y2-y1))
            mosaic[y1:y2, x1:x2] = resized_image
            
            # Adjust bounding boxes
            scale_x = (x2 - x1) / image.shape[1]
            scale_y = (y2 - y1) / image.shape[0]
            
            for bbox, label in zip(bboxes, class_labels):
                x_center, y_center, width, height = bbox
                
                # Scale and offset coordinates
                new_x_center = (x_center * scale_x + x1) / w
                new_y_center = (y_center * scale_y + y1) / h
                new_width = width * scale_x
                new_height = height * scale_y
                
                combined_bboxes.append([new_x_center, new_y_center, new_width, new_height])
                combined_labels.append(label)
        
        return mosaic, combined_bboxes, combined_labels
    
    def mixup(self, image1: np.ndarray, bboxes1: List[List[float]], labels1: List[int],
              image2: np.ndarray, bboxes2: List[List[float]], labels2: List[int],
              alpha: float = 0.2) -> Tuple[np.ndarray, List[List[float]], List[int], List[float]]:
        """
        Apply mixup augmentation
        
        Args:
            image1: First image
            bboxes1: Bounding boxes for first image
            labels1: Labels for first image
            image2: Second image
            bboxes2: Bounding boxes for second image
            labels2: Labels for second image
            alpha: Mixup parameter
        
        Returns:
            Tuple of (mixed_image, combined_bboxes, combined_labels, mix_weights)
        """
        # Generate mixup ratio
        lam = np.random.beta(alpha, alpha)
        
        # Mix images
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_image = mixed_image.astype(np.uint8)
        
        # Combine bounding boxes and labels
        combined_bboxes = bboxes1 + bboxes2
        combined_labels = labels1 + labels2
        
        # Mix weights for loss calculation
        mix_weights = [lam] * len(labels1) + [1 - lam] * len(labels2)
        
        return mixed_image, combined_bboxes, combined_labels, mix_weights
    
    def cutmix(self, image1: np.ndarray, bboxes1: List[List[float]], labels1: List[int],
               image2: np.ndarray, bboxes2: List[List[float]], labels2: List[int],
               alpha: float = 1.0) -> Tuple[np.ndarray, List[List[float]], List[int], List[float]]:
        """
        Apply CutMix augmentation
        
        Args:
            image1: First image
            bboxes1: Bounding boxes for first image
            labels1: Labels for first image
            image2: Second image
            bboxes2: Bounding boxes for second image
            labels2: Labels for second image
            alpha: CutMix parameter
        
        Returns:
            Tuple of (cutmixed_image, combined_bboxes, combined_labels, mix_weights)
        """
        h, w = image1.shape[:2]
        
        # Generate cutmix ratio
        lam = np.random.beta(alpha, alpha)
        
        # Calculate cut region
        cut_rat = np.sqrt(1 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # Random center
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Calculate bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        cutmixed_image = image1.copy()
        cutmixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]
        
        # Adjust bounding boxes
        combined_bboxes = []
        combined_labels = []
        mix_weights = []
        
        # Process first image bboxes
        for bbox, label in zip(bboxes1, labels1):
            x_center, y_center, width, height = bbox
            
            # Check if bbox overlaps with cut region
            bbox_x1 = (x_center - width/2) * w
            bbox_y1 = (y_center - height/2) * h
            bbox_x2 = (x_center + width/2) * w
            bbox_y2 = (y_center + height/2) * h
            
            # Calculate intersection
            intersection = max(0, min(bbox_x2, bbx2) - max(bbox_x1, bbx1)) * \
                          max(0, min(bbox_y2, bby2) - max(bbox_y1, bby1))
            bbox_area = (bbox_x2 - bbox_x1) * (bbox_y2 - bby1)
            
            if bbox_area > 0:
                overlap_ratio = intersection / bbox_area
                combined_bboxes.append(bbox)
                combined_labels.append(label)
                mix_weights.append(1 - overlap_ratio)
        
        # Process second image bboxes
        for bbox, label in zip(bboxes2, labels2):
            x_center, y_center, width, height = bbox
            
            # Check if bbox is in cut region
            bbox_x1 = (x_center - width/2) * w
            bbox_y1 = (y_center - height/2) * h
            bbox_x2 = (x_center + width/2) * w
            bbox_y2 = (y_center + height/2) * h
            
            # Check overlap with cut region
            if (bbox_x1 < bbx2 and bbox_x2 > bbx1 and 
                bbox_y1 < bby2 and bbox_y2 > bby1):
                combined_bboxes.append(bbox)
                combined_labels.append(label)
                mix_weights.append(1.0)
        
        return cutmixed_image, combined_bboxes, combined_labels, mix_weights


