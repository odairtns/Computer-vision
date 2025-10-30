"""
Training script for equipment detection model
"""
import os
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics import YOLO
import wandb
from tqdm import tqdm

from dataset import EquipmentDataset
from augmentations import EquipmentAugmentation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EquipmentTrainer:
    """
    Trainer class for equipment detection model
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.dataset = None
        self.model = None
        self.augmentation = EquipmentAugmentation(
            image_size=config.get('image_size', (640, 640))
        )
        
        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_dataset(self):
        """Setup training and validation datasets"""
        data_dir = self.config['data_dir']
        classes = self.config['classes']
        
        # Create dataset
        self.dataset = EquipmentDataset(data_dir, classes)
        
        # Split dataset if needed
        if self.config.get('split_dataset', True):
            self.dataset.split_dataset(
                train_ratio=self.config.get('train_ratio', 0.8),
                val_ratio=self.config.get('val_ratio', 0.1),
                test_ratio=self.config.get('test_ratio', 0.1)
            )
        
        logger.info(f"Dataset setup complete: {len(self.dataset)} samples")
    
    def setup_model(self):
        """Setup YOLO model"""
        model_name = self.config.get('model_name', 'yolov8n.pt')
        
        # Load pretrained model
        self.model = YOLO(model_name)
        
        # Move to device
        self.model.to(self.device)
        
        logger.info(f"Model setup complete: {model_name}")
    
    def create_yolo_config(self):
        """Create YOLO dataset configuration"""
        if not self.dataset:
            raise RuntimeError("Dataset not setup")
        
        config_path = Path(self.config['data_dir']) / 'dataset.yaml'
        self.dataset.create_yolo_config(str(config_path))
        
        logger.info(f"YOLO config created: {config_path}")
        return str(config_path)
    
    def train(self):
        """Train the model"""
        if not self.model or not self.dataset:
            raise RuntimeError("Model or dataset not setup")
        
        # Create YOLO config
        config_path = self.create_yolo_config()
        
        # Training parameters
        train_params = {
            'data': config_path,
            'epochs': self.config.get('epochs', 100),
            'imgsz': self.config['image_size'][0],
            'batch': self.config.get('batch_size', 16),
            'device': str(self.device),
            'project': self.config.get('project_name', 'equipment_detection'),
            'name': self.config.get('run_name', 'train'),
            'save': True,
            'save_period': self.config.get('save_period', 10),
            'patience': self.config.get('patience', 50),
            'lr0': self.config.get('learning_rate', 0.01),
            'lrf': self.config.get('final_lr_factor', 0.1),
            'momentum': self.config.get('momentum', 0.937),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'warmup_epochs': self.config.get('warmup_epochs', 3),
            'warmup_momentum': self.config.get('warmup_momentum', 0.8),
            'warmup_bias_lr': self.config.get('warmup_bias_lr', 0.1),
            'box': self.config.get('box_loss_gain', 0.05),
            'cls': self.config.get('cls_loss_gain', 0.5),
            'dfl': self.config.get('dfl_loss_gain', 1.5),
            'pose': self.config.get('pose_loss_gain', 12.0),
            'kobj': self.config.get('kobj_loss_gain', 2.0),
            'label_smoothing': self.config.get('label_smoothing', 0.0),
            'nbs': self.config.get('nominal_batch_size', 64),
            'overlap_mask': self.config.get('overlap_mask', True),
            'mask_ratio': self.config.get('mask_ratio', 4),
            'dropout': self.config.get('dropout', 0.0),
            'val': True,
            'plots': True,
            'verbose': True,
            'seed': self.config.get('seed', 0),
            'deterministic': self.config.get('deterministic', True),
            'single_cls': self.config.get('single_cls', False),
            'rect': self.config.get('rect', False),
            'cos_lr': self.config.get('cos_lr', False),
            'close_mosaic': self.config.get('close_mosaic', 10),
            'resume': self.config.get('resume', False),
            'amp': self.config.get('amp', True),
            'fraction': self.config.get('fraction', 1.0),
            'profile': self.config.get('profile', False),
            'freeze': self.config.get('freeze', None),
            'multi_scale': self.config.get('multi_scale', False),
            'overlap_mask': self.config.get('overlap_mask', True),
            'mask_ratio': self.config.get('mask_ratio', 4),
            'dropout': self.config.get('dropout', 0.0),
        }
        
        # Start training
        logger.info("Starting training...")
        results = self.model.train(**train_params)
        
        logger.info("Training completed!")
        return results
    
    def validate(self):
        """Validate the model"""
        if not self.model:
            raise RuntimeError("Model not setup")
        
        config_path = self.create_yolo_config()
        
        # Validation parameters
        val_params = {
            'data': config_path,
            'imgsz': self.config['image_size'][0],
            'batch': self.config.get('batch_size', 16),
            'device': str(self.device),
            'project': self.config.get('project_name', 'equipment_detection'),
            'name': 'validation',
            'save_json': True,
            'save_hybrid': False,
            'conf': self.config.get('conf_threshold', 0.001),
            'iou': self.config.get('iou_threshold', 0.6),
            'max_det': self.config.get('max_det', 300),
            'half': self.config.get('half_precision', True),
            'dnn': self.config.get('dnn', False),
            'plots': True,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'boxes': True,
        }
        
        logger.info("Starting validation...")
        results = self.model.val(**val_params)
        
        logger.info("Validation completed!")
        return results
    
    def export_model(self, format: str = 'onnx'):
        """Export model to different formats"""
        if not self.model:
            raise RuntimeError("Model not setup")
        
        export_path = self.model.export(format=format)
        logger.info(f"Model exported to: {export_path}")
        return export_path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config(output_path: str):
    """Create default training configuration"""
    config = {
        'data_dir': 'data',
        'classes': [
            'hard_hat', 'safety_vest', 'safety_boots', 'safety_glasses',
            'gloves', 'ear_protection', 'respirator', 'scrubs', 'stethoscope',
            'face_mask', 'lab_coat', 'safety_goggles', 'apron', 'shoe_covers',
            'hair_net', 'safety_shoes'
        ],
        'image_size': [640, 640],
        'batch_size': 16,
        'epochs': 100,
        'learning_rate': 0.01,
        'model_name': 'yolov8n.pt',
        'project_name': 'equipment_detection',
        'run_name': 'train',
        'split_dataset': True,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'save_period': 10,
        'patience': 50,
        'conf_threshold': 0.001,
        'iou_threshold': 0.6,
        'max_det': 300,
        'half_precision': True,
        'deterministic': True,
        'seed': 0,
        'amp': True,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'multi_scale': False,
        'rect': False,
        'single_cls': False,
        'label_smoothing': 0.0,
        'dropout': 0.0,
        'box_loss_gain': 0.05,
        'cls_loss_gain': 0.5,
        'dfl_loss_gain': 1.5,
        'pose_loss_gain': 12.0,
        'kobj_loss_gain': 2.0,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'final_lr_factor': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'overlap_mask': True,
        'mask_ratio': 4,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'dnn': False,
        'agnostic_nms': False,
        'retina_masks': False,
        'boxes': True,
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Default config created: {output_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train equipment detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--create-config', type=str, help='Create default config file')
    parser.add_argument('--validate-only', action='store_true', help='Only run validation')
    parser.add_argument('--export', type=str, help='Export model format (onnx, torchscript, etc.)')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        create_default_config(args.create_config)
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize trainer
    trainer = EquipmentTrainer(config)
    
    # Setup components
    trainer.setup_dataset()
    trainer.setup_model()
    
    if args.validate_only:
        # Run validation only
        trainer.validate()
    else:
        # Run training
        trainer.train()
        
        # Run validation after training
        trainer.validate()
    
    # Export model if requested
    if args.export:
        trainer.export_model(args.export)


if __name__ == '__main__':
    main()


