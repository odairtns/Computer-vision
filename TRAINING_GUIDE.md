# Equipment Detection Model Training Guide

This guide explains how to train and extend the equipment verification system for new equipment types and use cases.

## 📋 Table of Contents

1. [Dataset Preparation](#dataset-preparation)
2. [Training Setup](#training-setup)
3. [Training Process](#training-process)
4. [Model Evaluation](#model-evaluation)
5. [Deployment](#deployment)
6. [Extending for New Equipment](#extending-for-new-equipment)
7. [Troubleshooting](#troubleshooting)

## 🗂️ Dataset Preparation

### Dataset Structure

Organize your dataset in the following structure:

```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── checklists/
    ├── construction.json
    ├── medical.json
    └── laboratory.json
```

### Labeling Format

Use YOLO format for bounding box annotations. Each label file should contain:

```
class_id x_center y_center width height
```

Where:
- `class_id`: Integer class ID (0, 1, 2, ...)
- `x_center`, `y_center`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized dimensions (0-1)

### Example Label File

```
0 0.5 0.3 0.2 0.4
1 0.7 0.6 0.15 0.25
2 0.3 0.8 0.1 0.2
```

### Equipment Classes

Default classes included in the system:

```python
EQUIPMENT_CLASSES = [
    'hard_hat', 'safety_vest', 'safety_boots', 'safety_glasses',
    'gloves', 'ear_protection', 'respirator', 'scrubs', 'stethoscope',
    'face_mask', 'lab_coat', 'safety_goggles', 'apron', 'shoe_covers',
    'hair_net', 'safety_shoes'
]
```

## ⚙️ Training Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
pip install albumentations wandb  # For advanced training features
```

### 2. Create Training Configuration

```bash
cd backend/training
python train.py --create-config config.yaml
```

### 3. Customize Configuration

Edit `config.yaml` to match your requirements:

```yaml
data_dir: 'data'
classes:
  - 'hard_hat'
  - 'safety_vest'
  - 'safety_boots'
  - 'safety_glasses'
  - 'gloves'
  - 'ear_protection'
  - 'respirator'
  - 'scrubs'
  - 'stethoscope'
  - 'face_mask'
  - 'lab_coat'
  - 'safety_goggles'
  - 'apron'
  - 'shoe_covers'
  - 'hair_net'
  - 'safety_shoes'

image_size: [640, 640]
batch_size: 16
epochs: 100
learning_rate: 0.01
model_name: 'yolov8n.pt'
project_name: 'equipment_detection'
run_name: 'train'
```

## 🚀 Training Process

### 1. Start Training

```bash
python train.py --config config.yaml
```

### 2. Monitor Training

The training process will:
- Automatically split your dataset into train/validation/test sets
- Apply data augmentation
- Train the YOLOv8 model
- Save checkpoints and best model
- Generate training plots and metrics

### 3. Training Outputs

Training creates the following outputs:

```
equipment_detection/
├── train/
│   ├── weights/
│   │   ├── best.pt          # Best model weights
│   │   └── last.pt          # Last epoch weights
│   ├── results.png          # Training curves
│   ├── confusion_matrix.png # Confusion matrix
│   └── val_batch0_labels.jpg # Validation samples
```

## 📊 Model Evaluation

### 1. Run Validation

```bash
python train.py --config config.yaml --validate-only
```

### 2. Key Metrics

Monitor these metrics during training:

- **mAP50**: Mean Average Precision at IoU 0.5
- **mAP50-95**: Mean Average Precision at IoU 0.5-0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### 3. Model Performance

A good model should achieve:
- mAP50 > 0.7 for equipment detection
- mAP50-95 > 0.5
- Balanced precision and recall across all classes

## 🚀 Deployment

### 1. Export Model

```bash
python train.py --config config.yaml --export onnx
```

### 2. Update Backend

Replace the model in your backend:

```python
# In app/models/detector.py
detector = EquipmentDetector(model_path='path/to/your/best.pt')
```

### 3. Test Deployment

```bash
# Start the backend
cd backend
python -m uvicorn app.main:app --reload

# Test with sample image
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_image.jpg" \
  -F "checklist=construction"
```

## 🔧 Extending for New Equipment

### 1. Add New Equipment Classes

1. **Update class list** in `config.yaml`:
```yaml
classes:
  - 'hard_hat'
  - 'safety_vest'
  - 'new_equipment_type'  # Add your new class
```

2. **Update equipment classes** in `app/main.py`:
```python
EQUIPMENT_CLASSES = [
    "hard_hat", "safety_vest", "safety_boots", "safety_glasses",
    "gloves", "ear_protection", "respirator", "scrubs", "stethoscope",
    "face_mask", "lab_coat", "safety_goggles", "apron", "shoe_covers",
    "hair_net", "safety_shoes", "new_equipment_type"  # Add here
]
```

3. **Create new checklist** in `data/checklists/`:
```json
{
  "name": "Custom Equipment Checklist",
  "description": "Required equipment for custom use case",
  "required_equipment": [
    "hard_hat",
    "safety_vest",
    "new_equipment_type"
  ],
  "optional_equipment": [
    "gloves",
    "safety_glasses"
  ]
}
```

### 2. Collect and Label Data

1. **Collect images** of people wearing the new equipment
2. **Label using tools** like:
   - [LabelImg](https://github.com/tzutalin/labelImg)
   - [Roboflow](https://roboflow.com/)
   - [CVAT](https://github.com/openvinotoolkit/cvat)

3. **Ensure diversity** in your dataset:
   - Different lighting conditions
   - Various angles and poses
   - Different backgrounds
   - Multiple people

### 3. Retrain Model

```bash
# Update config with new classes
python train.py --config config.yaml

# The model will automatically detect new classes
```

## 🔍 Advanced Training Techniques

### 1. Transfer Learning

Start with a pretrained model:

```yaml
model_name: 'yolov8s.pt'  # or yolov8m.pt, yolov8l.pt, yolov8x.pt
```

### 2. Data Augmentation

Customize augmentation in `training/augmentations.py`:

```python
# Add custom augmentations
A.RandomBrightnessContrast(
    brightness_limit=0.3,  # Increase for more variation
    contrast_limit=0.3,
    p=0.7
),
```

### 3. Hyperparameter Tuning

Experiment with different settings:

```yaml
learning_rate: 0.001      # Lower for fine-tuning
batch_size: 32            # Increase if you have more GPU memory
epochs: 200               # More epochs for better convergence
weight_decay: 0.0001      # Regularization
```

### 4. Multi-GPU Training

```bash
# For multiple GPUs
python train.py --config config.yaml --device 0,1,2,3
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size
   - Use smaller model (yolov8n.pt)
   - Enable gradient checkpointing

2. **Poor Detection Performance**
   - Increase dataset size
   - Improve annotation quality
   - Adjust confidence threshold
   - Use data augmentation

3. **Training Not Converging**
   - Check learning rate
   - Verify data quality
   - Ensure balanced classes
   - Use learning rate scheduling

4. **Slow Training**
   - Use GPU acceleration
   - Reduce image size
   - Use mixed precision training
   - Optimize data loading

### Debugging Tips

1. **Visualize Dataset**:
```python
from training.dataset import EquipmentDataset
dataset = EquipmentDataset('data', classes)
dataset.visualize_sample(0, 'sample_visualization.jpg')
```

2. **Check Class Distribution**:
```python
distribution = dataset.get_class_distribution()
print(distribution)
```

3. **Validate Annotations**:
```python
# Check if annotations are properly formatted
for i in range(len(dataset)):
    image, annotations = dataset[i]
    print(f"Sample {i}: {len(annotations)} annotations")
```

## 📈 Performance Optimization

### 1. Model Optimization

- Use TensorRT for inference acceleration
- Quantize model for smaller size
- Optimize input resolution

### 2. Data Pipeline

- Use DataLoader with multiple workers
- Implement efficient data augmentation
- Cache preprocessed data

### 3. Inference Optimization

- Batch processing
- Model pruning
- Knowledge distillation

## 🔄 Continuous Learning

### 1. Active Learning

Implement active learning to select the most informative samples for labeling:

```python
# Select samples with low confidence predictions
uncertain_samples = select_uncertain_samples(model, unlabeled_data)
```

### 2. Online Learning

Update model with new data without full retraining:

```python
# Fine-tune on new data
model.fine_tune(new_data, epochs=10)
```

### 3. Model Monitoring

Track model performance over time:

```python
# Log metrics to Weights & Biases
wandb.log({
    'mAP50': mAP50,
    'precision': precision,
    'recall': recall
})
```

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Computer Vision Best Practices](https://github.com/microsoft/ComputerVision)
- [Data Augmentation Guide](https://albumentations.ai/)
- [Model Deployment Guide](https://pytorch.org/serve/)

## 🤝 Contributing

To contribute to the training pipeline:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


